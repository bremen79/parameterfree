# Copyright (c) 2023, 2024, 2026 Francesco Orabona
# Licensed under the MIT License. See LICENSE in the project root.

import math

import torch

from .base import BatchedAlgorithm, BatchedOptimizer


class KTAlgorithm(BatchedAlgorithm):
    """KT betting rule; reductions are over the last coordinate dimension."""

    supports_tensor_lists = True
    skip_zero_gradients = True
    vector_point_state_keys = ('anchor', 'theta')

    def __init__(self, w=1e-4):
        if not math.isfinite(w) or w < 0:
            raise ValueError('Invalid w value: {}'.format(w))
        self.config = dict(w=float(w))

    def init_state(self, initial_point):
        return dict(
            anchor=initial_point.clone(), theta=torch.zeros_like(initial_point),
            wealth=torch.full((initial_point.shape[0],), self.config['w'],
                              dtype=initial_point.dtype, device=initial_point.device),
            t=torch.ones(initial_point.shape[0], dtype=torch.int64, device=initial_point.device))

    def update(self, state, point, gradient):
        theta = state['theta'] + gradient
        gain = (gradient * (point - state['anchor'])).sum(dim=-1)
        wealth = state['wealth'] - gain
        t = state['t'] + 1
        scale = (wealth * t.to(wealth.dtype).reciprocal()).unsqueeze(-1)
        prediction = state['anchor'] - scale * theta
        return dict(anchor=state['anchor'], theta=theta, wealth=wealth, t=t), prediction

    def update_vector_blocks(self, states, points, gradients):
        return self._update_vector_blocks(states, points, gradients, foreach=False)

    def update_vector_blocks_foreach(self, states, points, gradients):
        return self._update_vector_blocks(states, points, gradients, foreach=True)

    def _update_vector_blocks(self, states, points, gradients, foreach):
        # Only the gain is global. Read the original tensors for that reduction
        # and update every tensor directly using the same shared wealth/clock.
        if foreach:
            displacements = torch._foreach_sub(points, [state['anchor'] for state in states])
            gains = [value.sum() for value in torch._foreach_mul(gradients, displacements)]
        else:
            gains = [(gradient * (point - state['anchor'])).sum()
                     for state, point, gradient in zip(states, points, gradients)]
        gain = sum(gains[1:], gains[0])
        active_blocks = [gradient.ne(0).any() for gradient in gradients]
        active = torch.stack(active_blocks).any()
        wealth = states[0]['wealth'] - gain
        t = states[0]['t'] + active.to(torch.int64)
        scale = (wealth * t.to(wealth.dtype).reciprocal()).unsqueeze(-1)
        if foreach:
            anchors = [state['anchor'] for state in states]
            thetas = torch._foreach_add([state['theta'] for state in states], gradients)
            predictions = torch._foreach_sub(anchors, torch._foreach_mul(thetas, scale.reshape(())))
            predictions = [torch.where(active, prediction, point)
                           for prediction, point in zip(predictions, points)]
            candidates = [dict(anchor=anchor, theta=theta, wealth=wealth, t=t)
                          for anchor, theta in zip(anchors, thetas)]
            return candidates, predictions
        candidates, predictions = [], []
        for state, point, gradient in zip(states, points, gradients):
            theta = state['theta'] + gradient
            candidates.append(dict(anchor=state['anchor'], theta=theta, wealth=wealth, t=t))
            predictions.append(torch.where(active, state['anchor'] - scale * theta, point))
        return candidates, predictions

    def update_coordinate_blocks(self, states, points, gradients):
        # Tensor-list operations expose independent pointwise updates to
        # horizontal fusion across the original model tensors.
        anchors = [state['anchor'] for state in states]
        thetas = torch._foreach_add([state['theta'] for state in states], gradients)
        displacements = torch._foreach_sub(points, anchors)
        gains = torch._foreach_mul(gradients, displacements)
        wealths = torch._foreach_sub([state['wealth'].unsqueeze(-1) for state in states], gains)
        clocks = torch._foreach_add([state['t'] for state in states], 1)
        reciprocals = [t.to(point.dtype).reciprocal().unsqueeze(-1).expand_as(point)
                       for t, point in zip(clocks, points)]
        scales = torch._foreach_mul(wealths, reciprocals)
        predictions = torch._foreach_sub(anchors, torch._foreach_mul(scales, thetas))
        candidates = [dict(anchor=anchor, theta=theta, wealth=wealth.squeeze(-1), t=t)
                      for anchor, theta, wealth, t in zip(anchors, thetas, wealths, clocks)]
        return candidates, predictions


class KT(BatchedOptimizer):
    """Euclidean KT coin betting, with a shared wealth and round counter.

    Based on Orabona and Pal, Coin Betting and Parameter-Free Online Learning
    (https://arxiv.org/abs/1602.04128). Requires effective gradient L2 norm <= 1.
    ``w`` is initial wealth (default 1e-4). All parameters share one learner;
    initial network weights are the anchor. An entirely zero effective gradient
    leaves parameters, wealth, and the counter unchanged. Missing components
    are zero feedback in this joint learner; an entirely missing gradient is
    also a no-op. max_grad_norm optionally
    clips raw gradients in the update (L2 norm), before decay and lr scaling;
    default None leaves clipping to the caller. Use max_grad_norm=1.0 for the
    unit raw-gradient bound when lr <= 1 and weight_decay == 0. foreach=None
    selects multi-tensor operations on CUDA; False preserves per-tensor execution.
    """

    algorithm_type = KTAlgorithm

    def __init__(self, params, w=1e-4, weight_decay=0.0,
                 error_if_nonfinite=True, compile=False, max_grad_norm=None, foreach=None):
        super().__init__(params, KTAlgorithm(w), weight_decay=weight_decay,
                         error_if_nonfinite=error_if_nonfinite, compile=compile, max_grad_norm=max_grad_norm, foreach=foreach)
