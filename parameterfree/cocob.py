# Copyright (c) 2023, 2024, 2026 Francesco Orabona
# Licensed under the MIT License. See LICENSE in the project root.

import torch
from torch.optim.optimizer import Optimizer

class COCOB(Optimizer):
    r"""Implements the COCOB algorithm.
    It has been proposed in `Training Deep Networks without Learning Rates Through Coin Betting`_.
    
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        alpha (float, optional): It was proposed to increase the stability in the first iterations,
            similarly and independently to the learning rate warm-up. The number roughly denotes the
            number of rounds of warm-up (default 100)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        compile (bool, optional): fuse tensor updates with torch.compile (default: False)
        error_if_nonfinite (bool, optional): validate gradients and candidate state
            before committing an update (default: True). On CUDA this synchronizes
            once per step; disable to omit these checks.
        foreach (bool, optional): use multi-tensor operations. Default None selects
            them on CUDA for parameters of one dtype/device. True forces this
            path, False keeps per-tensor operations. Checkpoints are unchanged.
    .. _Training Deep Networks without Learning Rates Through Coin Betting:
        https://arxiv.org/abs/1705.07795
    """

    def __init__(self, params, alpha: float = 100, eps: float = 1e-8, weight_decay: float = 0,
                 error_if_nonfinite: bool = True, compile: bool = False, foreach=None):
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))
        if not 0.0 <= weight_decay:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(weight_decay=weight_decay, lr=1.0)
        self._alpha = alpha
        self._eps = eps
        self.error_if_nonfinite = error_if_nonfinite
        self._compile = compile
        self._update = self._functional_step
        if compile:
            if not hasattr(torch, 'compile'):
                raise ValueError('compile=True requires torch.compile support')
            self._update = torch.compile(self._update, fullgraph=True)

        super(COCOB, self).__init__(params, defaults)
        if foreach is not None and not isinstance(foreach, bool):
            raise ValueError('foreach must be True, False, or None')
        self.foreach = foreach
        parameters = [p for group in self.param_groups for p in group['params']]
        homogeneous = len({(p.device, p.dtype) for p in parameters}) == 1
        if foreach and not homogeneous:
            raise ValueError('COCOB foreach=True requires one parameter dtype and device')
        available = all(hasattr(torch, name) for name in (
            '_foreach_abs', '_foreach_maximum', '_foreach_addcmul', '_foreach_copy_'))
        if foreach and not available:
            raise ValueError('foreach=True requires PyTorch tensor-list operators')
        self._foreach = (homogeneous and available and all(p.is_cuda for p in parameters)) if foreach is None else foreach

    def state_dict(self):
        state = super().state_dict()
        state['cocob'] = dict(alpha=self._alpha, eps=self._eps)
        return state

    def load_state_dict(self, state_dict):
        # Old checkpoints omitted these options. In that case retain the
        # constructor values; the original values cannot be recovered.
        config = state_dict.get('cocob', dict(alpha=self._alpha, eps=self._eps))
        alpha, eps = config['alpha'], config['eps']
        if not 0.0 <= alpha or not 0.0 <= eps:
            raise ValueError('Invalid COCOB checkpoint configuration')
        super().load_state_dict(state_dict)
        self._alpha, self._eps = alpha, eps

    @torch.no_grad()
    def step(self, closure = None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        points, gradients, states = [], [], []
        
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('COCOB does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    # Sum of the negative gradients
                    state['sum_negative_gradients'] = torch.zeros_like(p).detach()
                    # Sum of the absolute values of the stochastic subgradients
                    state['grad_norm_sum'] = torch.zeros_like(p).detach()
                    # Maximum observed scale
                    state['L'] = torch.full_like(p, self._eps)
                    # Reward/wealth of the algorithm for each coordinate
                    state['reward'] = torch.zeros_like(p).detach()
                    # We need to save the initial point because this is a FTRL-based algorithm
                    state['x0'] = torch.clone(p.data).detach()

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])
                    
                if lr != 1.0:
                    grad = grad * lr

                points.append(p)
                gradients.append(grad)
                states.append(state)

        if not points:
            return loss
        candidates, predictions, valid = self._update(points, gradients, states)
        if valid is not None:
            if not bool(valid):
                raise RuntimeError('COCOB received non-finite gradients or invalid state')
            for point, prediction in zip(points, predictions):
                point.copy_(prediction)
        for state, candidate in zip(states, candidates):
            state.update(candidate)
        return loss

    def _functional_step(self, points, gradients, states):
        if self._foreach:
            return self._foreach_step(points, gradients, states)
        # Compile all active tensors together, leaving closures and lazy state
        # initialization outside the graph. Preserve the existing checkpoint keys.
        candidates, predictions = [], []
        valid = None
        for point, gradient, state in zip(points, gradients, states):
            abs_grad = gradient.abs()
            scale = torch.maximum(state['L'], abs_grad)
            negative_sum = state['sum_negative_gradients'] - gradient
            norm_sum = state['grad_norm_sum'] + abs_grad
            reward = torch.addcmul(state['reward'], gradient,
                                   point - state['x0'], value=-1).clamp_min(0)
            denominator = torch.maximum(norm_sum + scale, scale * self._alpha) * scale
            prediction = (reward + scale) * negative_sum / denominator + state['x0']
            candidate = dict(sum_negative_gradients=negative_sum,
                             grad_norm_sum=norm_sum, L=scale, reward=reward,
                             x0=state['x0'])
            if self.error_if_nonfinite:
                check = torch.isfinite(gradient).all() & torch.isfinite(prediction).all()
                for value in candidate.values():
                    check = check & torch.isfinite(value).all()
                valid = check if valid is None else valid & check
                predictions.append(prediction)
                candidates.append(candidate)
            else:
                # Keep state storage stable and fuse writes with the update.
                # No model-sized candidates escape the unchecked compiled graph.
                for key in ('sum_negative_gradients', 'grad_norm_sum', 'L', 'reward'):
                    state[key].copy_(candidate[key])
                point.copy_(prediction)
        return candidates, predictions, valid

    def _foreach_step(self, points, gradients, states):
        anchors = [state['x0'] for state in states]
        absolute_gradients = torch._foreach_abs(gradients)
        scales = torch._foreach_maximum([state['L'] for state in states], absolute_gradients)
        negative_sums = torch._foreach_sub([state['sum_negative_gradients'] for state in states], gradients)
        norm_sums = torch._foreach_add([state['grad_norm_sum'] for state in states], absolute_gradients)
        rewards = torch._foreach_addcmul([state['reward'] for state in states], gradients,
                                        torch._foreach_sub(points, anchors), value=-1)
        rewards = torch._foreach_maximum(rewards, 0)
        denominators = torch._foreach_mul(torch._foreach_maximum(
            torch._foreach_add(norm_sums, scales), torch._foreach_mul(scales, self._alpha)), scales)
        predictions = torch._foreach_add(torch._foreach_div(torch._foreach_mul(
            torch._foreach_add(rewards, scales), negative_sums), denominators), anchors)
        candidates = [dict(sum_negative_gradients=negative_sum, grad_norm_sum=norm_sum,
                           L=scale, reward=reward, x0=anchor)
                      for negative_sum, norm_sum, scale, reward, anchor in
                      zip(negative_sums, norm_sums, scales, rewards, anchors)]
        valid = None
        if self.error_if_nonfinite:
            for gradient, prediction, candidate in zip(gradients, predictions, candidates):
                check = torch.isfinite(gradient).all() & torch.isfinite(prediction).all()
                for value in candidate.values():
                    check = check & torch.isfinite(value).all()
                valid = check if valid is None else valid & check
            return candidates, predictions, valid
        destinations, sources = list(points), list(predictions)
        for key in ('sum_negative_gradients', 'grad_norm_sum', 'L', 'reward'):
            destinations.extend(state[key] for state in states)
            sources.extend(candidate[key] for candidate in candidates)
        torch._foreach_copy_(destinations, sources)
        return [], [], None
