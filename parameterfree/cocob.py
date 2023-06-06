# Copyright (c) Francesco Orabona.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.optim.optimizer import Optimizer

class COCOB(Optimizer):
    r"""Implements COCOB algorithm.
    It has been proposed in `Training Deep Networks without Learning Rates Through Coin Betting`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        alpha (float, optional): It was proposed to increase the stability in the first iterations,
            similarly and independently to the learning rate warm-up. The number roughly denotes the
            number of rounds of warm-up (default 100)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    .. _Training Deep Networks without Learning Rates Through Coin Betting:
        https://arxiv.org/abs/1705.07795
    """

    def __init__(self, params, alpha: float = 100, eps: float = 1e-8, weight_decay: float = 0):
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))
        if not 0.0 <= weight_decay:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(weight_decay=weight_decay)
        self._alpha = alpha
        self._eps = eps

        super(COCOB, self).__init__(params, defaults)

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
        
        for group in self.param_groups:
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
                    state['L'] = self._eps*torch.ones_like(p).detach()
                    # Reward/wealth of the algorithm for each coordinate
                    state['reward'] = torch.zeros_like(p).detach()
                    # We need to save the initial point because this is a FTRL-based algorithm
                    state['x0'] = torch.clone(p.data).detach()

                sum_negative_gradients, grad_norm_sum, L, reward, x0 = (
                    state['sum_negative_gradients'],
                    state['grad_norm_sum'],
                    state['L'],
                    state['reward'],
                    state['x0'],
                )

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # update maximum rage of the gradients
                torch.max(L, torch.abs(grad), out=L)
                # udpate dual vector
                sum_negative_gradients.sub_(grad)
                # update sum of the absolute values of the gradients
                grad_norm_sum.add_(torch.abs(grad))
                # update the wealth
                reward.addcmul_(grad, p.data.sub(x0), value=-1)
                # reset the wealth to zero in case we lost all
                torch.maximum(reward, torch.zeros_like(reward), out=reward)
                # calculate denominator
                den = torch.maximum(grad_norm_sum.add(L), L.mul(self._alpha)).mul(L)
                # update model parameters
                p.data.copy_(reward.add(L).mul(sum_negative_gradients).div(den).add(x0))
                                
        return loss
