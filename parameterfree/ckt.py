# Copyright (c) Francesco Orabona.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.optim.optimizer import Optimizer

class cKT(Optimizer):
    r"""Implements the coordinate-wise KT algorithm.
    It has been proposed in Section 9.3 of `A Modern Introduction To Online Learning'_.
    
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    .. _A Modern Introduction to Online Learning:
        https://arxiv.org/abs/1912.13213
    .. _Coin Betting and Parameter-Free Online Learning:
        https://arxiv.org/abs/1602.04128
    """

    def __init__(self, params, w: float = 1e-4, weight_decay: float = 0):
        if not 0.0 <= weight_decay:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))
        
        self._wealth = w
        defaults = dict(weight_decay=weight_decay, lr=1.0)

        super(cKT, self).__init__(params, defaults)

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
        
        lr = max(group['lr'] for group in self.param_groups)
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('cKT does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    # Sum of the negative gradients
                    state['sum_negative_gradients'] = torch.zeros_like(p).detach()
                    # Sum of the absolute values of the stochastic subgradients
                    state['reward'] = self._wealth*torch.ones_like(p).detach()
                    # Number of updates for each weight
                    state['t'] = torch.ones_like(p).detach()                    
                    # We need to save the initial point because this is a FTRL-based algorithm
                    state['x0'] = torch.clone(p.data).detach()

                sum_negative_gradients, reward, t, x0 = (
                    state['sum_negative_gradients'],
                    state['reward'],
                    state['t'],
                    state['x0'],
                )

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])
                
                grad=grad*lr
                
                # udpate dual vector
                sum_negative_gradients.sub_(grad)
                # update the wealth
                reward.addcmul_(grad, p.data.sub(x0), value=-1)
                # update the number of updates made on each weight
                t.add_(torch.logical_not(torch.eq(grad, 0)).float())
                # update model parameters
                p.data.copy_(reward.mul(sum_negative_gradients).div(t).add(x0))
                                
        return loss
