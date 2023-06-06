# Copyright (c) Francesco Orabona.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.optim.optimizer import Optimizer

class KT(Optimizer):
    r"""Implements the KT algorithm.
    It has been proposed in `Coin Betting and Parameter-Free Online Learning`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        w (float, optional): Initial wealth (default 1e-4)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    .. _Coin Betting and Parameter-Free Online Learning:
        https://arxiv.org/abs/1602.04128
    """

    def __init__(self, params, w: float = 1e-4, weight_decay: float = 0):
        if not 0.0 <= w:
            raise ValueError("Invalid w value: {}".format(alpha))
        if not 0.0 <= weight_decay:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(weight_decay=weight_decay)
        self._wealth = w
        self._iter=1
        self._firstep = True

        super(KT, self).__init__(params, defaults)

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
            weight_decay = group['weight_decay']
            
            if self._firstep:
                x0 = group['x0'] = [torch.clone(p).detach() for p in group['params']]
                theta = group['theta'] = [torch.zeros_like(p).detach() for p in group['params']]
                self._firstep = False
            else:
                x0 = group['x0']
                theta = group['theta']
            
            gain = torch.stack([torch.dot((x-p.detach()).flatten(),p.grad.flatten()) for p,x in zip(group['params'],x0)]).sum().item()
            self._wealth += gain
            self._iter += 1
            
            if weight_decay > 0:
                for p in group['params']:
                    p.grad.add_(p, alpha=weight_decay)

            # update the sum of the negative gradients and the weights
            for p, t, x in zip(group['params'], theta, x0):
                if p.grad is None:
                    continue
                else:
                    t.add_(p.grad, alpha=-1)
                p.data.copy_(t.mul(self._wealth/self._iter).add(x))
                                
        return loss
