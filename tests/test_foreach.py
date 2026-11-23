# Copyright (c) 2026 Francesco Orabona
# Licensed under the MIT License. See LICENSE in the project root.

"""Multi-tensor updates must preserve the ordinary execution semantics."""

import copy
import unittest
from unittest.mock import patch

import torch

from parameterfree import COCOB, KT, cKT, CoordinateWise


class ForeachTests(unittest.TestCase):
    def parameters(self, dtype=torch.float64):
        return [torch.nn.Parameter(torch.tensor([[1., -2.], [3., 4.]], dtype=dtype).T),
                torch.nn.Parameter(torch.tensor(.5, dtype=dtype))]

    def compare(self, factory, dtype, checked, compiled=False):
        values = self.parameters(dtype)
        refs = [torch.nn.Parameter(p.detach().clone()) for p in values]
        def make(params, **kwargs):
            groups = [dict(params=[params[0]], weight_decay=.1, lr=.5),
                      dict(params=[params[1]], lr=.7)]
            options = {} if factory is COCOB else dict(max_grad_norm=1.)
            return factory(groups, error_if_nonfinite=checked, **options, **kwargs)
        actual = make(values, foreach=True, compile=compiled)
        reference = make(refs, foreach=False)
        for step in range(7):
            # Cover changing scheduler multipliers, all-missing rounds, zero
            # feedback, and missing parameters following an earlier update.
            for opt in (actual, reference):
                opt.param_groups[0]['lr'] = (1., .2, 0.)[step % 3]
            for index, (p, q) in enumerate(zip(values, refs)):
                grad = (None if step == 4 or (index == 1 and step == 2) else
                        torch.full_like(p, (0., 10., -.1)[step % 3]))
                p.grad = grad
                q.grad = None if grad is None else grad.clone()
            saved = [None if p.grad is None else p.grad.clone() for p in values]
            actual.step()
            reference.step()
            for p, q, gradient in zip(values, refs, saved):
                torch.testing.assert_close(p, q)
                if gradient is not None:
                    torch.testing.assert_close(p.grad, gradient, rtol=0, atol=0)
            old_state, new_state = reference.state_dict()['state'], actual.state_dict()['state']
            for key in old_state:
                for name, tensor in old_state[key].items():
                    torch.testing.assert_close(tensor, new_state[key][name])
        # The foreach flag does not enter checkpoint metadata: either execution
        # path must be able to resume the same mathematical state.
        copied = [torch.nn.Parameter(p.detach().clone()) for p in values]
        restored = make(copied, foreach=False)
        restored.load_state_dict(copy.deepcopy(actual.state_dict()))
        for params, opt in ((values, actual), (copied, restored)):
            for p in params:
                p.grad = torch.full_like(p, -.1)
            opt.step()
        for p, q in zip(values, copied):
            torch.testing.assert_close(p, q)

    def test_matches_per_tensor_updates_and_checkpoints(self):
        for factory in (COCOB, KT, cKT, lambda params, **kwargs: CoordinateWise(KT, params, **kwargs)):
            for dtype in (torch.float32, torch.float64):
                for checked in (True, False):
                    with self.subTest(factory=factory, dtype=dtype, checked=checked):
                        self.compare(factory, dtype, checked)

    @unittest.skipUnless(hasattr(torch, 'compile'), 'torch.compile is unavailable')
    def test_fullgraph_capture(self):
        compiler = torch.compile
        for factory in (COCOB, KT, cKT):
            for checked in (True, False):
                with self.subTest(factory=factory, checked=checked):
                    torch._dynamo.reset()
                    with patch('torch.compile', side_effect=lambda fn, **kw: compiler(fn, backend='eager', **kw)):
                        self.compare(factory, torch.float64, checked, compiled=True)

    def test_nonfinite_updates_are_rejected_before_commit(self):
        for factory in (COCOB, KT, cKT):
            params = self.parameters()
            opt = factory(params, foreach=True)
            for p in params:
                p.grad = torch.full_like(p, .1)
            opt.step()
            before = copy.deepcopy(opt.state_dict())
            points = [p.detach().clone() for p in params]
            params[1].grad.fill_(float('nan'))
            with self.assertRaisesRegex(RuntimeError, 'non-finite'):
                opt.step()
            for p, old in zip(params, points):
                torch.testing.assert_close(p, old, rtol=0, atol=0)
            for key, state in before['state'].items():
                for name, tensor in state.items():
                    torch.testing.assert_close(tensor, opt.state_dict()['state'][key][name], rtol=0, atol=0)

    def test_nonuniform_coordinate_clocks(self):
        p = torch.nn.Parameter(torch.ones(3))
        opt = cKT([p], foreach=True)
        state = opt.state_dict()
        state['state']['learner']['t'] = torch.tensor([2, 5, 8])
        opt.load_state_dict(state)
        p.grad = torch.tensor([.1, 0., -.2])
        opt.step()
        torch.testing.assert_close(opt.learner_state['t'], torch.tensor([3, 5, 9]))


if __name__ == '__main__':
    unittest.main()
