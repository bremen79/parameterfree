# Copyright (c) 2026 Francesco Orabona
# Licensed under the MIT License. See LICENSE in the project root.

"""COCOB compilation, validation, and checkpoint behavior."""

import copy
import unittest
from unittest.mock import patch

import torch

from parameterfree import COCOB


class COCOBTests(unittest.TestCase):
    def test_checkpoint_restores_configuration_and_lazy_parameter_state(self):
        for foreach in (False, True):
            for checked in (False, True):
                with self.subTest(foreach=foreach, error_if_nonfinite=checked):
                    points = [torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
                              for _ in range(2)]
                    original = COCOB(points, alpha=10, eps=0.05, foreach=foreach,
                                     error_if_nonfinite=checked)
                    for _ in range(3):
                        points[0].grad = torch.full_like(points[0], 0.2)
                        original.step()
                    checkpoint = copy.deepcopy(original.state_dict())
                    self.assertEqual(checkpoint['cocob'], dict(alpha=10, eps=0.05))
                    copies = [torch.nn.Parameter(p.detach().clone()) for p in points]
                    resumed = COCOB(copies, foreach=foreach, error_if_nonfinite=checked)
                    resumed.load_state_dict(checkpoint)
                    for _ in range(3):
                        for params, optimizer in ((points, original), (copies, resumed)):
                            for index, p in enumerate(params):
                                p.grad = torch.full_like(p, 0.2 if index == 0 else 0.001)
                            optimizer.step()
                        for p, q in zip(points, copies):
                            torch.testing.assert_close(p, q, rtol=0, atol=0)
                            for key, value in original.state[p].items():
                                torch.testing.assert_close(value, resumed.state[q][key], rtol=0, atol=0)

    def test_parameter_groups_match_independent_optimizers(self):
        for foreach in (False, True):
            for checked in (False, True):
                with self.subTest(foreach=foreach, error_if_nonfinite=checked):
                    points = [torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
                              for _ in range(3)]
                    copies = [torch.nn.Parameter(p.detach().clone()) for p in points]
                    groups = [dict(params=[p], lr=lr, weight_decay=0.1)
                              for p, lr in zip(points, (0., 0.25, 0.75))]
                    combined = COCOB(groups, foreach=foreach, error_if_nonfinite=checked)
                    separate = [COCOB([dict(params=[p], lr=group['lr'], weight_decay=0.1)],
                                      foreach=foreach, error_if_nonfinite=checked)
                                for p, group in zip(copies, groups)]
                    for step in range(5):
                        for index, (p, q) in enumerate(zip(points, copies)):
                            p.grad = torch.full_like(p, (step + 1) * (index + 1) * 0.1)
                            q.grad = p.grad.clone()
                        combined.step()
                        for optimizer in separate:
                            optimizer.step()
                        torch.testing.assert_close(points[0], torch.ones_like(points[0]), rtol=0, atol=0)
                        for p, q, optimizer in zip(points, copies, separate):
                            torch.testing.assert_close(p, q, rtol=0, atol=0)
                            for key, value in combined.state[p].items():
                                torch.testing.assert_close(value, optimizer.state[q][key], rtol=0, atol=0)

    def make_parameters(self):
        return [torch.nn.Parameter(torch.tensor([[1., -2.], [3., 4.]], dtype=torch.float64).T),
                torch.nn.Parameter(torch.tensor([0.5], dtype=torch.float64))]

    @unittest.skipUnless(hasattr(torch, 'compile'), 'torch.compile is unavailable')
    def test_compiled_trajectory_and_checkpoint_match_eager(self):
        compiler = torch.compile
        for checked in (False, True):
            with self.subTest(error_if_nonfinite=checked):
                actual, expected = self.make_parameters(), self.make_parameters()
                def construct(params, compile=False, **kwargs):
                    return COCOB([dict(params=[params[0]], lr=0.5, weight_decay=0.1),
                                  dict(params=[params[1]], lr=0.25)],
                                 error_if_nonfinite=checked, compile=compile, **kwargs)
                # Capture the constructor's real compilation path without requiring
                # a C++ toolchain in the unit suite. Inductor is also smoke-tested.
                with patch('torch.compile', side_effect=lambda fn, **kw:
                           compiler(fn, backend='eager', **kw)):
                    compiled = construct(actual, compile=True, alpha=10, eps=0.05)
                eager = construct(expected, alpha=10, eps=0.05)
                for step in range(5):
                    for index, (p, q) in enumerate(zip(actual, expected)):
                        grad = None if index == 1 and step == 2 else torch.full_like(p, (step - 1) * 0.1)
                        p.grad = grad
                        q.grad = None if grad is None else grad.clone()
                    saved_grads = [None if p.grad is None else p.grad.clone() for p in actual]
                    compiled.step()
                    eager.step()
                    for index, (p, q) in enumerate(zip(actual, expected)):
                        torch.testing.assert_close(p, q)
                        if saved_grads[index] is not None:
                            torch.testing.assert_close(p.grad, saved_grads[index], rtol=0, atol=0)
                        for key, value in eager.state[q].items():
                            torch.testing.assert_close(compiled.state[p][key], value)
                restored_params = [torch.nn.Parameter(p.detach().clone()) for p in actual]
                restored = construct(restored_params)
                restored.load_state_dict(copy.deepcopy(compiled.state_dict()))
                for params, opt in ((actual, compiled), (restored_params, restored)):
                    for p in params:
                        p.grad = torch.full_like(p, -0.05)
                    opt.step()
                for p, q in zip(actual, restored_params):
                    torch.testing.assert_close(p, q)

    def test_validation_rejects_entire_update_before_commit(self):
        params = self.make_parameters()
        opt = COCOB(params)
        for p in params:
            p.grad = torch.full_like(p, 0.1)
        opt.step()
        before_points = [p.detach().clone() for p in params]
        before_state = copy.deepcopy(opt.state_dict())
        params[1].grad.fill_(float('nan'))
        with self.assertRaisesRegex(RuntimeError, 'non-finite'):
            opt.step()
        for p, previous in zip(params, before_points):
            torch.testing.assert_close(p, previous, rtol=0, atol=0)
        for key, values in before_state['state'].items():
            for name, value in values.items():
                torch.testing.assert_close(opt.state_dict()['state'][key][name], value, rtol=0, atol=0)

    def test_no_gradients_and_closure(self):
        params = self.make_parameters()
        opt = COCOB(params)
        opt.step()
        self.assertEqual(len(opt.state), 0)
        def closure():
            opt.zero_grad()
            loss = sum(p.square().sum() for p in params)
            loss.backward()
            return loss
        loss = opt.step(closure)
        self.assertTrue(loss.requires_grad)
        self.assertTrue(all(p in opt.state for p in params))


if __name__ == '__main__':
    unittest.main()
