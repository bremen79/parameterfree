# Copyright (c) 2026 Francesco Orabona
# Licensed under the MIT License. See LICENSE in the project root.

"""Native parameter precision, including accumulated state and conic solves."""

import copy
import unittest

import torch

from parameterfree import COCOB, KT, cKT, ConicFTRL, CoordinateWise
from parameterfree.conic import _solve_scale_tensor


FACTORIES = (COCOB, KT, cKT, ConicFTRL,
             lambda params: CoordinateWise(ConicFTRL, params))


class PrecisionTests(unittest.TestCase):
    def test_state_matches_parameter_dtype_through_steps_and_reload(self):
        for dtype in (torch.float32, torch.float64):
            for factory in FACTORIES:
                with self.subTest(dtype=dtype, factory=factory):
                    p = torch.nn.Parameter(torch.tensor([.1, -.2], dtype=dtype))
                    opt = factory([p])
                    for _ in range(4):
                        p.grad = torch.tensor([.01, -.02], dtype=dtype)
                        opt.step()
                    q = torch.nn.Parameter(p.detach().clone())
                    restored = factory([q])
                    restored.load_state_dict(copy.deepcopy(opt.state_dict()))
                    for params, optimizer in (([p], opt), ([q], restored)):
                        for value in params:
                            value.grad = torch.full_like(value, -.01)
                        optimizer.step()
                        state = optimizer.state[value] if isinstance(optimizer, COCOB) else optimizer.learner_state
                        for name, tensor in state.items():
                            self.assertEqual(tensor.dtype, torch.int64 if name == 't' else dtype)
                        self.assertEqual(value.grad.dtype, dtype)
                    torch.testing.assert_close(p, q, rtol=0, atol=0)

    def test_float32_predictions_track_float64_reference(self):
        # Check cumulative rounding over many rounds, including cancellation,
        # against the unchanged double-precision algorithms.
        for algorithm in (KT, ConicFTRL):
            for coordinatewise in (False, True):
                p = torch.nn.Parameter(torch.linspace(-.2, .2, 16))
                q = torch.nn.Parameter(p.detach().double())
                factory = (lambda params: CoordinateWise(algorithm, params)) if coordinatewise else algorithm
                low, high = factory([p]), factory([q])
                generator = torch.Generator().manual_seed(5)
                for _ in range(300):
                    gradient = torch.randn(16, generator=generator) * .01
                    p.grad, q.grad = gradient, gradient.double()
                    low.step()
                    high.step()
                torch.testing.assert_close(p.double(), q, rtol=2e-5, atol=2e-6)

    def test_float32_solver_original_stationarity_equation(self):
        generator = torch.Generator().manual_seed(13)
        a = 10 ** torch.empty(20000).uniform_(-8, 8, generator=generator)
        c = a * 10 ** torch.empty_like(a).uniform_(-8, 6, generator=generator)
        b = a * 10 ** torch.empty_like(a).uniform_(-8, 6, generator=generator)
        norm = (2 * c).sqrt()
        m = a.sqrt() * 10 ** torch.empty_like(a).uniform_(-3, 0, generator=generator)
        scale, height = _solve_scale_tensor(norm, m, a, b, 1., 1.)
        self.assertEqual(scale.dtype, torch.float32)
        self.assertEqual(height.dtype, torch.float32)
        # Evaluate the residual in double to distinguish solver error from
        # rounding in the test itself. Inputs and solve stay float32.
        x_norm = scale.double() * norm.double()
        residual = ((1 / height.double() - b.double()) + x_norm.square() / 2).abs() / b.double()
        self.assertLess(residual.max().item(), 2e-6)
        self.assertTrue(torch.isfinite(height).all())
        self.assertTrue((height > 0).all())
        self.assertTrue((x_norm * height.double() <= (1 / (2 * m.double())) * (1 + 2e-6)).all())

    def test_old_mixed_precision_checkpoint_rejected(self):
        p = torch.nn.Parameter(torch.ones(2))
        opt = KT([p])
        checkpoint = copy.deepcopy(opt.state_dict())
        checkpoint['algorithm']['version'] = 1
        checkpoint['state']['learner']['wealth'] = checkpoint['state']['learner']['wealth'].double()
        with self.assertRaisesRegex(ValueError, 'version'):
            opt.load_state_dict(checkpoint)
        self.assertEqual(opt.learner_state['wealth'].dtype, p.dtype)


if __name__ == '__main__':
    unittest.main()
