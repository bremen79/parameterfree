# Copyright (c) 2026 Francesco Orabona
# Licensed under the MIT License. See LICENSE in the project root.

import unittest

import torch

from parameterfree.conic import _solve_scale, _solve_scale_tensor


class ConicSolverTests(unittest.TestCase):
    def test_stationarity_over_wide_scales(self):
        # Check the original stationarity equation (52), not the transformed
        # quadratic bounds or Newton recurrence used by the fast solver.
        generator = torch.Generator().manual_seed(8)
        A = 10 ** torch.empty(2000, dtype=torch.float64).uniform_(-80, 80, generator=generator)
        C = A * 10 ** torch.empty_like(A).uniform_(-16, 12, generator=generator)
        b = A * 10 ** torch.empty_like(A).uniform_(-80, 12, generator=generator)
        for gamma, epsilon in ((1e-6, 1e-4), (1., 1.), (1e6, 1e4)):
            a = A * gamma
            norm = (2 * gamma * C).sqrt()
            m = a.sqrt() * 10 ** torch.empty_like(A).uniform_(-6, 0, generator=generator)
            scale, height = _solve_scale_tensor(norm, m, a, b, epsilon, gamma)
            p = epsilon / height
            x_norm = scale * norm
            relative_residual = ((p - b) + gamma * x_norm.square() / 2).abs() / b
            self.assertLess(relative_residual.max().item(), 2e-12)
            self.assertTrue((height > 0).all())
            self.assertTrue(torch.isfinite(height).all())
            radius = epsilon / (2 * m)
            y_norm = x_norm * height
            self.assertTrue((y_norm <= radius * (1 + 2e-12)).all())
            derivative = norm - (a / epsilon + gamma / height) * y_norm
            interior = y_norm < radius * (1 - 1e-10)
            self.assertLess((derivative[interior].abs() / norm[interior]).max().item(), 2e-12)
            self.assertTrue((derivative[~interior] >= -norm[~interior] * 2e-12).all())

    def test_host_and_tensor_solvers_agree(self):
        torch.manual_seed(19)
        norm = torch.rand(100, dtype=torch.float64) * 30
        m = torch.rand_like(norm) + .01
        a = torch.rand_like(norm) * 30 + 1
        b = torch.rand_like(norm) * 30 + .01
        scale, height = _solve_scale_tensor(norm, m, a, b, 0.7, 0.3)
        for index in range(len(norm)):
            expected = _solve_scale(norm[index].item(), m[index].item(), a[index].item(),
                                    b[index].item(), 0.7, 0.3)
            self.assertAlmostEqual(scale[index].item(), expected[0], delta=abs(expected[0]) * 2e-12)
            self.assertAlmostEqual(height[index].item(), expected[1], delta=abs(expected[1]) * 2e-12)

    def test_zero_direction_and_rounded_bracket_endpoint(self):
        a = torch.tensor([1., 6.727220014866174e29, 3.7811074250287495e69], dtype=torch.float64)
        b = torch.tensor([1., 5.856515284698199e29, 1.724481883744842e62], dtype=torch.float64)
        norm = torch.tensor([0., (2 * 892619540615984.4) ** .5,
                             (2 * 1.1729245187615016e57) ** .5], dtype=torch.float64)
        m = torch.tensor([0., 506221433593433.3, 2.7466009335945114e32], dtype=torch.float64)
        scale, height = _solve_scale_tensor(norm, m, a, b, 1., 1.)
        self.assertEqual(scale[0].item(), 0)
        self.assertEqual(height[0].item(), 1)
        residual = ((1 / height - b) + (scale * norm).square() / 2).abs() / b
        self.assertLess(residual.max().item(), 2e-14)


if __name__ == '__main__':
    unittest.main()
