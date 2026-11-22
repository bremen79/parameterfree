# Copyright (c) 2026 Francesco Orabona
# Licensed under the MIT License. See LICENSE in the project root.

import copy
import unittest

import torch

from parameterfree import COCOB, KT, cKT, ConicFTRL


class OptimizerTests(unittest.TestCase):
    def test_quadratic_optimization(self):
        for optimizer_type in (COCOB, KT, cKT, ConicFTRL):
            with self.subTest(optimizer=optimizer_type.__name__):
                parameter = torch.nn.Parameter(torch.tensor([1.0]))
                optimizer = optimizer_type([parameter])
                for _ in range(5):
                    optimizer.zero_grad()
                    loss = 0.1 * parameter.square().sum()
                    loss.backward()
                    optimizer.step()
                self.assertTrue(torch.isfinite(parameter).all())
                self.assertLess(parameter.square().item(), 1.0)

    def test_kt_rejects_negative_wealth(self):
        with self.assertRaisesRegex(ValueError, 'Invalid w value: -1'):
            KT([torch.nn.Parameter(torch.ones(1))], w=-1)

    def test_kt_skips_missing_gradients(self):
        for weight_decay in (0.0, 0.1):
            with self.subTest(weight_decay=weight_decay):
                active = torch.nn.Parameter(torch.ones(1))
                inactive = torch.nn.Parameter(torch.ones(1))
                reference = torch.nn.Parameter(torch.ones(1))
                optimizer = KT([active, inactive], weight_decay=weight_decay)
                reference_optimizer = KT([reference], weight_decay=weight_decay)
                for _ in range(3):
                    active.grad = active.detach().clone() * 0.1
                    reference.grad = reference.detach().clone() * 0.1
                    optimizer.step()
                    reference_optimizer.step()
                torch.testing.assert_close(active, reference)
                torch.testing.assert_close(inactive, torch.ones(1))
                self.assertIsNone(inactive.grad)

    def test_kt_no_gradients_is_noop(self):
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = KT([parameter], weight_decay=0.1)
        optimizer.step()
        torch.testing.assert_close(parameter, torch.ones(1))
        self.assertEqual(optimizer.learner_state['t'].item(), 1)
        self.assertAlmostEqual(optimizer.learner_state['wealth'].item(), 1e-4)

    def test_checkpoint_continuation(self):
        for optimizer_type in (COCOB, KT, cKT, ConicFTRL):
            with self.subTest(optimizer=optimizer_type.__name__):
                parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
                optimizer = optimizer_type([parameter])
                for _ in range(3):
                    parameter.grad = parameter.detach().clone() * 0.1
                    optimizer.step()

                restored_parameter = torch.nn.Parameter(parameter.detach().clone())
                restored = optimizer_type([restored_parameter])
                restored.load_state_dict(copy.deepcopy(optimizer.state_dict()))
                for _ in range(5):
                    for value, opt in ((parameter, optimizer), (restored_parameter, restored)):
                        value.grad = value.detach().clone() * 0.1
                        opt.step()
                    torch.testing.assert_close(parameter, restored_parameter, rtol=0, atol=0)

    def test_kt_checkpoint_before_first_step(self):
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = KT([parameter], w=0.2)
        restored = KT([parameter])
        restored.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        parameter.grad = torch.ones_like(parameter) * 0.1
        restored.step()
        torch.testing.assert_close(parameter, torch.tensor([0.99]))

    def test_kt_rejects_older_checkpoints(self):
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = KT([parameter])
        parameter.grad = torch.ones_like(parameter) * 0.1
        optimizer.step()
        older_checkpoint = copy.deepcopy(optimizer.state_dict())
        del older_checkpoint['algorithm']

        restored = KT([torch.nn.Parameter(torch.ones(1))])
        with self.assertRaisesRegex(ValueError, 'older checkpoints are unsupported'):
            restored.load_state_dict(older_checkpoint)
        self.assertEqual(restored.learner_state['t'].item(), 1)
        self.assertNotIn('x0', restored.param_groups[0])

    def test_kt_rejects_sparse_gradients(self):
        parameter = torch.nn.Parameter(torch.ones(3))
        parameter.grad = torch.sparse_coo_tensor([[0]], [0.1], (3,))
        with self.assertRaisesRegex(RuntimeError, 'sparse gradients'):
            KT([parameter]).step()


if __name__ == '__main__':
    unittest.main()
