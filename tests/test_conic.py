# Copyright (c) 2026 Francesco Orabona
# Licensed under the MIT License. See LICENSE in the project root.

import copy
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from parameterfree import ConicFTRL
from parameterfree.conic import ConicAlgorithm


class ConicFTRLTests(unittest.TestCase):
    def test_compiler_detection_supports_older_pytorch_api(self):
        rule = ConicAlgorithm()
        point = torch.ones((1, 3), dtype=torch.float64)
        state = rule.init_state(point)
        gradient = torch.full_like(point, 0.1)
        expected_state, expected_point = rule.update(state, point, gradient)
        variants = (
            (None, None),  # PyTorch before torch.compile.
            (SimpleNamespace(), SimpleNamespace(is_compiling=lambda: False)),
            (SimpleNamespace(), SimpleNamespace(is_compiling=lambda: True)),
            (SimpleNamespace(is_compiling=lambda: True), None),
        )
        for compiler, dynamo in variants:
            with self.subTest(compiler=compiler, dynamo=dynamo):
                with patch.object(torch, 'compiler', compiler, create=True), \
                        patch.object(torch, '_dynamo', dynamo, create=True):
                    actual_state, actual_point = rule.update(state, point, gradient)
                torch.testing.assert_close(actual_point, expected_point)
                for key, value in expected_state.items():
                    torch.testing.assert_close(actual_state[key], value)

    def test_ftrl_optimality_and_statistics(self):
        # Check the original constrained objective, independently of the scalar
        # root equation used by the implementation (Section 8, equations 38-40).
        anchor = torch.tensor([2.0, -3.0], dtype=torch.float64)
        p = torch.nn.Parameter(anchor.clone())
        epsilon, gamma = 0.7, 0.3
        optimizer = ConicFTRL([p], epsilon=epsilon, gamma=gamma)
        theta = torch.zeros_like(p)
        m = v = delta = lifted_loss = 0.0
        saw_interior = saw_boundary = False
        gradients = [10.0] + [1.0] * 100 + [-3.0] * 35 + [100.0, -250.0]
        for value in gradients:
            grad = torch.tensor([value, value * 0.5], dtype=p.dtype)
            lifted_loss -= torch.dot(grad, p.detach() - anchor).item()
            norm = grad.norm().item()
            previous_m, m = m, max(m, norm)
            v += norm * norm
            delta += (m - previous_m) ** 2
            theta += grad
            p.grad = grad.clone()
            optimizer.step()

            state = {key: optimizer.learner_state[key].item() for key in
                     ('m', 'v', 'delta', 'lifted_loss', 'height')}
            for key, expected in (('m', m), ('v', v), ('delta', delta), ('lifted_loss', lifted_loss)):
                self.assertAlmostEqual(state[key], expected, delta=1e-9 * max(1, abs(expected)))
            torch.testing.assert_close(optimizer.learner_state['theta'][0], theta)
            height = state['height']
            self.assertGreater(height, 0)
            y = (p.detach() - anchor) * height
            radius = epsilon / (2 * m)
            a = gamma * epsilon + 4 * (v + m * m)
            scalar_derivative = (epsilon + lifted_loss + delta / (2 * gamma)
                                 - epsilon / height - gamma * y.square().sum().item() / (2 * height * height))
            self.assertAlmostEqual(scalar_derivative, 0, delta=1e-8)
            vector_derivative = (a / epsilon + gamma / height) * y + theta
            self.assertLessEqual(y.norm().item(), radius + 1e-12)
            if y.norm().item() < radius * (1 - 1e-8):
                saw_interior = True
                torch.testing.assert_close(vector_derivative, torch.zeros_like(y), atol=1e-8, rtol=0)
            else:
                saw_boundary = True
                multiplier = -torch.dot(vector_derivative, y) / y.square().sum()
                self.assertGreaterEqual(multiplier.item(), -1e-8)
                torch.testing.assert_close(vector_derivative + multiplier * y, torch.zeros_like(y), atol=1e-8, rtol=0)
        self.assertTrue(saw_interior)
        self.assertTrue(saw_boundary)

    def test_translation_and_parameter_partition_invariance(self):
        torch.manual_seed(3)
        anchor = torch.randn(6, dtype=torch.float64)
        shift = torch.tensor([12., -7., 5., 3., -2., 8.], dtype=torch.float64)
        whole = torch.nn.Parameter(anchor.clone())
        parts = [torch.nn.Parameter((anchor + shift)[:2].clone()),
                 torch.nn.Parameter((anchor + shift)[2:].clone())]
        old = ConicFTRL([whole], epsilon=0.2, gamma=2)
        new = ConicFTRL([{'params': [parts[0]]}, {'params': [parts[1]]}], epsilon=0.2, gamma=2)
        for _ in range(20):
            gradient = torch.randn_like(whole)
            whole.grad = gradient.clone()
            parts[0].grad, parts[1].grad = gradient[:2].clone(), gradient[2:].clone()
            old.step()
            new.step()
            torch.testing.assert_close(torch.cat(parts) - shift, whole.detach(), rtol=1e-12, atol=1e-12)

    def test_anchor_is_captured_at_construction(self):
        p = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
        optimizer = ConicFTRL([p])
        with torch.no_grad():
            p.fill_(3)
        p.grad = torch.ones_like(p)
        optimizer.step()
        torch.testing.assert_close(optimizer.learner_state['anchor'][0], torch.tensor([2.0], dtype=p.dtype))
        self.assertEqual(optimizer.learner_state['lifted_loss'].item(), -1.0)
        self.assertLess(p.item(), 2.0)

    def test_missing_components_match_zero_gradients(self):
        parameters = [torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64)) for _ in range(2)]
        vector = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
        optimizer, reference = ConicFTRL(parameters), ConicFTRL([vector])
        for gradients in ((1., 2.), (None, -1.), (None, 0.), (3., None)):
            for p, value in zip(parameters, gradients):
                p.grad = None if value is None else torch.full_like(p, value)
            vector.grad = torch.tensor([0 if g is None else g for g in gradients], dtype=vector.dtype)
            optimizer.step()
            reference.step()
            torch.testing.assert_close(torch.cat(parameters), vector.detach())

    def test_zero_rounds_leave_state_and_future_predictions_unchanged(self):
        p = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float64))
        optimizer = ConicFTRL([p])
        for gradients in (None, torch.zeros_like(p), torch.tensor([1., 2.], dtype=p.dtype)):
            p.grad = gradients
            optimizer.step()
        before = copy.deepcopy(optimizer.state_dict())
        value = p.detach().clone()
        for gradient in (torch.zeros_like(p), None):
            p.grad = gradient
            optimizer.step()
            torch.testing.assert_close(p.detach(), value, rtol=0, atol=0)
            for key, state_value in optimizer.learner_state.items():
                torch.testing.assert_close(state_value, before['state']['learner'][key], rtol=0, atol=0)

    def test_checkpoint_restores_anchor_hyperparameters_and_trajectory(self):
        p = torch.nn.Parameter(torch.tensor([2., -1.], dtype=torch.float64))
        optimizer = ConicFTRL([p], epsilon=0.2, gamma=3.0, weight_decay=0.1)
        for _ in range(4):
            p.grad = p.detach().clone()
            optimizer.step()
        q = torch.nn.Parameter(p.detach().clone())
        restored = ConicFTRL([q], epsilon=5, gamma=0.1)
        checkpoint = io.BytesIO()
        torch.save(optimizer.state_dict(), checkpoint)
        checkpoint.seek(0)
        restored.load_state_dict(torch.load(checkpoint))
        for _ in range(10):
            for param, opt in ((p, optimizer), (q, restored)):
                param.grad = param.detach().clone()
                opt.step()
            torch.testing.assert_close(p, q, rtol=0, atol=0)
            for key, value in optimizer.learner_state.items():
                torch.testing.assert_close(value, restored.learner_state[key], rtol=0, atol=0)

    def test_closure_weight_decay_and_schedule(self):
        p = torch.nn.Parameter(torch.tensor([2., -1.], dtype=torch.float64))
        q = torch.nn.Parameter(p.detach().clone())
        optimizer = ConicFTRL([p], weight_decay=0.2)
        reference = ConicFTRL([q])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
        for _ in range(5):
            def closure():
                optimizer.zero_grad()
                loss = p.square().sum() / 2
                loss.backward()
                return loss
            expected_loss = p.detach().square().sum() / 2
            expected_gradient = p.detach().clone()
            q.grad = q.detach().clone() * 1.2 * optimizer.param_groups[0]['lr']
            loss = optimizer.step(closure)
            reference.step()
            torch.testing.assert_close(loss, expected_loss)
            torch.testing.assert_close(p, q)
            torch.testing.assert_close(p.grad, expected_gradient)
            scheduler.step()

    def test_cancelling_gradients_return_to_anchor(self):
        p = torch.nn.Parameter(torch.tensor([2., -1.], dtype=torch.float64))
        anchor = p.detach().clone()
        optimizer = ConicFTRL([p])
        for grad in (torch.tensor([3., -4.], dtype=p.dtype),
                     torch.tensor([-3., 4.], dtype=p.dtype)):
            p.grad = grad
            optimizer.step()
        torch.testing.assert_close(p.detach(), anchor, rtol=0, atol=0)
        self.assertGreater(optimizer.learner_state['height'].item(), 0)

    def test_gradient_overflow_follows_parameter_precision(self):
        # A squared 1e20 gradient exceeds float32 range, as in native Adam.
        # Default validation must reject it before changing the prediction/state.
        p = torch.nn.Parameter(torch.ones(2))
        optimizer = ConicFTRL([p])
        p.grad = torch.ones_like(p)
        optimizer.step()
        before = p.detach().clone()
        p.grad = torch.full_like(p, 1e20)
        with self.assertRaisesRegex(RuntimeError, 'non-finite|precision'):
            optimizer.step()
        torch.testing.assert_close(p.detach(), before, rtol=0, atol=0)
        # Explicit float64 models retain the wider range; no hidden promotion.
        p64 = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
        optimizer64 = ConicFTRL([p64])
        for magnitude in (1e-6, 1., 1e20):
            p64.grad = torch.full_like(p64, magnitude)
            optimizer64.step()
            self.assertTrue(torch.isfinite(p64).all())
        self.assertGreater(optimizer64.learner_state['delta'].item(), 1e40)

    def test_rejects_invalid_inputs_without_changing_parameters(self):
        for name in ('epsilon', 'gamma'):
            for value in (0, -1, float('inf'), float('nan')):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    ConicFTRL([torch.nn.Parameter(torch.ones(1))], **{name: value})
        for value in (-1, float('inf'), float('nan')):
            with self.assertRaises(ValueError):
                ConicFTRL([torch.nn.Parameter(torch.ones(1))], weight_decay=value)
        p = torch.nn.Parameter(torch.ones(3))
        optimizer = ConicFTRL([p])
        for grad in (torch.sparse_coo_tensor([[0]], [1.], (3,)), torch.full_like(p, float('nan'))):
            p.grad = grad
            with self.assertRaises(RuntimeError):
                optimizer.step()
            torch.testing.assert_close(p.detach(), torch.ones_like(p))
            self.assertEqual(optimizer.learner_state['v'].item(), 0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is unavailable')
    def test_cuda_matches_cpu(self):
        cpu = torch.nn.Parameter(torch.tensor([1., -2., 3.], dtype=torch.float64))
        gpu = torch.nn.Parameter(cpu.detach().cuda())
        old, new = ConicFTRL([cpu]), ConicFTRL([gpu])
        for _ in range(10):
            for p, opt in ((cpu, old), (gpu, new)):
                p.grad = p.detach().clone() * 0.1
                opt.step()
            torch.testing.assert_close(cpu, gpu.cpu())


if __name__ == '__main__':
    unittest.main()
