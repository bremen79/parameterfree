# Copyright (c) 2026 Francesco Orabona
# Licensed under the MIT License. See LICENSE in the project root.

import copy
import io
import unittest

import torch

from parameterfree import BatchedAlgorithm, BatchedOptimizer, CoordinateWise, KT, cKT, ConicFTRL


class MomentumRule(BatchedAlgorithm):
    """A third-party rule: no registry entry or wrapper changes required."""

    def __init__(self, rate=0.1, beta=0.5):
        self.config = dict(rate=rate, beta=beta)

    def init_state(self, initial_point):
        return {'momentum': {'velocity': torch.zeros_like(initial_point)}}

    def update(self, state, point, gradient):
        velocity = self.config['beta'] * state['momentum']['velocity'] + gradient
        return {'momentum': {'velocity': velocity}}, point - self.config['rate'] * velocity


class CoordinateWiseTests(unittest.TestCase):
    def test_matches_independent_scalar_optimizers(self):
        for algorithm, options in ((KT, {'w': 0.2}),
                                   (ConicFTRL, {'epsilon': 0.3, 'gamma': 2.0})):
            for dtype in (torch.float32, torch.float64):
                with self.subTest(algorithm=algorithm.__name__, dtype=dtype):
                    torch.manual_seed(10)
                    parameters = [torch.nn.Parameter(torch.randn(2, 2, dtype=dtype).T),
                                  torch.nn.Parameter(torch.tensor(2., dtype=dtype))]
                    optimizer = CoordinateWise(algorithm, [
                        {'params': [parameters[0]], 'weight_decay': 0.02},
                        {'params': [parameters[1]], 'weight_decay': 0.01},
                    ], **options)
                    scalar_params = [torch.nn.Parameter(value.detach().clone())
                                     for p in parameters for value in p.reshape(-1)]
                    scalar_opts = [algorithm([value], weight_decay=0.02 if index < 4 else 0.01, **options)
                                   for index, value in enumerate(scalar_params)]
                    for step in range(20):
                        offset = 0
                        for group in optimizer.param_groups:
                            p = group['params'][0]
                            lr = (1., 0.7, 0., 0.1)[step % 4]
                            group['lr'] = lr
                            p.grad = None if step % 5 == 2 else torch.randn_like(p) * 0.05
                            if p.grad is not None and step % 3 == 0:
                                p.grad.zero_()
                            for local in range(p.numel()):
                                index = offset + local
                                scalar_opts[index].param_groups[0]['lr'] = lr
                                scalar_params[index].grad = (None if p.grad is None else
                                                            p.grad.reshape(-1)[local].clone())
                            offset += p.numel()
                        optimizer.step()
                        for scalar in scalar_opts:
                            scalar.step()
                        actual = torch.cat([p.detach().reshape(-1) for p in parameters])
                        expected = torch.stack([p.detach() for p in scalar_params])
                        torch.testing.assert_close(actual, expected)
                        for index, scalar in enumerate(scalar_opts):
                            for key, value in optimizer.learner_state.items():
                                torch.testing.assert_close(value[index:index + 1], scalar.learner_state[key])

    def test_zero_round_policies_and_missing_gradients(self):
        for algorithm in (KT, ConicFTRL):
            p = torch.nn.Parameter(torch.tensor([1., 2.], dtype=torch.float64))
            optimizer = CoordinateWise(algorithm, [p])
            p.grad = torch.tensor([0.1, -0.2], dtype=p.dtype)
            optimizer.step()
            before = copy.deepcopy(optimizer.learner_state)
            point = p.detach().clone()
            p.grad = None
            optimizer.step()
            torch.testing.assert_close(p, point, rtol=0, atol=0)
            for key, value in before.items():
                torch.testing.assert_close(value, optimizer.learner_state[key], rtol=0, atol=0)
            p.grad = torch.zeros_like(p)
            optimizer.step()
            torch.testing.assert_close(p, point, rtol=0, atol=0)
            for key, value in before.items():
                torch.testing.assert_close(value, optimizer.learner_state[key], rtol=0, atol=0)

    def test_conic_coordinates_cover_constraint_and_scale_jumps(self):
        anchor = torch.tensor([2., -3.], dtype=torch.float64)
        p = torch.nn.Parameter(anchor.clone())
        optimizer = CoordinateWise(ConicFTRL, [p], epsilon=0.7, gamma=0.3)
        scalars = [torch.nn.Parameter(value.clone()) for value in anchor]
        references = [ConicFTRL([value], epsilon=0.7, gamma=0.3) for value in scalars]
        saw_boundary = False
        for value in [10.] + [1.] * 100 + [-3.] * 35 + [100., -250., 0.]:
            p.grad = torch.tensor([value, value * 0.5], dtype=p.dtype)
            optimizer.step()
            for index, reference in enumerate(references):
                scalars[index].grad = p.grad[index].clone()
                reference.step()
            torch.testing.assert_close(p, torch.stack(scalars), rtol=1e-11, atol=1e-11)
            state = optimizer.learner_state
            y = (p.detach() - anchor) * state['height']
            radius = 0.7 / (2 * state['m'])
            self.assertTrue((y.abs() <= radius + 1e-12).all())
            saw_boundary |= bool(torch.isclose(y.abs(), radius, rtol=1e-8, atol=0).any())
        self.assertTrue(saw_boundary)

    def test_missing_tensor_pauses_only_its_coordinates(self):
        for algorithm in (KT, ConicFTRL):
            first = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
            second = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
            optimizer = CoordinateWise(algorithm, [first, second])
            first.grad, second.grad = torch.full_like(first, 0.1), torch.full_like(second, 0.2)
            optimizer.step()
            before = copy.deepcopy(optimizer.learner_state)
            point = first.detach().clone()
            first.grad, second.grad = None, torch.full_like(second, -0.3)
            optimizer.step()
            torch.testing.assert_close(first, point, rtol=0, atol=0)
            for key, value in before.items():
                torch.testing.assert_close(value[:2], optimizer.learner_state[key][:2], rtol=0, atol=0)

    def test_ckt_is_the_wrapper(self):
        p = torch.nn.Parameter(torch.ones(3, dtype=torch.float64))
        q = torch.nn.Parameter(p.detach().clone())
        old, new = cKT([p]), CoordinateWise(KT, [q])
        for gradient in ([0.1, 0., -0.2], [0., 0., 0.], [-0.3, 0.2, 0.]):
            p.grad = torch.tensor(gradient, dtype=p.dtype)
            q.grad = p.grad.clone()
            old.step()
            new.step()
            torch.testing.assert_close(p, q, rtol=0, atol=0)
        new.load_state_dict(copy.deepcopy(old.state_dict()))

    def test_checkpoint_and_closure(self):
        for algorithm, options in ((KT, {'w': 0.2}),
                                   (ConicFTRL, {'epsilon': 0.1, 'gamma': 3.})):
            p = torch.nn.Parameter(torch.tensor([1., -2., 3.], dtype=torch.float64))
            optimizer = CoordinateWise(algorithm, [p], **options)
            for _ in range(3):
                p.grad = p.detach().clone() * .1
                optimizer.step()
            q = torch.nn.Parameter(p.detach().clone())
            restored = CoordinateWise(algorithm, [q])
            buffer = io.BytesIO()
            torch.save(optimizer.state_dict(), buffer)
            buffer.seek(0)
            restored.load_state_dict(torch.load(buffer))
            self.assertEqual(restored.algorithm.config, optimizer.algorithm.config)
            for _ in range(4):
                for point, opt in ((p, optimizer), (q, restored)):
                    def closure():
                        opt.zero_grad()
                        loss = point.square().sum() * 0.05
                        loss.backward()
                        return loss
                    expected = point.detach().square().sum() * 0.05
                    torch.testing.assert_close(opt.step(closure), expected)
                torch.testing.assert_close(p, q, rtol=0, atol=0)

    def test_checkpoint_rejects_other_geometry_or_algorithm(self):
        p = torch.nn.Parameter(torch.ones(3))
        checkpoint = CoordinateWise(KT, [p]).state_dict()
        for optimizer in (KT([p]), CoordinateWise(ConicFTRL, [p])):
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                optimizer.load_state_dict(checkpoint)

    def test_extension_without_wrapper_changes(self):
        p = torch.nn.Parameter(torch.tensor([1., 2.], dtype=torch.float64))
        optimizer = CoordinateWise(MomentumRule, [p], rate=0.1, beta=0.5)
        p.grad = torch.tensor([1., -2.], dtype=p.dtype)
        optimizer.step()
        p.grad.zero_()
        optimizer.step()
        torch.testing.assert_close(p, torch.tensor([0.85, 2.3], dtype=p.dtype))
        checkpoint = copy.deepcopy(optimizer.state_dict())
        restored = CoordinateWise(MomentumRule, [p])
        restored.load_state_dict(checkpoint)
        torch.testing.assert_close(restored.learner_state['momentum']['velocity'],
                                   optimizer.learner_state['momentum']['velocity'])
        # The same rule can also be used as a single vector learner.
        q = torch.nn.Parameter(torch.tensor([1., 2.], dtype=p.dtype))
        vector = BatchedOptimizer([q], MomentumRule(rate=0.1, beta=0.5))
        for gradient in ([1., -2.], [0., 0.]):
            q.grad = torch.tensor(gradient, dtype=q.dtype)
            vector.step()
        torch.testing.assert_close(p, q)

    def test_rules_do_not_mutate_inputs(self):
        for optimizer_type in (KT, ConicFTRL):
            rule = optimizer_type.algorithm_type()
            point = torch.ones(4, 1, dtype=torch.float64)
            gradient = point * 0.1
            state = rule.init_state(point)
            before = copy.deepcopy(state)
            rule.update(state, point, gradient)
            for key, value in state.items():
                torch.testing.assert_close(value, before[key], rtol=0, atol=0)
            torch.testing.assert_close(point, torch.ones_like(point), rtol=0, atol=0)
            torch.testing.assert_close(gradient, torch.full_like(point, 0.1), rtol=0, atol=0)

    @unittest.skipUnless(hasattr(torch, 'compile'), 'torch.compile is unavailable')
    def test_rules_support_fullgraph_capture(self):
        for optimizer_type in (KT, ConicFTRL):
            for size in (1, 4):
                rule = optimizer_type.algorithm_type()
                point = torch.ones(size, 1, dtype=torch.float64)
                gradient = point * 0.1
                state = rule.init_state(point)
                update = torch.compile(rule.update, backend='eager', fullgraph=True)
                actual_state, actual_point = update(state, point, gradient)
                expected_state, expected_point = rule.update(state, point, gradient)
                torch.testing.assert_close(actual_point, expected_point)
                for key in expected_state:
                    torch.testing.assert_close(actual_state[key], expected_state[key])

    @unittest.skipUnless(hasattr(torch, 'compile'), 'torch.compile is unavailable')
    def test_adapter_masks_and_checks_support_fullgraph_capture(self):
        for algorithm in (KT, ConicFTRL):
            p = torch.nn.Parameter(torch.tensor([1., -2., 3.], dtype=torch.float64))
            q = torch.nn.Parameter(p.detach().clone())
            actual = CoordinateWise(algorithm, [p])
            reference = CoordinateWise(algorithm, [q])
            actual._update = torch.compile(
                actual._tensor_list_step if actual._use_tensor_lists else actual._packed_functional_step,
                backend='eager', fullgraph=True)
            for gradient in ([0.1, 0., -0.2], [0., 0., 0.], [0., -0.3, 0.1]):
                p.grad = torch.tensor(gradient, dtype=p.dtype)
                q.grad = p.grad.clone()
                actual.step()
                reference.step()
                torch.testing.assert_close(p, q)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is unavailable')
    def test_cuda_matches_cpu_and_restores_cpu_checkpoint(self):
        for algorithm in (KT, ConicFTRL):
            cpu = torch.nn.Parameter(torch.tensor([1., -2., 3.], dtype=torch.float64))
            gpu = torch.nn.Parameter(cpu.detach().cuda())
            old, new = CoordinateWise(algorithm, [cpu]), CoordinateWise(algorithm, [gpu])
            new.load_state_dict(copy.deepcopy(old.state_dict()))
            for _ in range(5):
                for point, opt in ((cpu, old), (gpu, new)):
                    point.grad = point.detach().clone() * 0.1
                    opt.step()
                torch.testing.assert_close(cpu, gpu.cpu())

    @unittest.skipUnless(hasattr(torch, 'compile'), 'torch.compile is unavailable')
    def test_compiled_packing_with_multiple_parameters_and_missing_gradients(self):
        for algorithm in (KT, ConicFTRL):
            values = [torch.nn.Parameter(torch.tensor([[1., -2.], [3., 4.]], dtype=torch.float64).T),
                      torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float64))]
            copies = [torch.nn.Parameter(value.detach().clone()) for value in values]
            actual = CoordinateWise(algorithm, values)
            reference = CoordinateWise(algorithm, copies)
            actual._update = torch.compile(
                actual._tensor_list_step if actual._use_tensor_lists else actual._packed_functional_step,
                backend='eager', fullgraph=True)
            for step in range(3):
                for index, (p, q) in enumerate(zip(values, copies)):
                    grad = None if index == 1 and step == 1 else torch.full_like(p, (step - 1) * 0.1)
                    p.grad = grad
                    q.grad = None if grad is None else grad.clone()
                actual.step()
                reference.step()
                for p, q in zip(values, copies):
                    torch.testing.assert_close(p, q)
                for key, value in actual.learner_state.items():
                    torch.testing.assert_close(value, reference.learner_state[key])


if __name__ == '__main__':
    unittest.main()
