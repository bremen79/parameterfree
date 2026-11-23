# Copyright (c) 2026 Francesco Orabona
# Licensed under the MIT License. See LICENSE in the project root.

"""Check the public example's metrics, epoch modes, and optimizer configuration."""

import importlib.util
import unittest

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


HAS_EXTRAS = (importlib.util.find_spec('torchvision') is not None
              and importlib.util.find_spec('numpy') is not None)
if HAS_EXTRAS:
    from examples.test_on_fashionmnist import OPTIMIZERS, evaluate, make_optimizer, train_epoch


class RecordingLinear(torch.nn.Linear):
    def __init__(self):
        super().__init__(2, 2)
        self.modes = []
        with torch.no_grad():
            self.weight.copy_(torch.tensor([[1., 0.], [0., 1.]]))
            self.bias.copy_(torch.tensor([.3, -.2]))

    def forward(self, value):
        self.modes.append(self.training)
        return super().forward(value)


@unittest.skipUnless(HAS_EXTRAS, 'FashionMNIST example requires torchvision and numpy')
class FashionMNISTExampleTests(unittest.TestCase):
    def test_evaluation_weights_partial_batches_by_samples(self):
        logits = torch.tensor([[5., 0.], [0., 5.], [3., 0.], [0., 1.], [-8., 8.]])
        targets = torch.tensor([0, 1, 1, 1, 0])
        loader = DataLoader(TensorDataset(logits, targets), batch_size=2)
        loss, accuracy = evaluate(torch.nn.Identity(), loader, torch.device('cpu'))
        self.assertAlmostEqual(loss, F.cross_entropy(logits, targets).item(), delta=1e-6)
        self.assertEqual(accuracy, .6)

    def test_training_mode_and_metrics_reset_each_epoch(self):
        inputs = torch.tensor([[-1., 2.], [3., -1.], [1., 1.], [0., 2.], [-2., 0.]])
        targets = torch.tensor([1, 0, 1, 0, 1])
        loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)
        model = RecordingLinear()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.)
        model.eval()
        for epoch in range(2):
            if epoch:
                with torch.no_grad():
                    model.bias.add_(torch.tensor([2., -2.]))
            logits = F.linear(inputs, model.weight.detach(), model.bias.detach())
            expected_loss = F.cross_entropy(logits, targets).item()
            expected_accuracy = (logits.argmax(1) == targets).float().mean().item()
            loss, accuracy = train_epoch(model, loader, optimizer, torch.device('cpu'), log_interval=0)
            self.assertAlmostEqual(loss, expected_loss, delta=1e-6)
            self.assertAlmostEqual(accuracy, expected_accuracy, delta=1e-7)
            evaluate(model, loader, torch.device('cpu'))
        self.assertEqual(model.modes, ([True] * 3 + [False] * 3) * 2)

    def test_all_six_optimizers_and_required_bounds(self):
        for name in OPTIMIZERS:
            with self.subTest(optimizer=name):
                parameter = torch.nn.Parameter(torch.tensor([2., -1.]))
                opt = make_optimizer(name, [parameter])
                parameter.grad = torch.full_like(parameter, 2.)
                opt.step()
                self.assertTrue(torch.isfinite(parameter).all())
                if name in ('KT', 'cKT'):
                    self.assertEqual(opt.max_grad_norm, 1.)
                    theta = opt.learner_state['theta']
                    if name == 'KT':
                        self.assertLessEqual(theta.norm().item(), 1.)
                    else:
                        self.assertLessEqual(theta.abs().max().item(), 1.)
                if name == 'cConicFTRL':
                    self.assertTrue(opt.coordinatewise)

    @unittest.skipUnless(hasattr(torch, 'compile'), 'torch.compile is unavailable')
    def test_compiled_adam_cosine_uses_dynamic_tensor_lr(self):
        torch._dynamo.reset()
        parameter = torch.nn.Parameter(torch.tensor([1., -2.]))
        reference = torch.nn.Parameter(parameter.detach().clone())
        actual = make_optimizer('Adam', [parameter], compile=True, cosine_decay=True)
        expected = make_optimizer('Adam', [reference], compile=True, cosine_decay=True)
        actual_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(actual, T_max=12)
        expected_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(expected, T_max=12)
        graphs = []
        def backend(graph, example_inputs):
            graphs.append(graph)
            return graph.forward
        update = torch.compile(actual.step, backend=backend)
        self.assertIsInstance(actual.param_groups[0]['lr'], torch.Tensor)
        for step in range(12):
            parameter.grad = torch.full_like(parameter, .1 * (step + 1))
            reference.grad = parameter.grad.clone()
            update()
            expected.step()
            actual_schedule.step()
            expected_schedule.step()
            torch.testing.assert_close(parameter, reference)
        self.assertLessEqual(len(graphs), 3, 'Adam recompiled for changing scheduler values')
        self.assertEqual(actual.param_groups[0]['lr'].item(), 0.)


if __name__ == '__main__':
    unittest.main()
