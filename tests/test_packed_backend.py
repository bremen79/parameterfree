# Copyright (c) 2026 Francesco Orabona
# Licensed under the MIT License. See LICENSE in the project root.

"""Compiled packed updates preserve numerical and checkpoint semantics."""

import copy
import unittest
from unittest.mock import patch

import torch

from parameterfree import ConicFTRL, CoordinateWise


class PackedBackendTests(unittest.TestCase):
    @unittest.skipUnless(hasattr(torch, 'compile'), 'torch.compile is unavailable')
    def test_compiled_updates_match_pure_updates_and_checkpoints(self):
        compiler = torch.compile
        for coordinatewise in (False, True):
            for dtype in (torch.float32, torch.float64):
                torch._dynamo.reset()
                with self.subTest(coordinatewise=coordinatewise, dtype=dtype):
                    values = [torch.nn.Parameter(torch.tensor([[1., -2.], [3., 4.]], dtype=dtype).T),
                              torch.nn.Parameter(torch.tensor(.5, dtype=dtype))]
                    refs = [torch.nn.Parameter(p.detach().clone()) for p in values]
                    def make(params, **kwargs):
                        if coordinatewise:
                            return CoordinateWise(ConicFTRL, params, error_if_nonfinite=False, **kwargs)
                        return ConicFTRL(params, error_if_nonfinite=False, **kwargs)
                    expected = make(refs)
                    with patch('torch.compile', side_effect=lambda fn, **kw: compiler(fn, backend='eager', **kw)):
                        actual = make(values, compile=True)
                    for step in range(5):
                        for index, (p, q) in enumerate(zip(values, refs)):
                            grad = (None if step == 3 or (index == 1 and step == 2) else
                                    torch.full_like(p, (0., .1, -.2)[step % 3]))
                            p.grad = grad
                            q.grad = None if grad is None else grad.clone()
                        actual.step()
                        expected.step()
                        for p, q in zip(values, refs):
                            torch.testing.assert_close(p, q)
                        for key, tensor in actual.learner_state.items():
                            torch.testing.assert_close(tensor, expected.learner_state[key])
                    copied = [torch.nn.Parameter(p.detach().clone()) for p in values]
                    restored = make(copied)
                    restored.load_state_dict(copy.deepcopy(actual.state_dict()))
                    for params, opt in ((values, actual), (copied, restored)):
                        for p in params:
                            p.grad = torch.full_like(p, .1)
                        opt.step()
                    for p, q in zip(values, copied):
                        torch.testing.assert_close(p, q)

    @unittest.skipUnless(torch.cuda.is_available() and hasattr(torch, 'compile'), 'CUDA compilation is unavailable')
    def test_compiled_cuda_packed_updates_match_cpu(self):
        for coordinatewise in (False, True):
            torch._dynamo.reset()
            cpu = [torch.nn.Parameter(torch.tensor([1., -2.])), torch.nn.Parameter(torch.tensor(.5))]
            gpu = [torch.nn.Parameter(p.detach().cuda()) for p in cpu]
            def make(params, **kwargs):
                if coordinatewise:
                    return CoordinateWise(ConicFTRL, params, error_if_nonfinite=False, **kwargs)
                return ConicFTRL(params, error_if_nonfinite=False, **kwargs)
            expected, actual = make(cpu), make(gpu, compile=True)
            for step in range(4):
                for index, (p, q) in enumerate(zip(cpu, gpu)):
                    p.grad = None if index == 1 and step == 2 else torch.full_like(p, .1 * (step - 1))
                    q.grad = None if p.grad is None else p.grad.cuda()
                expected.step()
                actual.step()
                for p, q in zip(cpu, gpu):
                    torch.testing.assert_close(p, q.cpu())
                for key, tensor in actual.learner_state.items():
                    torch.testing.assert_close(tensor.cpu(), expected.learner_state[key])


if __name__ == '__main__':
    unittest.main()
