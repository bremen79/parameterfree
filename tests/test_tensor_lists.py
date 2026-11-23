# Copyright (c) 2026 Francesco Orabona
# Licensed under the MIT License. See LICENSE in the project root.

"""Direct tensor updates must preserve packed KT semantics and clipping."""

import copy
import unittest
from unittest.mock import patch

import torch

from parameterfree import COCOB, KT, cKT, CoordinateWise


class TensorListTests(unittest.TestCase):
    def test_matches_packed_updates_and_external_clipping(self):
        for factory in (KT, cKT, lambda params, **kw: CoordinateWise(KT, params, **kw)):
            for checked in (True, False):
                for dtype in (torch.float32, torch.float64):
                    for grouped in (True, False):
                        with self.subTest(factory=factory, checked=checked, dtype=dtype, grouped=grouped):
                            parameters = [torch.nn.Parameter(torch.tensor([[1., -2.], [3., 4.]], dtype=dtype).T),
                                          torch.nn.Parameter(torch.tensor(.5, dtype=dtype))]
                            references = [torch.nn.Parameter(p.detach().clone()) for p in parameters]
                            def make(values, **options):
                                groups = ([dict(params=[values[0]], weight_decay=.1, lr=.5),
                                           dict(params=[values[1]], lr=.8)] if grouped else values)
                                return factory(groups, error_if_nonfinite=checked, **options)
                            actual = make(parameters, max_grad_norm=1.)
                            reference = make(references)
                            reference.state['learner'] = reference.learner_state
                            del reference.state['blocks']
                            reference._use_tensor_lists = False
                            reference._update = reference._packed_functional_step
                            for step in range(8):
                                if grouped:
                                    # Changing scheduler values must preserve clip -> decay -> lr.
                                    for opt in (actual, reference):
                                        opt.param_groups[0]['lr'] = (0., 1., .2)[step % 3]
                                for index, (p, q) in enumerate(zip(parameters, references)):
                                    grad = (None if step == 4 or (step == 2 and index == 1)
                                            else torch.full_like(p, (0., 10., -.3)[step % 3]))
                                    p.grad = grad
                                    q.grad = None if grad is None else grad.clone()
                                saved_gradients = [None if p.grad is None else p.grad.clone() for p in parameters]
                                norm = float('inf') if reference.coordinatewise else 2.
                                torch.nn.utils.clip_grad_norm_(references, 1., norm_type=norm)
                                actual.step()
                                reference.step()
                                for p, q, grad in zip(parameters, references, saved_gradients):
                                    torch.testing.assert_close(p, q)
                                    if grad is not None:
                                        torch.testing.assert_close(p.grad, grad, rtol=0, atol=0)
                                for key in actual.learner_state:
                                    torch.testing.assert_close(actual.learner_state[key], reference.learner_state[key])
                            # The execution path changes, but the checkpoint layout does not.
                            restored_params = [torch.nn.Parameter(p.detach().clone()) for p in parameters]
                            restored = make(restored_params, max_grad_norm=1.)
                            restored.load_state_dict(copy.deepcopy(actual.state_dict()))
                            for values, opt in ((parameters, actual), (restored_params, restored)):
                                for p in values:
                                    p.grad = torch.full_like(p, -.1)
                                opt.step()
                            for p, q in zip(parameters, restored_params):
                                torch.testing.assert_close(p, q, rtol=0, atol=0)

    @unittest.skipUnless(hasattr(torch, 'compile'), 'torch.compile is unavailable')
    def test_compiled_clipping_and_missing_gradients(self):
        compiler = torch.compile
        for factory in (KT, cKT):
            for checked in (True, False):
                with self.subTest(factory=factory, checked=checked):
                    torch._dynamo.reset()
                    p = torch.nn.Parameter(torch.tensor([1., -2.], dtype=torch.float64))
                    q = torch.nn.Parameter(torch.tensor(.5, dtype=torch.float64))
                    refs = [torch.nn.Parameter(value.detach().clone()) for value in (p, q)]
                    reference = factory(refs, max_grad_norm=1., error_if_nonfinite=checked)
                    with patch('torch.compile', side_effect=lambda fn, **kw: compiler(fn, backend='eager', **kw)):
                        actual = factory([p, q], max_grad_norm=1., error_if_nonfinite=checked, compile=True)
                    for step in range(3):
                        for index, (value, ref) in enumerate(zip((p, q), refs)):
                            value.grad = None if index == 1 and step == 1 else torch.full_like(value, 3. * (step - 1))
                            ref.grad = None if value.grad is None else value.grad.clone()
                        actual.step()
                        reference.step()
                        for value, ref in zip((p, q), refs):
                            torch.testing.assert_close(value, ref)

    def test_validation_is_transactional(self):
        for factory in (KT, cKT):
            p, q = torch.nn.Parameter(torch.ones(2)), torch.nn.Parameter(torch.ones(1))
            opt = factory([p, q], max_grad_norm=1.)
            p.grad, q.grad = torch.ones_like(p), torch.ones_like(q)
            opt.step()
            before = copy.deepcopy(opt.learner_state)
            points = [p.detach().clone(), q.detach().clone()]
            q.grad.fill_(float('nan'))
            with self.assertRaisesRegex(RuntimeError, 'non-finite'):
                opt.step()
            for key, value in before.items():
                torch.testing.assert_close(opt.learner_state[key], value, rtol=0, atol=0)
            for value, old in zip((p, q), points):
                torch.testing.assert_close(value, old, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available() and hasattr(torch, 'compile'), 'CUDA compilation is unavailable')
    def test_compiled_cuda_matches_cpu_with_clipping_and_missing_gradients(self):
        for factory in (COCOB, KT, cKT):
            torch._dynamo.reset()
            cpu = [torch.nn.Parameter(torch.tensor([1., -2.])), torch.nn.Parameter(torch.tensor(.5))]
            gpu = [torch.nn.Parameter(p.detach().cuda()) for p in cpu]
            options = {} if factory is COCOB else dict(max_grad_norm=1.)
            reference = factory(cpu, error_if_nonfinite=False, **options)
            actual = factory(gpu, error_if_nonfinite=False, compile=True, **options)
            for step in range(4):
                for index, (p, q) in enumerate(zip(cpu, gpu)):
                    p.grad = None if index == 1 and step == 2 else torch.full_like(p, 3. * (step - 1))
                    q.grad = None if p.grad is None else p.grad.cuda()
                reference.step()
                actual.step()
                for p, q in zip(cpu, gpu):
                    torch.testing.assert_close(p, q.cpu())

    def test_coordinate_clocks_pause_independently_and_restore_nonuniform_values(self):
        p = torch.nn.Parameter(torch.ones(3))
        opt = cKT([p], error_if_nonfinite=False)
        self.assertEqual(opt.state['blocks'][0]['t'].numel(), 3)
        p.grad = torch.tensor([.1, 0., -.2])
        opt.step()
        torch.testing.assert_close(opt.learner_state['t'], torch.tensor([2, 1, 2], dtype=torch.int64))
        checkpoint = copy.deepcopy(opt.state_dict())
        checkpoint['state']['learner']['t'] = torch.tensor([2, 4, 7], dtype=torch.int64)
        opt.load_state_dict(checkpoint)
        self.assertEqual(opt.state['blocks'][0]['t'].numel(), 3)
        opt.step()
        torch.testing.assert_close(opt.learner_state['t'], torch.tensor([3, 4, 8], dtype=torch.int64))

    def test_unchecked_steps_keep_state_storage(self):
        for factory in (COCOB, KT, cKT):
            p = torch.nn.Parameter(torch.ones(3))
            opt = factory([p], error_if_nonfinite=False)
            p.grad = torch.full_like(p, .1)
            opt.step()
            def pointers():
                blocks = [opt.state[p]] if isinstance(opt, COCOB) else opt.state['blocks']
                return [[value.data_ptr() for value in block.values()] for block in blocks]
            before = pointers()
            opt.step()
            self.assertEqual(pointers(), before)


if __name__ == '__main__':
    unittest.main()
