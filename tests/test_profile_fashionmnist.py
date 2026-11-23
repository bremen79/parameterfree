# Copyright (c) 2026 Francesco Orabona
# Licensed under the MIT License. See LICENSE in the project root.

"""Profiler summaries distinguish GPU work, memory operations, and CPU scopes."""

from types import SimpleNamespace
import unittest

import torch

try:
    from benchmarks.profile_fashionmnist import _summarize_trace
except ModuleNotFoundError as error:
    if error.name not in ('benchmarks', 'benchmarks.profile_fashionmnist'):
        raise
    # The profiler is a local development tool, outside the public release.
    raise unittest.SkipTest('optional local profiler is not included') from error


def event(name, device, duration, self_cpu=0, annotation=False):
    return SimpleNamespace(name=name, device_type=device, is_user_annotation=annotation,
                           time_range=SimpleNamespace(elapsed_us=lambda: duration),
                           self_cpu_time_total=self_cpu)


class ProfileSummaryTests(unittest.TestCase):
    def test_averages_device_events_without_counting_cpu_scopes_twice(self):
        cpu, cuda = torch.autograd.DeviceType.CPU, torch.autograd.DeviceType.CUDA
        events = [event('fused_update', cuda, 200), event('fused_update', cuda, 300),
                  event('Memcpy DtoH', cuda, 100),
                  event('cudaLaunchKernel', cpu, 40, self_cpu=40),
                  event('profile_optimizer_update', cpu, 2500, self_cpu=500)]
        metrics, details = _summarize_trace(events, steps=2, cuda=True)
        self.assertEqual(metrics['gpu_kernel_ms'], .25)
        self.assertEqual(metrics['gpu_kernels_per_step'], 1.)
        self.assertEqual(metrics['gpu_memory_ms'], .05)
        self.assertEqual(metrics['gpu_memory_events_per_step'], .5)
        self.assertEqual(metrics['launch_cpu_us_per_step'], 20.)
        self.assertEqual(len(details), 2)
        self.assertEqual(details[0]['name'], 'fused_update')
        self.assertEqual(details[0]['gpu_ms_per_step'], .25)

    def test_cuda_annotations_do_not_duplicate_kernel_work(self):
        cuda = torch.autograd.DeviceType.CUDA
        # Regression from the T4 trace: the optimizer annotation and its one
        # kernel had identical duration, making the old summary report twice
        # the work and twice the number of kernels.
        events = [event('triton_for_fused_0', cuda, 645.8906),
                  event('Optimizer.step#COCOB.step', cuda, 645.8906),
                  event('custom_gpu_scope', cuda, 900., annotation=True),
                  event('profile_optimizer_update', cuda, 1000.)]
        metrics, details = _summarize_trace(events, steps=1, cuda=True)
        self.assertAlmostEqual(metrics['gpu_kernel_ms'], .6458906)
        self.assertEqual(metrics['gpu_kernels_per_step'], 1.)
        self.assertEqual(len(details), 1)
        self.assertEqual(details[0]['name'], 'triton_for_fused_0')

    def test_missing_cuda_events_are_unavailable_not_zero_work(self):
        for cuda in (True, False):
            metrics, details = _summarize_trace([], steps=10, cuda=cuda)
            self.assertTrue(all(value is None for value in metrics.values()))
            self.assertEqual(details, [])


if __name__ == '__main__':
    unittest.main()
