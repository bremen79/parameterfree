# Copyright (c) 2026 Francesco Orabona
# Licensed under the MIT License. See LICENSE in the project root.

"""Run a batched algorithm independently on every parameter coordinate."""

from .base import BatchedAlgorithm, BatchedOptimizer


class CoordinateWise(BatchedOptimizer):
    """Vectorized independent scalar learners, anchored at construction.

    ``CoordinateWise(KT, params, w=...)`` and
    ``CoordinateWise(ConicFTRL, params, epsilon=..., gamma=...)`` use the same
    rules as their vector versions. ``algorithm`` can also be a custom
    BatchedAlgorithm subclass. Budgets such as ``w`` and ``epsilon`` are per
    coordinate; divide by the number of coordinates to split a total budget.

    Missing gradients pause the affected learners. Explicit zero gradients
    follow the algorithm's zero-round policy. No optimizer objects or Python
    loops are created per coordinate. Rules opting into tensor-list execution
    update each model tensor directly; other rules use packed tensors.
    max_grad_norm optionally clips the global Linf norm of raw gradients before
    decay and lr scaling. Default None leaves clipping to the caller.
    foreach=None selects multi-tensor operations on CUDA where supplied by the
    rule; foreach=False keeps per-tensor execution. Checkpoints are unchanged.
    """

    def __init__(self, algorithm, params, weight_decay=0.0,
                 error_if_nonfinite=True, compile=False, max_grad_norm=None, foreach=None, **algorithm_options):
        if isinstance(algorithm, type) and issubclass(algorithm, BatchedOptimizer):
            algorithm = algorithm.algorithm_type
        if not isinstance(algorithm, type) or not issubclass(algorithm, BatchedAlgorithm):
            raise TypeError('algorithm must provide a BatchedAlgorithm implementation')
        super().__init__(params, algorithm(**algorithm_options), coordinatewise=True,
                         weight_decay=weight_decay, error_if_nonfinite=error_if_nonfinite,
                         compile=compile, max_grad_norm=max_grad_norm, foreach=foreach)
