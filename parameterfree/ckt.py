# Copyright (c) 2024, 2026 Francesco Orabona
# Licensed under the MIT License. See LICENSE in the project root.

from .coordinatewise import CoordinateWise
from .kt import KT


class cKT(CoordinateWise):
    """Convenience name for CoordinateWise(KT, ...).

    Requires each effective gradient coordinate to lie in [-1, 1]. Wealth is
    per coordinate. Explicit zero-gradient rounds advance each learner's clock,
    matching independent KT copies; older cKT skipped those rounds instead.
    max_grad_norm=1.0 optionally applies global Linf clipping inside the update,
    before decay and lr scaling. Default None leaves clipping to the caller.
    foreach=None selects multi-tensor operations on CUDA; False uses per-tensor
    execution. Both modes use the same KT rule and checkpoint representation.
    """

    def __init__(self, params, w=1e-4, weight_decay=0.0,
                 error_if_nonfinite=True, compile=False, max_grad_norm=None, foreach=None):
        super().__init__(KT, params, w=w, weight_decay=weight_decay,
                         error_if_nonfinite=error_if_nonfinite, compile=compile, max_grad_norm=max_grad_norm, foreach=foreach)
