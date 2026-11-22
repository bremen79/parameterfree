# Copyright (c) 2026 Francesco Orabona
# Licensed under the MIT License. See LICENSE in the project root.

"""Batched learner interface and the shared PyTorch optimizer adapter."""

from abc import ABC, abstractmethod
import copy
import math

import torch
from torch.optim import Optimizer


class BatchedAlgorithm(ABC):
    """Interface for independent learners with a common event dimension.

    Points and gradients have shape (learners, coordinates). Each learner must
    be independent: reduce only over the last dimension. All state leaves must
    be tensors with the same leading learner dimension. ``update`` must return
    (new_state, new_points) without mutating its inputs. State structure and
    shapes remain fixed. Put immutable constructor options in ``config``;
    ``type(self)(**self.config)`` must recreate the algorithm for checkpoints.

    Set ``skip_zero_gradients=True`` if zero feedback must leave a learner
    unchanged. The adapter handles those rounds and missing gradients using
    masks. Updates must still return finite candidates for zero feedback.

    Optional: set supports_tensor_lists=True for direct coordinate-wise updates
    on each model tensor without concatenation. A vector rule additionally
    declares vector_point_state_keys (flat dictionary leaves of shape (1, d))
    and implements pure update_vector_blocks(states, points, gradients), returning
    lists of candidate states and predictions. Other leaves are shared global
    statistics; return the same updated value for them in every block. Inputs
    have shape (1, block_coordinates), including zero feedback for missing
    tensors. The adapter preserves the packed checkpoint layout in both paths.
    Uniform per-coordinate scalar state that stays uniform for a whole model
    tensor (e.g. KT's integer clock) can be declared in shared_coordinate_state_keys.
    Such leaves may have shape (1,) in the direct coordinate-wise update and must
    broadcast; checkpoint export restores the full per-coordinate shape.
    Optional foreach hooks are update_vector_blocks_foreach (same vector API)
    and update_coordinate_blocks (active tensors only, shape (n, 1)). Both
    return pure candidate-state and prediction lists. The coordinate hook is
    used only when skip_zero_gradients=False, preserving zero-round policies.
    """

    skip_zero_gradients = False
    # Optional execution optimization. Coordinate-wise rules can opt in without
    # changing update(). Vector rules also provide update_vector_blocks() and
    # name the state leaves partitioned over the event dimension.
    supports_tensor_lists = False
    vector_point_state_keys = ()
    shared_coordinate_state_keys = ()

    @abstractmethod
    def init_state(self, initial_point):
        """Create tensor state, capturing any required initial anchor."""

    @abstractmethod
    def update(self, state, point, gradient):
        """Return new tensor state and points for all learners in parallel."""

    def is_valid(self, state):
        """Optional additional validity condition, one boolean per learner."""
        return None


def _map_state(function, state):
    if isinstance(state, dict):
        return {key: _map_state(function, value) for key, value in state.items()}
    if not isinstance(state, torch.Tensor):
        raise TypeError('BatchedAlgorithm state must contain dictionaries and tensors')
    return function(state)


def _merge_state(previous, candidate, active):
    if isinstance(previous, dict):
        if not isinstance(candidate, dict) or previous.keys() != candidate.keys():
            raise ValueError('BatchedAlgorithm.update must preserve state structure')
        return {key: _merge_state(value, candidate[key], active) for key, value in previous.items()}
    if (candidate.shape != previous.shape or candidate.dtype != previous.dtype
            or candidate.device != previous.device):
        raise ValueError('BatchedAlgorithm.update must preserve state shapes and dtypes')
    if candidate is previous or active is None:
        return candidate
    mask = active.reshape((-1,) + (1,) * (previous.ndim - 1))
    return torch.where(mask, candidate, previous)


def _state_is_finite(state):
    if isinstance(state, dict):
        result = None
        for value in state.values():
            valid = _state_is_finite(value)
            if valid is not None:
                result = valid if result is None else result & valid
        return result
    return torch.isfinite(state).all() if state.is_floating_point() else None


class BatchedOptimizer(Optimizer):
    """Adapt a BatchedAlgorithm to one vector learner or scalar learners.

    Parameters must be dense float32/float64 tensors of one dtype on one device.
    The parameter set is fixed at construction; multiple initial parameter
    groups are supported. ``lr`` scales the gradient before *all* algorithm
    calculations, and L2 weight decay is applied without changing ``p.grad``.

    ``error_if_nonfinite=True`` validates candidate state before committing it.
    On CUDA that check synchronizes once per step; disabling it skips the
    adapter check (algorithms may still reject invalid state). ``compile=True`` compiles
    the update with torch.compile, requiring a PyTorch version that supports it.

    max_grad_norm optionally clips raw gradients globally before weight decay
    and lr scaling, without changing p.grad. Uses L2 for vector learners and
    Linf for coordinate-wise learners, matching clip_grad_norm_'s 1e-6 term.
    The default None preserves unclipped behavior. This execution option, like
    compile and error_if_nonfinite, is supplied again when resuming a checkpoint.
    foreach=None selects multi-tensor pointwise operations on CUDA where the
    algorithm provides them; False uses per-tensor operations, True forces the
    multi-tensor path (also usable on CPU). It does not alter checkpoint state.
    """

    algorithm_type = None

    def __init__(self, params, algorithm, coordinatewise=False, weight_decay=0.0,
                 error_if_nonfinite=True, compile=False, max_grad_norm=None, foreach=None):
        if not isinstance(algorithm, BatchedAlgorithm):
            raise TypeError('algorithm must implement BatchedAlgorithm')
        self.algorithm = algorithm
        self.coordinatewise = coordinatewise
        self.error_if_nonfinite = error_if_nonfinite
        self._compile = compile
        if max_grad_norm is not None and (not math.isfinite(max_grad_norm) or max_grad_norm <= 0):
            raise ValueError('max_grad_norm must be finite and positive')
        self.max_grad_norm = max_grad_norm
        if foreach is not None and not isinstance(foreach, bool):
            raise ValueError('foreach must be True, False, or None')
        self.foreach = foreach
        self._ready = False
        super().__init__(params, dict(lr=1.0, weight_decay=weight_decay))
        self._params = [p for group in self.param_groups for p in group['params']]
        if not self._params or not sum(p.numel() for p in self._params):
            raise ValueError('BatchedOptimizer requires at least one parameter element')
        if len({id(p) for p in self._params}) != len(self._params):
            raise ValueError('BatchedOptimizer does not support duplicate parameters')
        if len({(p.device, p.dtype) for p in self._params}) != 1:
            raise ValueError('Parameters must have one dtype on one device')
        initial = self._point()
        state = algorithm.init_state(initial)
        def check_leaf(value):
            if value.ndim == 0 or value.shape[0] != initial.shape[0]:
                raise ValueError('State tensors must have a leading learner dimension')
            if value.device != initial.device:
                raise ValueError('State tensors must be on the parameter device')
            return value
        self.state['learner'] = _map_state(check_leaf, state)
        self._ready = True
        self._set_update()

    @property
    def learner_state(self):
        """Packed state for inspection/export; direct execution packs on demand.

        Treat this as read-only. The tensor-list path owns independent buffers
        and only materializes a packed snapshot when this property is read.
        """
        if 'blocks' not in self.state:
            return self.state['learner']
        blocks = self.state['blocks']
        def pack(values):
            if isinstance(values[0], dict):
                return {key: pack([value[key] for value in values]) for key in values[0]}
            return torch.cat(values, dim=0)
        if self.coordinatewise:
            # Uniform clocks can be stored once per model tensor. Export the
            # original per-coordinate shape, including nonuniform loaded clocks.
            expanded = [{key: (value.expand(point.numel())
                                if key in self.algorithm.shared_coordinate_state_keys else value)
                         for key, value in block.items()}
                        for block, point in zip(blocks, self._params)]
            return pack(expanded)
        return {key: (torch.cat([block[key] for block in blocks], dim=-1)
                      if key in self.algorithm.vector_point_state_keys else blocks[0][key])
                for key in blocks[0]}

    def _set_update(self):
        available = all(hasattr(torch, name) for name in (
            '_foreach_add', '_foreach_sub', '_foreach_mul', '_foreach_copy_'))
        if self.foreach and not available:
            raise ValueError('foreach=True requires PyTorch tensor-list operators')
        self._foreach = (self._params[0].is_cuda and available) if self.foreach is None else self.foreach
        self._use_tensor_lists = self.algorithm.supports_tensor_lists and (
            self.coordinatewise or bool(self.algorithm.vector_point_state_keys))
        if self._use_tensor_lists and 'learner' in self.state:
            previous = self.state.pop('learner')
            views = self._state_blocks(previous, self._params)
            if self.coordinatewise:
                blocks = [_map_state(lambda value: value.clone(), block) for block in views]
                for block in blocks:
                    for key in self.algorithm.shared_coordinate_state_keys:
                        value = block[key]
                        # Only at construction/load, never in the step. Retain
                        # arbitrary nonuniform checkpoint values without loss.
                        if torch.equal(value, value[:1].expand_as(value)):
                            block[key] = value[:1].clone()
            else:
                shared = {key: value.clone() for key, value in previous.items()
                          if key not in self.algorithm.vector_point_state_keys}
                blocks = [{key: (value.clone() if key in self.algorithm.vector_point_state_keys else shared[key])
                           for key, value in block.items()} for block in views]
            self.state['blocks'] = blocks
        self._update = self._tensor_list_step if self._use_tensor_lists else self._packed_functional_step
        if self._compile:
            if not hasattr(torch, 'compile'):
                raise ValueError('compile=True requires torch.compile support')
            self._update = torch.compile(self._update, fullgraph=True)

    def _state_blocks(self, previous, points):
        blocks, offset = [], 0
        for point in points:
            size = point.numel()
            if self.coordinatewise:
                block = _map_state(lambda value: value.narrow(0, offset, size), previous)
            else:
                block = {key: (value.narrow(-1, offset, size)
                               if key in self.algorithm.vector_point_state_keys else value)
                         for key, value in previous.items()}
            blocks.append(block)
            offset += size
        return blocks

    def _commit_blocks(self, blocks, candidates, points, predictions):
        destinations, sources = [], []
        def copy_state(destination, source):
            if isinstance(destination, dict):
                for key in destination:
                    copy_state(destination[key], source[key])
            elif destination is not source:
                destinations.append(destination)
                sources.append(source)
        for index, (block, candidate, point, prediction) in enumerate(
                zip(blocks, candidates, points, predictions)):
            if candidate is None:
                continue
            if self.coordinatewise:
                copy_state(block, candidate)
            else:
                # Shared vector statistics occur in every block, but are
                # committed only once. Point-shaped state has independent storage.
                for key in block:
                    if index == 0 or key in self.algorithm.vector_point_state_keys:
                        copy_state(block[key], candidate[key])
            destinations.append(point)
            sources.append(prediction.reshape_as(point))
        if self._foreach:
            torch._foreach_copy_(destinations, sources)
        else:
            for destination, source in zip(destinations, sources):
                destination.copy_(source)

    def _clip_gradients(self, gradients):
        if self.max_grad_norm is None:
            return gradients
        present = [gradient for gradient in gradients if gradient is not None]
        norm_type = float('inf') if self.coordinatewise else 2.0
        # Explicit reductions lower into the compiled graph on both CPU and
        # CUDA; _foreach_norm can remain an opaque eager call (notably Linf).
        norms = [torch.linalg.vector_norm(gradient, ord=norm_type) for gradient in present]
        total = torch.linalg.vector_norm(torch.stack(norms), ord=norm_type)
        coefficient = (self.max_grad_norm / (total + 1e-6)).clamp(max=1)
        if self._foreach:
            scaled = iter(torch._foreach_mul(present, coefficient))
            return [None if gradient is None else next(scaled) for gradient in gradients]
        return [None if gradient is None else gradient * coefficient for gradient in gradients]

    def _tensor_list_step(self, blocks, points, gradients, clip):
        # All model tensors enter one compiled graph. Parameters are never
        # concatenated. Independent state storage avoids slice_scatter copies
        # when the compiler functionalizes in-place updates to views.
        if clip:
            gradients = self._clip_gradients(gradients)
        if self.coordinatewise:
            candidates, predictions, valid = [], [], None
            if (self._foreach and hasattr(self.algorithm, 'update_coordinate_blocks')
                    and not self.algorithm.skip_zero_gradients):
                present = [index for index, gradient in enumerate(gradients) if gradient is not None]
                active_blocks = [blocks[index] for index in present]
                matrix_points = [points[index].reshape(-1, 1) for index in present]
                matrix_gradients = [gradients[index].reshape(-1, 1) for index in present]
                active_candidates, active_predictions = self.algorithm.update_coordinate_blocks(
                    active_blocks, matrix_points, matrix_gradients)
                candidates, predictions = [None] * len(points), [None] * len(points)
                for index, block, candidate, prediction, point, gradient in zip(
                        present, active_blocks, active_candidates, active_predictions, matrix_points, matrix_gradients):
                    candidates[index], predictions[index] = candidate, prediction
                    if self.error_if_nonfinite:
                        _merge_state(block, candidate, None)
                        if (prediction.shape != point.shape or prediction.dtype != point.dtype
                                or prediction.device != point.device):
                            raise ValueError('BatchedAlgorithm.update must preserve point shape and dtype')
                        check = (_state_is_finite(candidate) & torch.isfinite(prediction).all()
                                 & torch.isfinite(gradient).all())
                        extra = self.algorithm.is_valid(candidate)
                        if extra is not None:
                            check = check & extra.all()
                        valid = check if valid is None else valid & check
            else:
                for block, point, gradient in zip(blocks, points, gradients):
                    if gradient is None:
                        candidates.append(None)
                        predictions.append(None)
                        continue
                    point = point.reshape(-1, 1)
                    gradient = gradient.reshape(-1, 1)
                    active = gradient.ne(0).any(dim=-1) if self.algorithm.skip_zero_gradients else None
                    candidate, prediction, check = self._functional_step(block, point, gradient, active)
                    candidates.append(candidate)
                    predictions.append(prediction)
                    if check is not None:
                        valid = check if valid is None else valid & check
        else:
            matrix_points = [point.reshape(1, -1) for point in points]
            matrix_gradients = [torch.zeros_like(point) if gradient is None else gradient.reshape(1, -1)
                                for point, gradient in zip(matrix_points, gradients)]
            update = (self.algorithm.update_vector_blocks_foreach
                      if self._foreach and hasattr(self.algorithm, 'update_vector_blocks_foreach')
                      else self.algorithm.update_vector_blocks)
            candidates, predictions = update(blocks, matrix_points, matrix_gradients)
            valid = None
            if self.error_if_nonfinite:
                for block, candidate, prediction, point, gradient in zip(
                        blocks, candidates, predictions, matrix_points, matrix_gradients):
                    _merge_state(block, candidate, None)
                    if (prediction.shape != point.shape or prediction.dtype != point.dtype
                            or prediction.device != point.device):
                        raise ValueError('BatchedAlgorithm.update must preserve point shape and dtype')
                    check = (_state_is_finite(candidate) & torch.isfinite(prediction).all()
                             & torch.isfinite(gradient).all())
                    extra = self.algorithm.is_valid(candidate)
                    if extra is not None:
                        check = check & extra.all()
                    valid = check if valid is None else valid & check
        if not self.error_if_nonfinite:
            # These writes fuse with the arithmetic, without returning model-
            # sized intermediates or allocating fresh persistent state buffers.
            self._commit_blocks(blocks, candidates, points, predictions)
            return [], [], None
        return candidates, predictions, valid

    def _step_tensor_lists(self):
        gradients, plain = [], True
        for group in self.param_groups:
            lr, decay = group['lr'], group['weight_decay']
            if not math.isfinite(lr) or lr < 0 or not math.isfinite(decay) or decay < 0:
                raise ValueError('lr and weight_decay must be finite and nonnegative')
            plain = plain and lr == 1 and decay == 0
            for p in group['params']:
                if p.grad is not None and p.grad.layout != torch.strided:
                    raise RuntimeError('Batched optimizers do not support sparse gradients')
                gradients.append(p.grad)
        clip = self.max_grad_norm is not None
        if not plain:
            # Preserve clip-before-decay/scheduling semantics. Keep changing
            # scheduler scalars outside compilation to avoid recompiling each step.
            gradients = self._clip_gradients(gradients)
            effective, index = [], 0
            for group in self.param_groups:
                for p in group['params']:
                    grad = gradients[index]
                    if grad is not None:
                        if group['weight_decay'] != 0:
                            grad = grad.add(p, alpha=group['weight_decay'])
                        if group['lr'] != 1:
                            grad = grad * group['lr']
                    effective.append(grad)
                    index += 1
            gradients, clip = effective, False
        blocks = self.state['blocks']
        candidates, predictions, valid = self._update(blocks, self._params, gradients, clip)
        if valid is not None:
            if not bool(valid):
                raise RuntimeError('{} received non-finite gradients or invalid state'.format(
                    type(self.algorithm).__name__))
            self._commit_blocks(blocks, candidates, self._params, predictions)

    def _packed_functional_step(self, previous, points, gradients, active):
        # Keep packing and zero masks inside the compiled region. In particular,
        # coordinate-wise updates can fuse reads of the original parameter and
        # gradient tensors instead of materializing both packed intermediates.
        shape = (-1, 1) if self.coordinatewise else (1, -1)
        point = torch.cat(points).reshape(shape)
        gradient = torch.cat(gradients).reshape(shape)
        if self.algorithm.skip_zero_gradients:
            nonzero = gradient.ne(0).any(dim=-1)
            active = nonzero if active is None else active & nonzero
        return self._functional_step(previous, point, gradient, active)

    def _functional_step(self, previous, point, gradient, active):
        """Pure tensor portion, including masks and checks, for optional fusion."""
        candidate, prediction = self.algorithm.update(previous, point, gradient)
        if (prediction.shape != point.shape or prediction.dtype != point.dtype
                or prediction.device != point.device):
            raise ValueError('BatchedAlgorithm.update must preserve point shape and dtype')
        candidate = _merge_state(previous, candidate, active)
        if active is not None:
            prediction = torch.where(active.unsqueeze(-1), prediction, point)
        valid = None
        if self.error_if_nonfinite:
            valid = torch.isfinite(prediction).all() & torch.isfinite(gradient).all()
            state_valid = _state_is_finite(candidate)
            if state_valid is not None:
                valid = valid & state_valid
            extra = self.algorithm.is_valid(candidate)
            if extra is not None:
                valid = valid & extra.all()
        return candidate, prediction, valid

    def add_param_group(self, param_group):
        if self._ready:
            raise ValueError('The parameter set is fixed; specify all groups at construction')
        group = dict(param_group)
        params = group['params']
        group['params'] = [params] if isinstance(params, torch.Tensor) else list(params)
        for name in ('lr', 'weight_decay'):
            value = group.get(name, self.defaults[name])
            if not math.isfinite(value) or value < 0:
                raise ValueError('Invalid {} value: {}'.format(name, value))
        for p in group['params']:
            if p.layout != torch.strided or p.dtype not in (torch.float32, torch.float64):
                raise ValueError('Parameters must be dense float32 or float64 tensors')
        super().add_param_group(group)

    def _point(self):
        flat = torch.cat([p.detach().reshape(-1) for p in self._params])
        return flat.reshape(-1, 1) if self.coordinatewise else flat.reshape(1, -1)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if not any(p.grad is not None for p in self._params):
            return loss
        if self._use_tensor_lists:
            self._step_tensor_lists()
            return loss
        gradients, masks = [], []
        missing = any(p.grad is None for p in self._params)
        clipped = iter(self._clip_gradients([p.grad for p in self._params]))
        for group in self.param_groups:
            lr, decay = group['lr'], group['weight_decay']
            if not math.isfinite(lr) or lr < 0 or not math.isfinite(decay) or decay < 0:
                raise ValueError('lr and weight_decay must be finite and nonnegative')
            for p in group['params']:
                raw_gradient = next(clipped)
                if raw_gradient is None:
                    gradients.append(torch.zeros_like(p).reshape(-1))
                else:
                    if p.grad.layout != torch.strided:
                        raise RuntimeError('Batched optimizers do not support sparse gradients')
                    grad = raw_gradient if decay == 0 else raw_gradient.add(p, alpha=decay)
                    if lr != 1:
                        grad = grad * lr
                    gradients.append(grad.reshape(-1))
                if self.coordinatewise and missing:
                    masks.append(torch.full((p.numel(),), p.grad is not None,
                                            dtype=torch.bool, device=p.device))
        points = [p.detach().reshape(-1) for p in self._params]
        active = torch.cat(masks) if self.coordinatewise and missing else None
        previous = self.learner_state
        candidate, prediction, valid = self._update(previous, points, gradients, active)
        if valid is not None and not bool(valid):
            raise RuntimeError('{} received non-finite gradients or invalid state'.format(
                type(self.algorithm).__name__))
        self.state['learner'] = candidate
        flat, offset = prediction.reshape(-1), 0
        for p in self._params:
            if not self.coordinatewise or p.grad is not None:
                p.copy_(flat[offset:offset + p.numel()].reshape_as(p))
            offset += p.numel()
        return loss

    def state_dict(self):
        state = super().state_dict()
        if 'blocks' in self.state:
            state['state'] = dict(state['state'])
            state['state'].pop('blocks')
            state['state']['learner'] = self.learner_state
        state['algorithm'] = dict(
            version=2, name=type(self.algorithm).__module__ + '.' + type(self.algorithm).__qualname__,
            config=copy.deepcopy(self.algorithm.config), coordinatewise=self.coordinatewise,
            shapes=[tuple(p.shape) for p in self._params], dtype=str(self._params[0].dtype))
        return state

    def load_state_dict(self, state_dict):
        metadata = state_dict.get('algorithm')
        if metadata is None:
            raise ValueError('Missing batched algorithm state; older checkpoints are unsupported')
        expected = self.state_dict()['algorithm']
        for key in ('version', 'name', 'coordinatewise', 'shapes', 'dtype'):
            if metadata.get(key) != expected[key]:
                raise ValueError('Checkpoint algorithm/layout mismatch: {}'.format(key))
        algorithm = type(self.algorithm)(**metadata['config'])
        base = dict(state_dict)
        base.pop('algorithm')
        base['state'] = dict(base['state'])
        device = self._params[0].device
        restored = _map_state(lambda value: value.detach().to(device).clone(), base['state']['learner'])
        template = algorithm.init_state(self._point())
        _merge_state(template, restored, None)
        base['state']['learner'] = restored
        super().load_state_dict(base)
        self.algorithm = algorithm
        self._set_update()
