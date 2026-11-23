<!--
Copyright (c) 2026 Francesco Orabona
Licensed under the MIT License. See LICENSE in the project root.
-->

# Contributing

## Development and tests

Install the package with `python -m pip install -e .`, then run the tests with
`python -m unittest discover -s tests -v`. Install `torchvision` and `numpy`
to run the public example tests as well. Tests for the optional local profiler
are skipped when that development tool is absent from the checkout or source
distribution.

## Adding algorithms

Implement [BatchedAlgorithm](parameterfree/base.py), then use the same rule in
vector or coordinate-wise mode. No dispatch table or wrapper changes are needed:

```python
import torch
from parameterfree import BatchedAlgorithm, BatchedOptimizer, CoordinateWise

class MyRule(BatchedAlgorithm):
    def __init__(self, rate=0.1):
        self.config = dict(rate=rate)

    def init_state(self, initial_point):
        return dict(anchor=initial_point.clone())

    def update(self, state, point, gradient):
        return state, point - self.config['rate'] * gradient

model = torch.nn.Linear(10, 1)
vector = BatchedOptimizer(model.parameters(), MyRule(rate=0.1))
# Or, on another model:
other_model = torch.nn.Linear(10, 1)
coordinate = CoordinateWise(MyRule, other_model.parameters(), rate=0.1)
```

Points and gradients have shape `(learners, coordinates)`: `(1, d)` in vector
mode and `(d, 1)` in coordinate-wise mode. Reduce only the last dimension so
learners remain independent. State is a possibly nested dictionary of tensors,
each with the same leading learner dimension. `update` returns new state and
points without modifying its inputs; preserve shapes, dtypes, devices, and
dictionary keys. Keep constructor options in `config` so
`type(rule)(**rule.config)` recreates the rule on checkpoint restoration.
Set `skip_zero_gradients=True` for a zero-round no-op policy. An optional
`is_valid(state)` can return a boolean per learner for additional checks.

For a named optimizer class, subclass `BatchedOptimizer`, set
`algorithm_type = MyRule`, and construct it with an instance of that rule.
`CoordinateWise(MyOptimizer, params, ...)` then works automatically. Ordinary
PyTorch optimizers without this interface are not supported by the wrapper.

Rules can optionally set `supports_tensor_lists=True` for direct coordinate-wise
updates without parameter packing. The same `update` method is applied to
independent state blocks for each model tensor. To optimize a vector rule too, define
`vector_point_state_keys` and the pure `update_vector_blocks` method described
in `BatchedAlgorithm`; KT is an example. Unmodified rules retain the packed
execution path. Both paths use the same checkpoint layout. A vector rule can
optionally supply `update_vector_blocks_foreach` for multi-tensor execution.
A coordinate-wise rule can supply
`update_coordinate_blocks(states, points, gradients)`, returning candidate state
and prediction lists for the active tensors. The adapter masks zero coordinates
when `skip_zero_gradients=True`. KT provides both examples; rules
without these hooks retain their existing update method.

## Publishing a release

Normal commits and pushes run tests without uploading to PyPI. Publishing a
GitHub Release with a tag such as `v0.0.2` runs the test suite, builds and checks
the distributions, and uploads them to PyPI. The tag must match the version in
`setup.py`; tagging a commit without publishing a GitHub Release does not upload.

One-time setup: create the GitHub environment `pypi`, then add a GitHub Trusted
Publisher in the existing `parameterfree` project's PyPI publishing settings:

| Field | Value |
| --- | --- |
| Owner | `bremen79` |
| Repository | `parameterfree` |
| Workflow filename | `publish.yml` |
| Environment | `pypi` |

No API token is needed. For each release, update and commit the version in
`setup.py`, then publish a GitHub Release with the matching `v`-prefixed tag.
