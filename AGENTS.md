# Project guidance

## Purpose and fidelity

`jax-shapley` ports FastSVERL to end-to-end JAX using Flax NNX, preserving
upstream behaviour while improving experiment speed. FastSVERL commit
`25b2cbe66b0d4a0decc3b63dbc798a79bf2c7d5b` is the behavioural reference.
Preserve transition, learning, and evaluation semantics, including source quirks.

Follow the attached compile-first Python ML discipline: fail before an expensive
run starts. Static checking, backend compilation, and numerical validation are
separate requirements; none substitutes for the others.

## Required gates

Run all applicable gates before considering a change complete or launching a long
training/evaluation job:

```bash
uv run ty check
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

`./check.sh` runs these gates. Backend compile tests belong in pytest. Update and
commit `uv.lock` with dependency changes; do not manually pin every runtime
library to imitate a lockfile. Keep the pinned ty version deliberate.

Install the project with:

```bash
uv sync --all-groups                 # CPU, including Apple Silicon
uv sync --all-groups --extra cuda    # NVIDIA CUDA 13
```

Add `--extra render` for rendering. JAX is required, not an optional backend.
Generate PyTorch reference fixtures in a separate environment; normal tests must
remain offline and independent of FastSVERL/PyTorch installation.

## Static contracts and boundaries

- Treat every enabled ty warning/error as a build failure. Keep
  `error-on-warning = true` and the deliberately selected strict rules in
  `pyproject.toml`; do not enable `all = "error"` indiscriminately.
- The pinned `ty==0.0.73` supports `invalid-assignment`, not the template's
  `unsound-assignment` name. Use the supported rule.
- Keep Ruff's annotation, stub, and blanket-ignore checks enabled. Ruff enforces
  annotation/suppression policy; ty checks semantic types.
- Do not weaken project-wide checking to accommodate one function or dependency.
- Do not propagate `Any`, unknown types, or heterogeneous catch-all dictionaries
  through configuration, batches, model APIs, optimizer/training state, losses,
  metrics, or public interfaces.
- Parse and validate external YAML/JSON/CLI input once, then use typed
  configuration. Prefer frozen dataclasses (with slots where practical), enums or
  Literal for closed choices, and Protocols for behavioural interfaces.
- Use Flax struct dataclasses for JAX state. Narrow optionals/unions before use.
  Prefer explicit fields and named structures to ambiguous tuples or `extra`.
- Isolate poorly typed libraries, serialization, and telemetry behind small typed
  adapters. Dynamic external data must not determine the core's data model.
- Suppressions must target the exact diagnostic on the smallest expression/line,
  with an explanation. No blanket ignores. Review casts as boundary assertions,
  not substitutes for fixing an incorrect application contract.

## Tensor shapes, layouts, and dtypes

- Annotate tensor APIs with the backend array type at minimum. Use jaxtyping for
  important shape/dtype contracts; annotations document intent but do not prove
  symbolic shape relationships under ty.
- Environment observations, actions, and counters use int32; flags use bool;
  rewards, discounts, and episode returns use float32. Rendered frames use uint8.
  Convert observations to the chosen floating dtype at the model boundary.
- Keep precision deliberate. Compile/test each dtype used in production; do not
  introduce mixed precision silently.
- Prefer einops with meaningful axis names for nontrivial layout transformations.
  Simple locally obvious flattening may use reshape. Compile-test einops patterns.
- State rank/dimension relationships in annotations, structures, boundary checks,
  or compile-test specifications. Do not discover shape errors on production data.

## JAX and NNX numerical code

- Prefer explicit state + inputs + key -> new state + outputs + new key functions.
  Split/pass PRNG keys explicitly; no hidden random/global state.
- Use JAX control flow for traced conditions/loops. Static Python control flow is
  allowed when it depends only on static configuration.
- Avoid ordinary container mutation, implicit host NumPy conversions, logging,
  file/network I/O, environment-variable changes, and thread/process creation in
  compiled regions. Keep side effects in host orchestration. Necessary callbacks
  must be narrow, explicit, and compile-tested.
- NNX model/optimizer mutation is allowed when deliberately modelled by NNX and
  compatible with its transformations. Preserve useful NNX graph/state semantics.
- Use Baxtub's explicit RunState/UpdateState and small rollout/update functions,
  refined with typed configuration and named loss/metric structures. FastSVERL
  remains the authority for algorithm behaviour and hyperparameters.
- Preserve Gymnax auto-reset. Distinguish raw transition observations from reset
  observations and termination from truncation when collecting learning data.
- Never catch tracing/lowering/compiler failures to fall back to eager execution.
  Compiler, shape-contract, and type failures stop the build. Operational recovery
  (optional telemetry, recoverable checkpoint I/O) belongs outside numerical code.

## Validation

For important numerical units, provide static checking, compile tests, output
shape/dtype checks, and known-example/reference numerical tests. For differentiable
code, check gradient structure and finite values. Check JIT/eager equivalence when
useful; compilation alone does not establish numerical correctness.

Trace, lower, and compile representative abstract inputs before expensive runs:

```python
compiled = jax.jit(function).trace(input_spec).lower().compile()
```

Use the equivalent NNX API when the transform owns graph state. Compile tests
must not start tracking SDKs, checkpoint writers, workers, or a training job. Cover
representative batch/sequence sizes, static specializations, optional branches,
and actual production dtypes with small inputs preserving dimension relationships.

Keep compile tests in `tests/compile/`; numerical and integration checks may live
in the existing test modules or focused new modules. No PyTorch implementation
compile tests are needed for this JAX-only project. CPU validation does not prove
GPU compilation/performance; GPU tests must skip explicitly when unavailable.
