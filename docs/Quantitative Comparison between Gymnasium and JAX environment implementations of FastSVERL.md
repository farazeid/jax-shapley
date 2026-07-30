# Quantitative Comparison between Gymnasium and JAX environment implementations of FastSVERL

## Result

The JAX ports match **all 143,136 exhaustive FastSVERL transitions** on both CPU and
GPU. Throughput depends strongly on batch size: GPU has overhead at small batches,
then reaches hundreds of millions of transitions per second at 2,048 environments.

| JAX backend | Transitions tested | Mismatched | Proportion |
|---|---:|---:|---:|
| CPU | 143,136 | 0 | 0.000000000000 |
| GPU | 143,136 | 0 | 0.000000000000 |

The mismatch criterion is record-level: a transition is different if any next
observation, environment-state field, reward, termination, or truncation value differs.
Component-level counts and every per-configuration result are retained in
`docs/data/envs-port--parity--cpu.csv` and `docs/data/envs-port--parity--gpu.csv`.

## Throughput

Rates below are median environment transitions per second over five samples. The
Gridworld, Hypercube 2×2, and Mastermind rows use 2,048 environments. Hypercube 6×5
uses 16—the largest safe original point—so all three backends are directly comparable.

| Environment | Workers | Gymnasium CPU | JAX CPU | JAX GPU | CPU speedup | GPU speedup |
|---|---:|---:|---:|---:|---:|---:|
| Gridworld | 2,048 | 216,062 | 8,935,941 | 196,861,104 | 41.4× | 911.1× |
| Hypercube 2×2 | 2,048 | 222,354 | 27,859,808 | 207,388,886 | 125.3× | 932.7× |
| Hypercube 6×5 | 16 | 211,298 | 8,201,712 | 1,948,432 | 38.8× | 9.2× |
| Mastermind-222 | 2,048 | 66,845 | 6,597,059 | 176,219,751 | 98.7× | 2,636.2× |

![Environment throughput](figures/envs-port--throughput.svg)

The complete 1, 2, 4, …, 2,048 curves, five raw samples per successful point, elapsed
times, IQRs, and 95% confidence bounds are available in `docs/data/`. The shaded
areas are deterministic percentile-bootstrap confidence intervals for the median,
using 10,000 resamples with seed 0. Initial
reset, construction, action
generation, transfers, and compilation are excluded. Timed execution includes automatic
resets encountered during 1,024-step rollout blocks.

The original uses FastSVERL's `gym.vector.SyncVectorEnv`. JAX uses `jit(vmap(step))`
inside `lax.scan`, synchronizing with `block_until_ready()`. Identical deterministic
action schedules are used. The original implementation has no GPU path.

For Hypercube 6×5, the unmodified source owns a large transition dictionary per worker.
The safe-source probe reported: `one 6x5 source env: 34320384 bytes RSS, 6.359s construction; safe cap 16`. Original points above 16 are retained as
`memory_infeasible`; JAX CPU and GPU include the complete curve through 2,048.

## Machine and software

- CPU: Intel(R) Core(TM) i9-14900K
- Logical CPUs: 32; physical cores: 24
- RAM: 62.5 GiB
- GPU: NVIDIA GeForce RTX 4080 SUPER, 580.65.06
- Driver/CUDA report: NVIDIA-SMI 580.65.06              Driver Version: 580.65.06      CUDA Version: 13.0
- OS: Linux-6.8.0-124-generic-x86_64-with-glibc2.39
- Python: 3.12.3
- JAX/JAXLIB: 0.11.0 / 0.11.0
- Gymnax: 0.0.9 (locked)
- Gymnasium: 0.29.1
- uv lock SHA-256: `ba3f74e0648eab6e2c632a9d4e742efbcf7eb8c68071e0bf8415ed91d1a1dbdc`
- FastSVERL source commit: `25b2cbe66b0d4a0decc3b63dbc798a79bf2c7d5b`

## Reproduction

```bash
uv sync --all-groups --extra cuda

# Offline semantic tests
JAX_PLATFORMS=cpu uv run --extra cuda pytest
uv run --extra cuda pytest -m gpu

# Parity CSVs
PYTHONPATH=. JAX_PLATFORMS=cpu uv run --extra cuda python scripts/report_parity.py --output docs/data/envs-port--parity--cpu.csv
PYTHONPATH=. uv run --extra cuda python scripts/report_parity.py --output docs/data/envs-port--parity--gpu.csv

# Throughput; SOURCE_ROOT is a temporary checkout at the pinned commit
uv run --group benchmark python scripts/benchmark_envs.py --backend original --source-root "$SOURCE_ROOT" --output docs/data/envs-port--throughput--gymnasium-cpu.csv
JAX_PLATFORMS=cpu uv run --all-groups --extra cuda python scripts/benchmark_envs.py --backend jax --output docs/data/envs-port--throughput--jax-cpu.csv
uv run --all-groups --extra cuda python scripts/benchmark_envs.py --backend jax --output docs/data/envs-port--throughput--jax-gpu.csv
uv run --group benchmark python scripts/generate_environment_report.py

```

All script settings shown above—and the rollout sizes, sample counts, timing threshold,
figure dimensions, and output paths—are adjustable as `tyro` CLI options. Run any script
with `--help` to inspect them.

## Provenance and limitations

Fixtures were captured from the pinned source and are offline test inputs; FastSVERL is
not installed or imported by normal tests. NumPy and JAX PRNG streams are intentionally
not expected to be seed-for-seed identical. This benchmark measures environments only,
without policy inference or learning. Agents, FastSVERL models, replay buffers, training,
and paper evaluation reproduction remain deferred.

Fixture provenance:

```json
{
  "fastsverl_commit": "25b2cbe66b0d4a0decc3b63dbc798a79bf2c7d5b",
  "fixture_sha256": "1c8637568704afaa463ff6b6d20735ba9634a0da5fa51f1e55edf180f352a611",
  "source_sha256": {
    "gridworld": "fa63ba4c4376f0ca3a4e6707c1758739a6fd5084b736ccd230b9188ecaca48bb",
    "hypercube": "af586a54482cd0606a8137ea6df54f0a1c8b594c6e4dfef7fb8c5aacf3361e75",
    "mastermind": "16ae8c47583ddf506ab074c965f5c0099c048a23d9090b3704e257e758663b5a"
  },
  "transition_counts": {
    "gridworld": 20,
    "hypercube": 143052,
    "mastermind_222": 64,
    "total": 143136
  }
}
```
