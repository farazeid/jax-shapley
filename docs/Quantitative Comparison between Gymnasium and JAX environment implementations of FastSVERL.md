# Quantitative Comparison between Gymnasium and JAX environment implementations of FastSVERL

## Result

The JAX ports match **all 143,136 exhaustive FastSVERL transitions** on both CPU and
GPU. Throughput depends strongly on batch size: GPU has overhead at small batches,
then reaches hundreds of millions of transitions per second at 2,048 environments.

| JAX backend | Transitions tested | Mismatched | Proportion |
| --- | ---: | ---: | ---: |
| CPU | 143,136 | 0 | 0.000000000000 |
| GPU | 143,136 | 0 | 0.000000000000 |

The mismatch criterion is record-level: a transition is different if any next
observation, environment-state field, reward, termination, or truncation value differs.
Component-level counts and every per-configuration result are retained in
`docs/data/envs-port--parity--cpu.csv` and `docs/data/envs-port--parity--gpu.csv`.

## Throughput

Rates below are mean environment transitions per second over five samples. The
Gridworld, Hypercube 2x2, and Mastermind rows use 2,048 environments. Hypercube 6x5
uses 16—the largest safe original point—so all three backends are directly comparable.

| Environment | Workers | Gymnasium CPU | JAX CPU | JAX GPU | CPU speedup | GPU speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gridworld | 2,048 | 216,064 | 8,988,350 | 164,905,947 | 41.6x | 763.2x |
| Hypercube 2x2 | 2,048 | 220,402 | 26,517,196 | 180,082,076 | 120.3x | 817.1x |
| Hypercube 6x5 | 16 | 210,675 | 6,175,486 | 1,626,073 | 29.3x | 7.7x |
| Mastermind-222 | 2,048 | 66,670 | 6,619,422 | 154,320,168 | 99.3x | 2,314.7x |

![Environment throughput](figures/envs-port--throughput.svg)

The complete 1, 2, 4, …, 2,048 curves, five raw samples per successful point, elapsed
times, sample standard deviations, and 95% confidence bounds are available in
`docs/data/`. Each shaded region is the two-sided Student's $t$-distribution interval
$\bar{x} \pm t_{0.975,n-1}s/\sqrt{n}$ around the sample mean. Here $n=5$ and the
critical value has four degrees of freedom. The interval assumes independent,
approximately normally distributed throughput measurements; five sequential samples give
limited evidence for those assumptions. Initial reset, construction, action
generation, transfers, and compilation are excluded. Timed execution includes automatic
resets encountered during 1,024-step rollout blocks.

The original uses FastSVERL's `gym.vector.SyncVectorEnv`. JAX wraps each environment in
`EpisodeStatisticsWrapper`, then uses `jit(vmap(step))` inside `lax.scan`, synchronizing
with `block_until_ready()`. Identical deterministic
action schedules are used. The original implementation has no GPU path.

The original benchmark uses Gymnasium's `RecordEpisodeStatistics`; the JAX benchmark
uses the fixed-shape `EpisodeStatisticsWrapper`:

| Episode statistic or storage | Gymnasium/FastSVERL benchmark | JAX benchmark |
| --- | --- | --- |
| Running return | `episode_returns` (`float32`) | `EpisodeStatisticsState.episode_return` (`float32`) |
| Running length | `episode_lengths` (`int32`) | `EpisodeStatisticsState.episode_length` (`int32`) |
| Completed return and length | `info["episode"]["r"]` and `info["episode"]["l"]` | `info["episode"]["return"]` and `info["episode"]["length"]` |
| Completion indicator | Gymnasium vector mask `info["_episode"]` | Scalar `info["episode"]["completed"]`, vectorized by `vmap` |
| Completed-episode count | `episode_count` | `EpisodeStatisticsState.episode_count` |
| Wall-clock duration | `episode_start_times` and `info["episode"]["t"]` | Omitted because host time is incompatible with pure compiled transitions |
| Recent completed episodes | Per-environment Python deques retaining 100 returns and lengths | Omitted; `lax.scan` retains the fixed-shape masked event stream |

The JAX accumulators reset after completion while the completed return and length are
emitted on that transition. Raw `step_env()` remains the unwrapped FastSVERL parity
boundary.

For Hypercube 6x5, the unmodified source owns a large transition dictionary per worker.
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
- uv lock SHA-256: `50aa45872c12b3fec023defe8c5e3853175860cb0d71f7286e9989cc8617c7ef`
- FastSVERL source commit: `25b2cbe66b0d4a0decc3b63dbc798a79bf2c7d5b`

## Reproduction

Run the complete pipeline from the repository root. It temporarily clones the pinned
FastSVERL commit and removes the checkout on exit:

```bash
bash scripts/envs_port/run.sh
```

The equivalent individual stages are:

```bash
uv sync --all-groups --extra cuda

# Offline semantic tests
JAX_PLATFORMS=cpu uv run --extra cuda pytest
JAX_PLATFORMS=cuda uv run --extra cuda pytest tests/test_env_transition_parity.py -m gpu

# Reference transitions; SOURCE_ROOT is a local checkout at the pinned commit
uv run scripts/envs_port/collect_fastsverl_transitions.py --source-root "$SOURCE_ROOT"

# Parity CSVs are optional pytest outputs
JAX_PLATFORMS=cpu uv run --extra cuda pytest tests/test_env_transition_parity.py -m "not gpu" --parity-output docs/data/envs-port--parity--cpu.csv
JAX_PLATFORMS=cuda uv run --extra cuda pytest tests/test_env_transition_parity.py -m gpu --parity-output docs/data/envs-port--parity--gpu.csv

# Throughput; SOURCE_ROOT is the same pinned local checkout
uv run --group benchmark scripts/envs_port/benchmark_throughput.py --backend original --source-root "$SOURCE_ROOT" --output docs/data/envs-port--throughput--gymnasium-cpu.csv
JAX_PLATFORMS=cpu uv run --all-groups --extra cuda scripts/envs_port/benchmark_throughput.py --backend jax --output docs/data/envs-port--throughput--jax-cpu.csv
JAX_PLATFORMS=cuda uv run --all-groups --extra cuda scripts/envs_port/benchmark_throughput.py --backend jax --output docs/data/envs-port--throughput--jax-gpu.csv

# Derived data and figures
uv run --group benchmark scripts/envs_port/summarise_throughput.py
uv run --group benchmark scripts/envs_port/plot_throughput.py
```

GPU-specific test and benchmark stages log that they were skipped and exit successfully
when no JAX GPU backend is available. Skipped GPU benchmark points are recorded with
`status=gpu_unavailable` and excluded from the generated summary and figure.

All adjustable script settings—including rollout sizes, sample counts, timing thresholds,
figure dimensions, and output paths—are exposed through `tyro`. Run
any entry point with `--help` to inspect them. This Markdown report is maintained by hand;
the scripts regenerate only measurements, the summary CSV, and figures.

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
