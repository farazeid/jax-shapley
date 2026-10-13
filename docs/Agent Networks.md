# Agent networks

The behavioural reference is FastSVERL commit `25b2cbe66b0d4a0decc3b63dbc798a79bf2c7d5b`, particularly `models.py`, `dqn.py`, and `ppo.py`. `jax_shapley.common.networks` holds the layers and MLP; `jax_shapley.algorithms.dqn` the Q-network and `jax_shapley.algorithms.ppo` the PPO actor and critic.

The Flax minimum is 0.12.8, the version validated for the NNX graph containers, parameter access, and abstract tracing APIs used by this implementation. `uv.lock` fixes the tested dependency resolution.

## Architecture and outputs

`MLPConfig` contains an input feature count and a nonempty tuple of `LayerConfig` values. Each layer specifies its width, activation (`identity`, `relu`, `tanh`, or `softmax`) and initialization. Configuration is immutable and validated before tracing. Supported observations use int32 or float32; the model casts them to float32. Parameters and outputs are float32. The final input axis must equal the configured feature count, with arbitrary leading axes preserved.

| Network | Output | Final layer |
| --- | --- | --- |
| `QNetwork` | `QOutput.q_values`, shape `(..., actions)` | Identity |
| `PPOActor` | `PolicyOutput.probabilities`, shape `(..., actions)` | Softmax over actions |
| `PPOCritic` | `ValueOutput.value`, shape `(..., 1)` | One identity output |

The PPO actor emits probabilities, as expected by the source's `Categorical(probs=...)` calls. Actor and critic are independent NNX modules so later PPO can preserve separate optimizers. Q-values are unreduced; greedy policies, tie handling and value reductions belong to the later agent implementation.

`dqn.initialize` builds the ReLU MLP architecture of the supplied experiment scripts from a `DQNConfig`: the environment fixes the input and output sizes and `hidden_widths` the hidden layers. Gridworld and Mastermind-222 use two hidden layers of width 64; Hypercube and larger Mastermind use width 128.

Upstream exposes configurable PPO networks but does not provide paper-specific PPO architecture presets. For example, a supported representative actor configuration is:

```python
from flax import nnx
from jax_shapley.algorithms.ppo import PPOActor
from jax_shapley.common.networks import LayerConfig, MLPConfig, OrthogonalInit

config = MLPConfig(
    input_features=8,
    layers=(
        LayerConfig(16, "tanh", OrthogonalInit(gain=2**0.5)),
        LayerConfig(4, "softmax", OrthogonalInit(gain=0.01)),
    ),
)
actor = PPOActor(config, rngs=nnx.Rngs(0))
```

This example exercises source-supported components; its widths and gains are not presented as paper hyperparameters.

## Initialization and fidelity

`TorchLinearInit` is the default: both kernel and bias are uniform in `[-1/sqrt(fan_in), 1/sqrt(fan_in)]`, matching PyTorch Linear rather than Flax's default initializer. `OrthogonalInit` reproduces the source builder's explicit `layer_init` override, including gain and constant bias. Orthogonal kernels are initialized in PyTorch's `(out, in)` orientation, then transposed into `(in, out)` storage.

Initialization distributions and deterministic JAX key handling are validated; seed-for-seed PyTorch/JAX weight equality is not claimed. All forward/gradient parity tests load the same captured parameters into both implementations. Tolerances account for floating-point differences between backends. Numerical tests do not establish learning equivalence or training speed.

## Validation and fixture reproduction

`tests/fixtures/fastsverl_networks.npz` contains six cases: four source DQN experiment architectures and representative configurable PPO actor/critic architectures. Captured inputs, weights, outputs, scalar probe loss, and input/parameter gradients come directly from the pinned `SimpleNN` builder. The nonuniform probe is `sum(outputs * probe)` so actor gradient checks cannot pass merely because probabilities sum to one. The adjacent JSON records source/fixture hashes, seed, layout, and library versions.

Normal tests only read these committed fixtures. To regenerate them, use an isolated PyTorch environment and a clean checkout of the pinned FastSVERL source:

```bash
uv run --script scripts/fixtures/collect_fastsverl_networks.py \
  /path/to/fastsverl tests/fixtures/fastsverl_networks.npz
```

The script verifies the source commit and `models.py` contents before capture. It imports PyTorch through a typed optional-library adapter and does not add it to the project dependencies.

Run `./check.sh` for the complete gates. Network compilation tests trace abstract observations through NNX forward and gradient boundaries, lower and compile before executing small numerical checks. They cover unbatched, batch, and time/batch shapes for both input dtypes, as well as composed NNX `vmap` and `scan`. GPU compile checks use an available GPU or skip explicitly; CPU success does not establish GPU performance.

## Training configuration

`MLPConfig` remains a network architecture contract. DQN training settings live in
`DQNConfig` and Adam settings in `AdamConfig`; experiments supply them through the
Python configs in `src/jax_shapley/configs/`. See [DQN Training](DQN%20Training.md).

PPO actor/critic networks remain available; PPO training is not ported, as the paper reports DQN agents only.
