# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["torch==2.6.0", "numpy>=2.0"]
# ///
"""Capture behaviour and prediction explanation units from the pinned FastSVERL source.

Execute the unchanged characteristic, Shapley, sampler and DQN class ASTs with their
numerical dependencies. Weights are seeded, while coalitions, batches, actions and
old log-probabilities are fixed inputs recorded in the fixture, so JAX tests never
need identical Torch/NumPy random streams. Observations come from the committed
upstream transition capture. Never imported by normal pytest runs.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.util
import itertools
import json
import math
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import] -- fixed read-only git operations.
from collections import namedtuple
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Literal, Protocol, cast

import numpy as np

type FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float32]]
type IntArray = np.ndarray[tuple[int, ...], np.dtype[np.int32]]
type BoolArray = np.ndarray[tuple[int, ...], np.dtype[np.bool_]]
type HostArray = FloatArray | IntArray | BoolArray
type Kind = Literal["behaviour", "prediction"]

REFERENCE_COMMIT = "25b2cbe66b0d4a0decc3b63dbc798a79bf2c7d5b"
SOURCES = ("characteristics.py", "shapley.py", "utils.py", "models.py", "dqn.py")
SEED = 1729
EPSILON = 0.2
LEARNING_RATE = 2.5e-4
STEPS = 3
BATCH_SIZE = 8
OFF_POLICY_VARIANTS = (
    ("weighted", True, math.inf),
    ("unweighted", False, math.inf),
    ("weighted_clipped", True, 0.99),
)


@dataclass(frozen=True, slots=True)
class Case:
    """One environment layout: unique upstream states and their exact-buffer multiplicities."""

    name: str
    width: int
    actions: int
    states: IntArray

    @property
    def features(self) -> int:
        return int(self.states.shape[1])


class Tensor(Protocol):
    grad: Tensor | None

    @property
    def shape(self) -> tuple[int, ...]: ...
    def detach(self) -> Tensor: ...
    def cpu(self) -> Tensor: ...
    def numpy(self) -> FloatArray: ...
    def float(self) -> Tensor: ...
    def expand(self, *shape: int) -> Tensor: ...
    def __sub__(self, other: Tensor) -> Tensor: ...
    def __getitem__(self, index: int | slice) -> Tensor: ...


class Torch(Protocol):
    __version__: str
    long: object
    float32: object

    def manual_seed(self, seed: int) -> object: ...
    def set_num_threads(self, count: int) -> None: ...
    def from_numpy(self, values: HostArray) -> Tensor: ...
    def tensor(self, values: list[int], *, dtype: object) -> Tensor: ...
    def device(self, name: str) -> object: ...
    def exp(self, values: Tensor) -> Tensor: ...
    def clamp(self, values: Tensor, low: float, high: float) -> Tensor: ...
    def unique(self, values: Tensor, *, dim: int, return_counts: bool) -> tuple[Tensor, Tensor]: ...


class Model(Protocol):
    def parameters(self) -> Iterator[Tensor]: ...


class Categorical(Protocol):
    probs: Tensor


class Sampler(Protocol):
    categorical: Categorical
    tril: FloatArray
    sample: Callable[..., Tensor]
    sample_rand: Callable[..., Tensor]


class ActionInfo(Protocol):
    logprob: Tensor


class Agent(Protocol):
    model: Model
    target_critic: Model
    epsilon: float

    def pi_greedy(self, obs: Tensor) -> Tensor: ...
    def pi_explore(self, obs: Tensor) -> Tensor: ...
    def V(self, obs: Tensor) -> Tensor: ...  # ruff: ignore[invalid-function-name] -- source method.
    def choose_action(self, obs: Tensor, exp: bool, action: Tensor) -> ActionInfo: ...


class Characteristic(Protocol):
    model: Model
    sampler: Sampler
    agent: Agent
    explore: bool
    clip_coef: float
    exact: dict[tuple[float, ...], Tensor]
    MASK: int

    def predict(self, *inputs: Tensor) -> Tensor: ...
    def v_F(self, obs: Tensor) -> Tensor: ...  # ruff: ignore[invalid-function-name] -- source attribute.
    def null(self, obs: Tensor) -> Tensor: ...
    def null_exact(self, obs: Tensor) -> Tensor: ...
    def get_val(self, e_obs: Tensor, coalitions: Tensor) -> Tensor: ...
    def get_exact(self) -> None: ...
    def get_exact_val(self, e_ob: Tensor, coalition: Tensor) -> Tensor: ...
    def weight_func(self, ratio: Tensor) -> Tensor: ...
    def forward(self, *inputs: Tensor) -> object: ...
    def backward(self, loss: object) -> object: ...
    def exact_loss(self) -> object: ...


class Shapley(Protocol):
    model: Model
    with_exact_char: bool
    exact: dict[tuple[float, ...], Tensor]

    def predict(self, obs: Tensor) -> Tensor: ...
    def normalise(self, obs: Tensor, sv: Tensor) -> Tensor: ...
    def sv(self, obs: Tensor) -> Tensor: ...
    def get_exact(self) -> None: ...
    def forward(self, obs: Tensor) -> object: ...
    def backward(self, loss: object) -> object: ...
    def exact_loss(self) -> object: ...


class Builder(Protocol):
    def SimpleNN(self, name: str, output_shape: tuple[int, ...], **architecture: object) -> Model: ...  # ruff: ignore[invalid-function-name] -- source constructor.


class ExactBuffer:
    """Sequential read-only stand-in for the upstream Buffer, as used by exact computations."""

    def __init__(self, observations: Tensor) -> None:
        self.observations = observations
        self.size = int(observations.shape[0])

    def sample(self, batch_size: int, *keys: str, start: int | None = None) -> list[Tensor]:
        if keys != ("obs",) or start is None or start + batch_size > self.size:
            raise ValueError("exact buffers are read sequentially as observations only")
        return [self.observations[start : start + batch_size]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--transitions", type=Path, default=Path("tests/fixtures/fastsverl_transitions.npz"))
    args = parser.parse_args()
    root, output, transitions = Path(args.source_root), Path(args.output), Path(args.transitions)
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git is required")
    commit = subprocess.check_output([git, "-C", str(root), "rev-parse", "HEAD"], text=True).strip()  # ruff: ignore[subprocess-without-shell-equals-true] -- read-only fixed argv.
    if commit != REFERENCE_COMMIT:
        raise ValueError("unexpected reference commit")
    hashes: dict[str, str] = {}
    for name in SOURCES:
        relative = f"fastsverl/fastsverl/{name}"
        committed = subprocess.check_output([git, "-C", str(root), "show", f"HEAD:{relative}"])  # ruff: ignore[subprocess-without-shell-equals-true] -- read-only fixed argv.
        if committed != (root / relative).read_bytes():
            raise ValueError(f"modified reference {name}")
        hashes[name] = hashlib.sha256(committed).hexdigest()
    transitions_sha256 = hashlib.sha256(transitions.read_bytes()).hexdigest()
    recorded = json.loads(transitions.with_suffix(".json").read_text())["fixture_sha256"]
    if transitions_sha256 != recorded:
        raise ValueError("transition fixture does not match its metadata")
    cases = load_cases(transitions)
    arrays, version = capture(root, cases)
    np.savez_compressed(output, allow_pickle=False, **arrays)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "fastsverl_commit": commit,
                "source_sha256": hashes,
                "transitions_sha256": transitions_sha256,
                "fixture_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "torch_version": version,
                "seed": SEED,
                "epsilon": EPSILON,
                "learning_rate": LEARNING_RATE,
                "steps": STEPS,
                "batch_size": BATCH_SIZE,
                "mask": -2,
                "off_policy_variants": {
                    name: {"weighting": weighting, "clip_coef": "inf" if math.isinf(clip) else clip}
                    for name, weighting, clip in OFF_POLICY_VARIANTS
                },
                "cases": {
                    case.name: {"features": case.features, "actions": case.actions, "width": case.width}
                    for case in cases
                },
                "collector": "unchanged characteristic/Shapley/sampler/DQN ASTs, CPU, fixed coalitions and batches",
            },
            indent=2,
        )
        + "\n"
    )


def load_cases(path: Path) -> tuple[Case, ...]:
    with np.load(path, allow_pickle=False) as transitions:
        grid = np.unique(transitions["grid_state"].astype(np.int32), axis=0)
        boards = transitions["mm_board"].astype(np.int32)
        mastermind = np.unique(boards.reshape(boards.shape[0], -1), axis=0)
        config = transitions["cube_config"]
        cube = np.unique(transitions["cube_state"][(config[:, 0] == 3) & (config[:, 1] == 5)].astype(np.int32), axis=0)
    # Every tenth side-3, dimension-5 state keeps all 2^5 coalitions cheap while spanning the cube.
    return (
        Case("gridworld", 64, 4, grid),
        Case("mastermind222", 64, 4, mastermind),
        Case("hypercube", 128, 10, cube[::10]),
    )


def capture(root: Path, cases: tuple[Case, ...]) -> tuple[dict[str, HostArray], str]:
    torch = cast(Torch, importlib.import_module("torch"))  # Optional reference-only boundary.
    namespace = _reference_namespace(root, torch)
    torch.set_num_threads(1)
    # Upstream ShapleySampler seeds its own generator from the legacy global stream; draws are never recorded.
    np.random.seed(SEED)
    arrays: dict[str, HostArray] = {}
    for case in cases:
        arrays.update(capture_case(case, torch, namespace))
    return arrays, torch.__version__


def capture_case(case: Case, torch: Torch, namespace: dict[str, object]) -> dict[str, HostArray]:
    arrays: dict[str, HostArray] = {}
    prefix = case.name
    env = SimpleNamespace(
        single_observation_space=SimpleNamespace(shape=(case.features,)),
        single_action_space=SimpleNamespace(n=case.actions),
    )
    multiplicities = (1 + (np.arange(case.states.shape[0]) * 5) % 4).astype(np.int32)
    buffer_states = exact_buffer_states(case.states, multiplicities)
    obs = torch.from_numpy(buffer_states.astype(np.float32))
    batch = obs[:BATCH_SIZE]
    arrays[f"{prefix}_states"] = case.states
    arrays[f"{prefix}_multiplicities"] = multiplicities
    arrays[f"{prefix}_exact_buffer"] = buffer_states
    unique, counts = torch.unique(obs, dim=0, return_counts=True)
    arrays[f"{prefix}_unique"] = _array(unique).astype(np.int32)
    arrays[f"{prefix}_counts"] = counts.numpy().astype(np.int32)

    agent = build_agent(case, torch, namespace, env)
    for index, parameter in enumerate(agent.model.parameters()):
        arrays[f"{prefix}_agent_{index}"] = _array(parameter)
    arrays[f"{prefix}_agent_greedy"] = _array(agent.pi_greedy(unique))
    arrays[f"{prefix}_agent_explore"] = _array(agent.pi_explore(unique))
    arrays[f"{prefix}_agent_value"] = _array(agent.V(unique))

    coalitions = all_coalitions(case.features)
    arrays[f"{prefix}_coalitions"] = coalitions
    buffer = ExactBuffer(obs)
    exact_chars: dict[str, Characteristic] = {}
    exact_cases: tuple[tuple[str, Kind, str], ...] = (
        ("behaviour", "behaviour", "greedy"),
        ("behaviour_explore", "behaviour", "explore"),
        ("prediction", "prediction", "greedy"),
    )
    for label, kind, policy in exact_cases:
        char = build_characteristic(case, torch, namespace, env, agent, buffer, kind, policy)
        char.get_exact()
        table = [
            [
                _array(char.get_exact_val(state, torch.from_numpy(coalition.astype(np.int32))))
                for state in unique_rows(unique)
            ]
            for coalition in coalitions
        ]
        arrays[f"{prefix}_exact_{label}"] = np.asarray(table, np.float32)
        arrays[f"{prefix}_exact_{label}_null"] = _array(char.null_exact(unique))
        exact_chars[label] = char

    kinds: tuple[Kind, ...] = ("behaviour", "prediction")
    for kind in kinds:
        name = f"{prefix}_char_{kind}"
        char = exact_chars[kind]
        arrays[f"{name}_sampler_probs"] = _array(char.sampler.categorical.probs)
        arrays[f"{name}_sampler_tril"] = np.asarray(char.sampler.tril, np.float32)
        initial = _parameters("", char.model)
        arrays.update(train_characteristic(name, char, batch, case, torch))
        arrays[f"{name}_exact_loss"] = _array(_loss(char.exact_loss()))
        # Learnt grand-coalition and null values after training, the inputs to learnt normalisation.
        grand = torch.from_numpy(np.ones(tuple(unique.shape), np.int32))
        arrays[f"{name}_grand"] = _array(char.get_val(unique, grand))
        arrays[f"{name}_null"] = _array(char.null(unique[0:1])[0])
        for variant, weighting, clip in OFF_POLICY_VARIANTS:
            off_policy = build_characteristic(
                case, torch, namespace, env, agent, buffer, kind, "greedy", weighting=weighting, clip_coef=clip
            )
            arrays.update(train_off_policy(f"{name}_off_{variant}", off_policy, initial, batch, case, torch))
        arrays.update(train_shapley(f"{prefix}_shapley_{kind}", kind, char, batch, unique, case, torch, namespace, env))
    return arrays


def build_agent(case: Case, torch: Torch, namespace: dict[str, object], env: SimpleNamespace) -> Agent:
    cls = cast(type[Agent], namespace["DQN"])  # Source class adapter.
    builder = cast(Builder, namespace["models"])
    torch.manual_seed(SEED)
    agent = object.__new__(cls)
    agent.model = builder.SimpleNN("DQN", (case.actions,), **_architecture(case, softmax=False))
    agent.target_critic = builder.SimpleNN("DQN", (case.actions,), **_architecture(case, softmax=False))
    args = SimpleNamespace(start_e=EPSILON, learning_rate=LEARNING_RATE)
    # Invoke unchanged initialization on the prepared models, without a replay buffer.
    init = cast(Callable[[Agent, SimpleNamespace, SimpleNamespace], None], namespace["DQN_Super"].__init__)
    init(agent, env, args)
    return agent


def build_characteristic(
    case: Case,
    torch: Torch,
    namespace: dict[str, object],
    env: SimpleNamespace,
    agent: Agent,
    buffer: ExactBuffer,
    kind: Kind,
    policy: str,
    *,
    weighting: bool | None = None,
    clip_coef: float | None = None,
) -> Characteristic:
    names = {"behaviour": "PolicyCharacteristicModel", "prediction": "ValueCharacteristicModel"}
    cls = cast(Callable[..., Characteristic], namespace[names[kind]])  # Source class adapter.
    char_args = SimpleNamespace(
        model_arch=_architecture(case, softmax=kind == "behaviour"),
        learning_rate=LEARNING_RATE,
        policy=policy,
        verbose=False,
    )
    if weighting is not None:
        char_args.off_policy = True
        char_args.weighting = weighting
        char_args.clip_coef = clip_coef
    # One seed for every characteristic, so off-policy variants share the on-policy initial weights.
    torch.manual_seed(SEED + 1)
    return cls(agent, env, char_args, None, None, buffer)


def train_characteristic(
    name: str, char: Characteristic, batch: Tensor, case: Case, torch: Torch
) -> dict[str, HostArray]:
    arrays = _parameters(f"{name}_initial", char.model)
    arrays[f"{name}_targets"] = _array(char.v_F(batch))
    for step in range(STEPS):
        coalitions = characteristic_coalitions(BATCH_SIZE, case.features, step)
        arrays[f"{name}_coalitions_{step}"] = coalitions
        masks = torch.from_numpy(coalitions.astype(np.int32))
        char.sampler.sample_rand = _fixed(masks)
        arrays[f"{name}_predictions_{step}"] = _array(char.get_val(batch, masks))
        loss = char.forward(batch)
        char.backward(loss)
        arrays[f"{name}_loss_{step}"] = _array(_loss(loss))
        if step == 0:
            arrays.update(_gradients(f"{name}_gradient", char.model))
    arrays.update(_parameters(f"{name}_final", char.model))
    return arrays


def train_off_policy(
    name: str, char: Characteristic, initial: dict[str, HostArray], batch: Tensor, case: Case, torch: Torch
) -> dict[str, HostArray]:
    """Shares the on-policy initial weights and coalitions; records only what the loss changes."""
    if any(not np.array_equal(value, initial[key]) for key, value in _parameters("", char.model).items()):
        raise RuntimeError("off-policy characteristic must start from the on-policy weights")
    arrays: dict[str, HostArray] = {}
    actions = torch.tensor([index % case.actions for index in range(BATCH_SIZE)], dtype=torch.long)
    old_logprobs = torch.from_numpy(np.log(np.linspace(0.2, 0.9, BATCH_SIZE, dtype=np.float32)))
    arrays[f"{name}_actions"] = np.arange(BATCH_SIZE, dtype=np.int32) % case.actions
    arrays[f"{name}_old_logprobs"] = _array(old_logprobs)
    new_logprobs = char.agent.choose_action(batch, exp=char.explore, action=actions).logprob
    ratio = char.weight_func(torch.exp(new_logprobs - old_logprobs))
    arrays[f"{name}_new_logprobs"] = _array(new_logprobs)
    arrays[f"{name}_weights"] = _array(torch.clamp(ratio, 1 - char.clip_coef, 1 + char.clip_coef))
    for step in range(STEPS):
        coalitions = characteristic_coalitions(BATCH_SIZE, case.features, step)
        char.sampler.sample_rand = _fixed(torch.from_numpy(coalitions.astype(np.int32)))
        loss = char.forward(batch, actions, old_logprobs)
        char.backward(loss)
        arrays[f"{name}_loss_{step}"] = _array(_loss(loss))
        if step == 0:
            arrays.update(_gradients(f"{name}_gradient", char.model))
    arrays.update(_parameters(f"{name}_final", char.model))
    return arrays


def train_shapley(
    name: str,
    kind: Kind,
    char: Characteristic,
    batch: Tensor,
    unique: Tensor,
    case: Case,
    torch: Torch,
    namespace: dict[str, object],
    env: SimpleNamespace,
) -> dict[str, HostArray]:
    names = {"behaviour": "PolicyShapley", "prediction": "ValueShapley"}
    cls = cast(Callable[..., Shapley], namespace[names[kind]])  # Source class adapter.
    sv_args = SimpleNamespace(
        model_arch=_architecture(case, softmax=False), learning_rate=LEARNING_RATE, with_exact_char=True, verbose=False
    )
    torch.manual_seed(SEED + 2)
    shapley = cls(env, sv_args, char)
    arrays = _parameters(f"{name}_initial", shapley.model)
    for step in range(STEPS):
        coalition = shapley_coalition(case.features, step)
        arrays[f"{name}_coalition_{step}"] = coalition
        masks = torch.from_numpy(coalition[None].astype(np.int32))
        char.sampler.sample = _fixed(masks)
        arrays[f"{name}_targets_{step}"] = _array(char.get_val(batch, masks.expand(BATCH_SIZE, -1)) - char.null(batch))
        loss = shapley.forward(batch)
        shapley.backward(loss)
        arrays[f"{name}_loss_{step}"] = _array(_loss(loss))
        if step == 0:
            arrays.update(_gradients(f"{name}_gradient", shapley.model))
    arrays.update(_parameters(f"{name}_final", shapley.model))
    raw = shapley.predict(batch).detach()
    arrays[f"{name}_raw"] = _array(raw)
    arrays[f"{name}_normalised"] = _array(shapley.normalise(batch, raw))
    arrays[f"{name}_sv_unique"] = _array(shapley.sv(unique))
    for label, with_exact_char in (("with_exact_char", True), ("without_exact_char", False)):
        # Equivalent to constructing with sv_args.with_exact_char; clearing the cache recomputes.
        shapley.with_exact_char = with_exact_char
        vars(shapley).pop("exact", None)
        shapley.get_exact()
        arrays[f"{name}_exact_{label}"] = np.stack([_array(shapley.exact[key]) for key in _keys(unique)])
        arrays[f"{name}_exact_loss_{label}"] = _array(_loss(shapley.exact_loss()))
    return arrays


def exact_buffer_states(states: IntArray, multiplicities: IntArray) -> IntArray:
    """Interleave states so each appears its multiplicity times, without sorting the buffer."""
    rows = [
        states[index]
        for repeat in range(int(multiplicities.max()))
        for index in np.flatnonzero(multiplicities > repeat)
    ]
    return np.asarray(rows, np.int32)


def all_coalitions(features: int) -> BoolArray:
    """Matches torch.cartesian_prod over [0, 1] per feature: the first feature varies slowest."""
    return np.asarray(list(itertools.product((0, 1), repeat=features)), np.bool_)


def characteristic_coalitions(batch_size: int, features: int, step: int) -> BoolArray:
    rows, columns = np.meshgrid(np.arange(batch_size), np.arange(features), indexing="ij")
    return ((rows * 3 + columns * 5 + step * 7) % 5) < 2 + step % 2


def shapley_coalition(features: int, step: int) -> BoolArray:
    """A proper nonempty subset, as the Shapley-kernel sampler never draws the empty or grand coalition."""
    return (np.arange(features) + step) % 2 == 0


def unique_rows(unique: Tensor) -> list[Tensor]:
    return [unique[index] for index in range(int(unique.shape[0]))]


def _reference_namespace(root: Path, torch: Torch) -> dict[str, object]:
    package = root / "fastsverl/fastsverl"
    models = _load(package / "models.py", "reference_models")
    namespace: dict[str, object] = {
        "torch": torch,
        "np": np,
        "itertools": itertools,
        "namedtuple": namedtuple,  # ruff: ignore[collections-named-tuple] -- unchanged upstream runtime tuple factory.
        "comb": math.comb,
        "tqdm": _passthrough,
        "F": importlib.import_module("torch.nn.functional"),
        "Categorical": importlib.import_module("torch.distributions").Categorical,
        "DEVICE": torch.device("cpu"),
        "models": models,
        "SimpleNN": models.SimpleNN,
        "ImportanceWeightedMSELoss": models.ImportanceWeightedMSELoss,
    }
    for name, classes in (
        ("utils.py", ("ShapleySampler",)),
        ("dqn.py", ("DQN_Super", "DQN")),
        (
            "characteristics.py",
            (
                "Characteristic",
                "CharacteristicModel",
                "ValuePolicyCharacteristic",
                "ValuePolicyCharacteristicModel",
                "ValueCharacteristicModel",
                "PolicyCharacteristicModel",
            ),
        ),
        ("shapley.py", ("Shapley", "PolicyShapley", "ValueShapley")),
    ):
        path = package / name
        tree = ast.parse(path.read_text())
        definitions: list[ast.stmt] = [
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in classes
        ]
        if len(definitions) != len(classes):
            raise RuntimeError(f"missing reference classes in {name}")
        code = compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec")
        exec(code, namespace)  # ruff: ignore[exec-builtin] -- verified pinned source, no external configuration code.
    return namespace


def _architecture(case: Case, *, softmax: bool) -> dict[str, list[object]]:
    """The experiment-script layout; upstream fills the None input/output sizes in place."""
    width = case.width
    architecture: dict[str, list[object]] = {
        "input": ["Linear", [case.features, width]],
        "input_activation": ["ReLU", []],
        "hidden1": ["Linear", [width, width]],
        "hidden1_activation": ["ReLU", []],
        "output": ["Linear", [width, None]],
    }
    if softmax:
        architecture["output_activation"] = ["Softmax", [-1]]
    return architecture


def _parameters(name: str, model: Model) -> dict[str, HostArray]:
    return {f"{name}_{index}": _array(parameter) for index, parameter in enumerate(model.parameters())}


def _gradients(name: str, model: Model) -> dict[str, HostArray]:
    arrays: dict[str, HostArray] = {}
    for index, parameter in enumerate(model.parameters()):
        if parameter.grad is None:
            raise RuntimeError("reference gradient missing")
        arrays[f"{name}_{index}"] = _array(parameter.grad)
    return arrays


def _fixed(coalitions: Tensor) -> Callable[..., Tensor]:
    def sample(*_: object, **__: object) -> Tensor:
        return coalitions

    return sample


def _keys(unique: Tensor) -> list[tuple[float, ...]]:
    return [tuple(float(value) for value in row) for row in _array(unique)]


def _loss(losses: object) -> Tensor:
    # Upstream returns a one-field namedtuple named after the model class.
    return cast(Tensor, cast(tuple[object, ...], losses)[0])


def _passthrough[T](iterable: T, *_: object) -> T:
    return iterable


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _array(tensor: Tensor) -> FloatArray:
    return tensor.detach().cpu().numpy().astype(np.float32)


if __name__ == "__main__":
    main()
