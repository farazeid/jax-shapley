"""Characteristic runs on the agent's replay (paper Section 4): buffers, weighting variants and resume."""

import csv
import json
from contextlib import suppress
from pathlib import Path

import jax
import numpy as np
import pytest

from jax_shapley import train_dqn
from jax_shapley.explanation_io import (
    ExplanationSnapshot,
    ExplanationSummary,
    ExplanationWriter,
    load_characteristic,
    parse_run_settings,
)
from jax_shapley.run_io import CallbackToken, load_replay
from jax_shapley.train_explanations import CharacteristicRunConfig, main, parse_args


@pytest.fixture(scope="module")
def agent(tmp_path_factory: pytest.TempPathFactory) -> Path:
    summary = train_dqn.main(_dqn(tmp_path_factory.mktemp("agent")))
    assert summary is not None
    return Path(summary.folder)


def test_replay_buffers_are_shuffled_replay_rows_with_their_behaviour(tmp_path: Path, agent: Path) -> None:
    summary = _single(main(_characteristic(agent, tmp_path, "--off-policy")))
    loaded = load_characteristic(Path(summary.folder))
    behaviour = loaded.buffers.behaviour
    assert behaviour is not None
    replay = load_replay(agent).data
    size = 64
    # The same (observation, action) recurs with different behaviour log-probabilities as epsilon decays.
    rows = {
        (
            tuple(np.asarray(replay.observations[index]).tolist()),
            int(replay.actions[index]),
            round(float(replay.log_probabilities[index]), 5),
        )
        for index in range(size)
    }
    for observations, actions, log_probabilities in (
        (loaded.buffers.train, behaviour.train_actions, behaviour.train_log_probabilities),
        (loaded.buffers.validation, behaviour.validation_actions, behaviour.validation_log_probabilities),
    ):
        for observation, action, log_probability in zip(
            np.asarray(observations).tolist(), np.asarray(actions).tolist(), np.asarray(log_probabilities), strict=True
        ):
            assert (tuple(observation), action, round(float(log_probability), 5)) in rows
    # The train rows and validation rows are disjoint draws from one permutation of all 64 rows.
    assert loaded.buffers.train.shape == (32, 2) and loaded.buffers.validation.shape == (8, 2)
    assert loaded.buffers.exact.shape == (16, 2)  # Fresh rollouts of the explained policy.


@pytest.mark.parametrize("variant", [(), ("--off-policy",), ("--off-policy", "--no-weighting")])
def test_weighting_variants_train(tmp_path: Path, agent: Path, variant: tuple[str, ...]) -> None:
    summary = _single(main(_characteristic(agent, tmp_path, *variant)))
    rows = _rows(Path(summary.folder))
    assert len(rows) == 4 and all(np.isfinite(row).all() for row in rows)
    saved = parse_run_settings((Path(summary.folder) / "config.json").read_text()).config
    assert isinstance(saved, CharacteristicRunConfig)
    assert saved.data == "replay" and saved.off_policy == ("--off-policy" in variant)


def test_replay_run_resumes_to_the_uninterrupted_result(
    tmp_path: Path, agent: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = _single(main(_characteristic(agent, tmp_path / "baseline", "--off-policy")))
    original = ExplanationWriter.checkpoint

    def checkpoint(self: ExplanationWriter, snapshot: ExplanationSnapshot) -> CallbackToken:
        token = original(self, snapshot)
        self.store.wait()
        if int(snapshot.epoch) == 2:
            raise OSError("injected interruption")
        return token

    with monkeypatch.context() as patch:
        patch.setattr(ExplanationWriter, "checkpoint", checkpoint)
        with pytest.raises(Exception, match="injected interruption"):
            main(_characteristic(agent, tmp_path / "interrupted", "--off-policy"))
        with suppress(RuntimeError):
            jax.effects_barrier()  # Clear the expected failed callback effect.
    folder = next((tmp_path / "interrupted").iterdir())
    resumed = _single(main(["resume", "--run", str(folder)]))
    assert _rows(Path(resumed.folder)) == _rows(Path(baseline.folder))
    assert load_characteristic(Path(resumed.folder)).sha256 == load_characteristic(Path(baseline.folder)).sha256


def test_settings_saved_before_replay_data_still_parse(tmp_path: Path, agent: Path) -> None:
    summary = _single(main(_characteristic(agent, tmp_path, rollouts=True)))
    envelope = json.loads((Path(summary.folder) / "config.json").read_text())
    for name in ("data", "off_policy", "weighting", "clip_coef"):
        del envelope["config"][name]
    legacy = parse_run_settings(json.dumps(envelope))
    current = parse_run_settings((Path(summary.folder) / "config.json").read_text())
    assert legacy == current
    assert isinstance(legacy.config, CharacteristicRunConfig) and legacy.config.data == "rollouts"


def test_replay_validation(tmp_path: Path, agent: Path) -> None:
    for invalid in (
        ("--off-policy",),  # off-policy weighting without replay data
        ("--data", "replay", "--train.exact-buffer-size", "None"),  # replay without a separate exact buffer
        ("--data", "replay", "--clip-coef", "-1"),
    ):
        with pytest.raises(SystemExit):  # tyro reports configuration errors and exits.
            arguments = _characteristic(agent, tmp_path, rollouts=True)
            parse_args([*arguments, *invalid])
    with pytest.raises(ValueError, match="transitions in the agent's replay"):
        main([*_characteristic(agent, tmp_path), "--train.buffer-size", "80"])
    assert not list(tmp_path.glob("*characteristic*"))


def _single(result: ExplanationSummary | list[ExplanationSummary] | None) -> ExplanationSummary:
    assert isinstance(result, ExplanationSummary)
    return result


def _dqn(output: Path) -> list[str]:
    return [
        "gridworld",
        "--hidden-widths",
        "8",
        "--chunk-steps",
        "4",
        "--train.num-envs",
        "2",
        "--train.total-timesteps",
        "64",
        "--train.buffer-size",
        "64",
        "--train.batch-size",
        "4",
        "--train.learning-starts",
        "0",
        "--train.train-frequency",
        "1",
        "--output",
        str(output),
    ]


def _characteristic(agent: Path, output: Path, *extra: str, rollouts: bool = False) -> list[str]:
    return [
        "characteristic",
        "--agent",
        str(agent),
        *(() if rollouts else ("--data", "replay")),
        "--hidden-widths",
        "8",
        "--train.epochs",
        "4",
        "--train.buffer-size",
        "40",
        "--train.num-envs",
        "4",
        "--train.batch-size",
        "8",
        "--train.exact-buffer-size",
        "16",
        "--output",
        str(output),
        *extra,
    ]


def _rows(folder: Path) -> list[list[float]]:
    keys = ("epoch", "step", "train_loss", "validation_loss", "exact_loss", "best_epoch")
    with (folder / "metrics.csv").open() as source:
        return [[float(row[key]) for key in keys] for row in csv.DictReader(source)]
