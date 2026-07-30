"""Write exhaustive FastSVERL/JAX transition parity measurements."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import jax
import tyro


@dataclass(frozen=True)
class Args:
    """Configuration for an exhaustive parity report."""

    output: Path = Path("docs/data/envs-port--parity--cpu.csv")


def main() -> None:
    args = tyro.cli(Args)
    from tests.test_parity import parity_counts

    backend = jax.default_backend()
    results = parity_counts()
    rows = transition_rows(backend, results)
    write_rows(args.output, rows)
    print(f"Wrote {len(rows)} rows for {backend} to {args.output}")


def transition_rows(backend: str, results: dict[str, dict[str, int]]) -> list[dict[str, object]]:
    rows = [
        {
            "backend": backend,
            "environment": environment,
            **counts,
            "mismatch_proportion": counts["mismatched"] / counts["tested"],
        }
        for environment, counts in results.items()
    ]
    count_fields = (
        "tested",
        "mismatched",
        "observation",
        "state",
        "reward",
        "terminated",
        "truncated",
    )
    aggregate = {key: sum(row[key] for row in results.values()) for key in count_fields}
    rows.append(
        {
            "backend": backend,
            "environment": "aggregate",
            **aggregate,
            "mismatch_proportion": aggregate["mismatched"] / aggregate["tested"],
        }
    )
    return rows


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "backend",
        "environment",
        "tested",
        "mismatched",
        "mismatch_proportion",
        "observation",
        "state",
        "reward",
        "terminated",
        "truncated",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
