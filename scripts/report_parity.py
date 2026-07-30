"""Write exhaustive FastSVERL/JAX transition parity measurements."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import jax

from tests.test_parity import parity_counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    backend = jax.default_backend()
    results = parity_counts()
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
    rows = []
    for environment, counts in results.items():
        rows.append(
            {
                "backend": backend,
                "environment": environment,
                **counts,
                "mismatch_proportion": counts["mismatched"] / counts["tested"],
            }
        )
    aggregate = {
        key: sum(row[key] for row in results.values())
        for key in (
            "tested",
            "mismatched",
            "observation",
            "state",
            "reward",
            "terminated",
            "truncated",
        )
    }
    rows.append(
        {
            "backend": backend,
            "environment": "aggregate",
            **aggregate,
            "mismatch_proportion": aggregate["mismatched"] / aggregate["tested"],
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows for {backend} to {args.output}")


if __name__ == "__main__":
    main()
