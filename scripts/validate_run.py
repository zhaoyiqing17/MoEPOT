#!/usr/bin/env python3
"""Validate a completed MoE-POT run before starting another dataset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    args = parser.parse_args()

    required = ["status.json", "history.csv", "training.log", "latest.pth", "final.pth", "best.pth"]
    missing = [name for name in required if not (args.run_dir / name).is_file()]
    if missing:
        raise SystemExit(f"run validation failed: missing {missing}")
    with (args.run_dir / "status.json").open() as stream:
        status = json.load(stream)
    if status.get("state") != "finished":
        raise SystemExit(f"run validation failed: state={status.get('state')}")
    with (args.run_dir / "history.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != args.expected_rows:
        raise SystemExit(
            f"run validation failed: history has {len(rows)} rows, expected {args.expected_rows}"
        )
    last = rows[-1]
    if last.get("test_l2_full") in (None, "", "nan", "NaN"):
        raise SystemExit("run validation failed: final test_l2_full is missing")
    print(
        "RUN_OK",
        f"epoch={last['epoch']}",
        f"test_l2_full={last['test_l2_full']}",
        f"run_dir={args.run_dir}",
    )


if __name__ == "__main__":
    main()
