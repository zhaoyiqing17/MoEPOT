#!/usr/bin/env python3
"""Estimate FT200 duration and cost from completed smoke history."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def smoke_seconds(run_dir: Path) -> list[float]:
    path = run_dir / "history.csv"
    if not path.is_file():
        return []
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return [float(row["epoch_seconds"]) for row in rows if row.get("epoch_seconds")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("/data/nestor_moepot_ns/runs"))
    parser.add_argument("--gpu-price", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result: dict[str, object] = {
        "note": "Estimates use the five-epoch smoke history and include per-epoch evaluation.",
        "gpu_price_per_hour": args.gpu_price,
        "datasets": {},
    }
    datasets: dict[str, object] = {}
    total_hours = 0.0
    for tag in ("ns1e-5", "ns1e-3"):
        values = smoke_seconds(args.runs_root / tag / "smoke")
        if not values:
            datasets[tag] = {"status": "pending", "completed_smoke_epochs": 0}
            continue
        seconds = sum(values) / len(values)
        hours = seconds * 200 / 3600
        total_hours += hours
        datasets[tag] = {
            "status": "estimated",
            "completed_smoke_epochs": len(values),
            "average_seconds_per_epoch": seconds,
            "estimated_ft200_hours": hours,
            "estimated_cost": hours * args.gpu_price if args.gpu_price is not None else None,
        }
    result["datasets"] = datasets
    result["combined_estimated_hours"] = total_hours
    result["combined_estimated_cost"] = (
        total_hours * args.gpu_price if args.gpu_price is not None else None
    )
    output = args.output or args.runs_root / "summary" / "runtime_estimate.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"output: {output}")


if __name__ == "__main__":
    main()
