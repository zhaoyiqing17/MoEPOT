#!/usr/bin/env python3
"""Estimate full FT200 duration and cost from five-epoch smoke logs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


EPOCH_RE = re.compile(r"epoch\s+\d+,\s*time\s+([0-9.eE+-]+)")


def parse_epochs(run_dir: Path) -> list[float]:
    log = run_dir / "logs.txt"
    if not log.is_file():
        return []
    text = log.read_text(encoding="utf-8", errors="replace")
    return [float(value) for value in EPOCH_RE.findall(text)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("/data/nestor_moepot_ns/runs"))
    parser.add_argument("--gpu-price", type=float, default=None, help="BitaHub price per GPU-hour")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.runs_root / "finetune"
    result = {
        "note": "Estimate uses completed smoke epochs only; formal test overhead is included.",
        "gpu_price_per_hour": args.gpu_price,
        "datasets": {},
    }
    combined_hours = 0.0
    for tag in ("ns1e-5", "ns1e-3"):
        values = parse_epochs(root / tag / "smoke")
        if not values:
            result["datasets"][tag] = {"status": "pending", "completed_smoke_epochs": 0}
            continue
        average = sum(values) / len(values)
        hours = average * 200 / 3600
        combined_hours += hours
        result["datasets"][tag] = {
            "status": "estimated",
            "completed_smoke_epochs": len(values),
            "average_seconds_per_epoch": average,
            "estimated_ft200_hours": hours,
            "estimated_cost": hours * args.gpu_price if args.gpu_price is not None else None,
        }

    result["combined_estimated_hours"] = combined_hours
    result["combined_estimated_cost"] = (
        combined_hours * args.gpu_price if args.gpu_price is not None else None
    )
    output = args.output or args.runs_root / "summary" / "runtime_estimate.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"output: {output}")


if __name__ == "__main__":
    main()
