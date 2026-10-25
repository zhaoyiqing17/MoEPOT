#!/usr/bin/env python3
"""Collect zero-shot and FT200 L2RE logs into JSON, CSV and Markdown."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


REFERENCES = {
    "ns1e-5": {
        "id": "ns2d_fno_1e-5",
        "zero_shot": 0.0682,
        "ft200": 0.0456,
    },
    "ns1e-3": {
        "id": "ns2d_fno_1e-3",
        "zero_shot": 0.00768,
        "ft200": 0.00493,
    },
}

ZERO_RE = re.compile(r"^(ns2d_fno_1e-5|ns2d_fno_1e-3):\s*([0-9.eE+-]+)\s*$", re.MULTILINE)
FT_RE = re.compile(
    r"epoch\s+(\d+).*?test l2 full\s+\[([^\]]+)\]",
    re.IGNORECASE,
)


def read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def parse_zero(log_path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    for dataset_id, raw_value in ZERO_RE.findall(read_text(log_path)):
        values[dataset_id] = float(raw_value)
    return values


def parse_finetune(run_dir: Path) -> dict[str, Any] | None:
    candidates = [run_dir / "logs.txt", run_dir / "nohup.out"]
    matches: list[tuple[int, float]] = []
    for path in candidates:
        for raw_epoch, raw_value in FT_RE.findall(read_text(path)):
            values = [item.strip() for item in raw_value.split(",")]
            if values:
                matches.append((int(raw_epoch), float(values[0])))
    if not matches:
        return None
    return {"epoch": matches[-1][0], "l2re": matches[-1][1]}


def relative_error(value: float, reference: float) -> float:
    return abs(value - reference) / reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("/data/nestor_moepot_ns/runs"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    runs_root = args.runs_root.resolve()
    output_dir = (args.output_dir or runs_root / "summary").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    zero_log = runs_root / "zero_shot" / "zero_shot.log"
    zero_values = parse_zero(zero_log)
    rows: list[dict[str, Any]] = []

    for tag, reference in REFERENCES.items():
        dataset_id = reference["id"]
        zero = zero_values.get(dataset_id)
        ft_dir = runs_root / "finetune" / tag / "full"
        ft = parse_finetune(ft_dir)
        ft_value = ft["l2re"] if ft else None

        zero_error = relative_error(zero, reference["zero_shot"]) if zero is not None else None
        ft_error = relative_error(ft_value, reference["ft200"]) if ft_value is not None else None
        warning = None
        if zero_error is not None and zero_error > 0.01:
            warning = "zero-shot relative error > 1%"
        if ft_error is not None and ft_error > 0.05:
            warning = (
                f"{warning}; FT200 relative error > 5%" if warning else "FT200 relative error > 5%"
            )

        rows.append(
            {
                "dataset": dataset_id,
                "zero_shot_l2re": zero,
                "zero_shot_reference": reference["zero_shot"],
                "zero_shot_relative_error": zero_error,
                "ft200_l2re": ft_value,
                "ft200_epoch": ft["epoch"] if ft else None,
                "ft200_reference": reference["ft200"],
                "ft200_relative_error": ft_error,
                "warning": warning,
            }
        )

    payload = {
        "runs_root": str(runs_root),
        "zero_shot_log": str(zero_log),
        "rows": rows,
        "overall_warning": any(row["warning"] for row in rows),
        "warnings_are_advisory": True,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def fmt(value: Any, digits: int = 5) -> str:
        if value is None:
            return "pending"
        if isinstance(value, float):
            return f"{value:.{digits}g}"
        return str(value)

    lines = [
        "# MoE-POT-Tiny NS 复现结果",
        "",
        f"- runs root: `{runs_root}`",
        "- Thresholds are advisory; raw values are the primary result.",
        "",
        "| Dataset | Zero-shot | Paper | Relative error | FT200 | Paper | Relative error | Warning |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {zero} | {zero_ref} | {zero_err} | {ft} | {ft_ref} | {ft_err} | {warning} |".format(
                dataset=row["dataset"],
                zero=fmt(row["zero_shot_l2re"]),
                zero_ref=fmt(row["zero_shot_reference"]),
                zero_err=fmt(row["zero_shot_relative_error"]),
                ft=fmt(row["ft200_l2re"]),
                ft_ref=fmt(row["ft200_reference"]),
                ft_err=fmt(row["ft200_relative_error"]),
                warning=row["warning"] or "none",
            )
        )
    lines.extend(
        [
            "",
            "FT200 must come from `finetune/<dataset>/full/logs.txt`; smoke logs are ignored.",
            "A pending value means the corresponding run has not finished or its log is missing.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"outputs: {output_dir}")


if __name__ == "__main__":
    main()
