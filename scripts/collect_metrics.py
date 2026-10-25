#!/usr/bin/env python3
"""Collect zero-shot and FT200 results into JSON, CSV and Markdown."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REFERENCES = {
    "ns1e-5": {"id": "ns2d_fno_1e-5", "zero_shot": 0.0682, "ft200": 0.0456},
    "ns1e-3": {"id": "ns2d_fno_1e-3", "zero_shot": 0.00768, "ft200": 0.00493},
}


def relative_error(value: float, reference: float) -> float:
    return abs(value - reference) / reference


def read_last_history(run_dir: Path) -> dict[str, str] | None:
    path = run_dir / "history.csv"
    if not path.is_file():
        return None
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return rows[-1] if rows else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("/data/nestor_moepot_ns/runs"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    runs_root = args.runs_root.resolve()
    output_dir = (args.output_dir or runs_root / "summary").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    zero_path = runs_root / "zero_shot" / "pretrained_eval.json"
    zero_payload = json.loads(zero_path.read_text()) if zero_path.is_file() else {}
    zero_values = zero_payload.get("test_l2re", {})

    rows: list[dict[str, Any]] = []
    for tag, reference in REFERENCES.items():
        zero = zero_values.get(reference["id"])
        last = read_last_history(runs_root / tag / "ft200")
        ft_value = float(last["test_l2_full"]) if last else None
        ft_epoch = int(last["epoch"]) if last else None
        zero_error = relative_error(zero, reference["zero_shot"]) if zero is not None else None
        ft_error = relative_error(ft_value, reference["ft200"]) if ft_value is not None else None
        warning = []
        if zero_error is not None and zero_error > 0.01:
            warning.append("zero-shot relative error > 1%")
        if ft_error is not None and ft_error > 0.05:
            warning.append("FT200 relative error > 5%")
        rows.append(
            {
                "dataset": reference["id"],
                "zero_shot_l2re": zero,
                "zero_shot_reference": reference["zero_shot"],
                "zero_shot_relative_error": zero_error,
                "ft200_l2re": ft_value,
                "ft200_epoch": ft_epoch,
                "ft200_reference": reference["ft200"],
                "ft200_relative_error": ft_error,
                "warning": "; ".join(warning) or None,
            }
        )

    payload = {
        "runs_root": str(runs_root),
        "zero_shot": str(zero_path),
        "rows": rows,
        "warnings_are_advisory": True,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def fmt(value: Any) -> str:
        if value is None:
            return "pending"
        if isinstance(value, float):
            return f"{value:.6g}"
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
            "Zero-shot comes from `runs/zero_shot/pretrained_eval.json`.",
            "FT200 comes from `runs/<dataset>/ft200/history.csv`; smoke runs are ignored.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"outputs: {output_dir}")


if __name__ == "__main__":
    main()
