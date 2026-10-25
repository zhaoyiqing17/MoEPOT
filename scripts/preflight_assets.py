#!/usr/bin/env python3
"""Inspect local/BitaHub data files without loading full trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    import h5py
except ImportError as exc:
    raise SystemExit("h5py is required; run scripts/setup_env_moepot.sh first") from exc


SUFFIXES = {".h5", ".hdf5", ".mat"}


def _dataset_entries(handle: h5py.File, max_datasets: int = 40) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    def visitor(name: str, obj: Any) -> None:
        if len(entries) >= max_datasets or not isinstance(obj, h5py.Dataset):
            return
        entries.append(
            {
                "path": name,
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
            }
        )

    handle.visititems(visitor)
    return entries


def inspect_file(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
    }

    if h5py.is_hdf5(path):
        result["container"] = "hdf5"
        try:
            with h5py.File(path, "r") as handle:
                result["keys"] = list(handle.keys())
                result["datasets"] = _dataset_entries(handle)
                result["status"] = "ok"
        except Exception as exc:
            result["status"] = "error"
            result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    if path.suffix.lower() == ".mat":
        result["container"] = "mat-v5-or-unknown"
        try:
            from scipy.io import whosmat

            variables = whosmat(path)
            result["variables"] = [
                {"name": name, "shape": list(shape), "class": klass}
                for name, shape, klass in variables
            ]
            result["status"] = "ok"
        except Exception as exc:
            result["status"] = "error"
            result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    result["container"] = "unknown"
    result["status"] = "unsupported"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", required=True, help="Directory to scan; repeat as needed")
    parser.add_argument("--output", type=Path, default=Path("asset_inventory.json"))
    args = parser.parse_args()

    files: list[Path] = []
    missing_roots: list[str] = []
    for raw_root in args.root:
        root = Path(raw_root)
        if not root.exists():
            missing_roots.append(str(root))
            continue
        if root.is_file() and root.suffix.lower() in SUFFIXES:
            files.append(root)
        elif root.is_dir():
            files.extend(
                path
                for path in root.rglob("*")
                if path.is_file() and path.suffix.lower() in SUFFIXES
            )

    inventory = {
        "roots": args.root,
        "missing_roots": missing_roots,
        "files": [inspect_file(path) for path in sorted(set(files))],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(inventory, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"inventory: {args.output.resolve()}")
    print(f"found: {len(inventory['files'])} HDF5/MAT file(s)")
    for entry in inventory["files"]:
        summary = entry.get("datasets") or entry.get("variables") or entry.get("keys")
        print(f"- {entry['path']} [{entry['status']}] {summary}")
    if missing_roots:
        print("missing roots:")
        for root in missing_roots:
            print(f"- {root}")


if __name__ == "__main__":
    main()
