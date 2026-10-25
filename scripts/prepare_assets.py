#!/usr/bin/env python3
"""Inventory, convert and register both FNO Navier-Stokes datasets."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import h5py

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from link_data import EXPECTED as LINK_EXPECTED, safe_link
from prepare_fno_ns import EXPECTED, Source, detect_layout, write_split


def find_splits(root: Path, tag: str) -> tuple[Path, Path] | None:
    train_path = root / f"ns2d_{tag}_train.hdf5"
    test_path = root / f"ns2d_{tag}_test.hdf5"
    if train_path.is_file() and test_path.is_file():
        return train_path, test_path

    expected_train = LINK_EXPECTED[f"ns2d_{tag}_train.hdf5"]
    expected_test = LINK_EXPECTED[f"ns2d_{tag}_test.hdf5"]
    found_train: list[Path] = []
    found_test: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".h5", ".hdf5"}:
            continue
        if not h5py.is_hdf5(path):
            continue
        try:
            with h5py.File(path, "r") as handle:
                if "data" not in handle:
                    continue
                shape = tuple(handle["data"].shape)
        except Exception:
            continue
        if shape == expected_train:
            found_train.append(path)
        if shape == expected_test:
            found_test.append(path)
    if found_train and found_test:
        return found_train[0], found_test[0]
    return None


def find_raw_source(root: Path, viscosity: str) -> Path | None:
    expected = EXPECTED[viscosity]
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".h5", ".hdf5", ".mat"}:
            continue
        if not h5py.is_hdf5(path):
            continue
        try:
            with h5py.File(path, "r") as handle:
                for variable in ("u", "data"):
                    if variable not in handle or not isinstance(handle[variable], h5py.Dataset):
                        continue
                    _, normalized = detect_layout(tuple(handle[variable].shape), expected["timesteps"])
                    if normalized[0] == expected["n_total"]:
                        return path
        except Exception:
            continue
    return None


def convert(source_path: Path, viscosity: str, output_dir: Path, overwrite: bool) -> tuple[Path, Path]:
    expected = EXPECTED[viscosity]
    tag = "1e-5" if viscosity == "1e-5" else "1e-3"
    variable = "u"
    if h5py.is_hdf5(source_path):
        with h5py.File(source_path, "r") as handle:
            if "u" not in handle and "data" in handle:
                variable = "data"
    source = Source(source_path, variable)
    try:
        layout, normalized = detect_layout(source.shape, expected["timesteps"])
        if normalized[0] != expected["n_total"]:
            raise ValueError(
                f"NS {viscosity}: expected {expected['n_total']} trajectories, got {normalized[0]}"
            )
        train_path = output_dir / f"ns2d_{tag}_train.hdf5"
        test_path = output_dir / f"ns2d_{tag}_test.hdf5"
        write_split(
            source, train_path, layout, 0, 1000, expected["timesteps"],
            25 if viscosity == "1e-5" else 10, None, overwrite,
        )
        write_split(
            source, test_path, layout, normalized[0] - 200, normalized[0],
            expected["timesteps"], 25 if viscosity == "1e-5" else 10, None, overwrite,
        )
        return train_path, test_path
    finally:
        source.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ns1e5-root", type=Path, default=Path("/data/nestor_1e5"))
    parser.add_argument("--ns1e5-source", type=Path)
    parser.add_argument("--ns1e3-source", type=Path)
    parser.add_argument("--work-root", type=Path, default=Path("/data/nestor_moepot_ns"))
    parser.add_argument("--repo-root", type=Path, default=SCRIPT_DIR.parent)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()

    inventory_dir = args.work_root / "inventory"
    inventory_dir.mkdir(parents=True, exist_ok=True)
    preflight = SCRIPT_DIR / "preflight_assets.py"
    subprocess.run(
        [sys.executable, str(preflight), "--root", str(args.ns1e5_root), "--output", str(inventory_dir / "ns1e5.json")],
        check=True,
    )
    raw_root = args.work_root / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, str(preflight), "--root", str(raw_root), "--output", str(inventory_dir / "ns1e3_raw.json")],
        check=True,
    )
    if args.inventory_only:
        print("INVENTORY_OK")
        return

    large_dir = args.work_root / "large"
    large_dir.mkdir(parents=True, exist_ok=True)

    split_1e5 = find_splits(args.ns1e5_root, "1e-5")
    if split_1e5 is None:
        source_1e5 = args.ns1e5_source or find_raw_source(args.ns1e5_root, "1e-5")
        if source_1e5 is None:
            raise SystemExit(
                "NS 1e-5 has no canonical split and no raw u/data array was detected; "
                "pass --ns1e5-source or inspect /data/nestor_moepot_ns/inventory/ns1e5.json"
            )
        split_1e5 = convert(source_1e5, "1e-5", large_dir, args.overwrite)

    source_1e3 = args.ns1e3_source or find_raw_source(raw_root, "1e-3")
    if source_1e3 is None:
        raise SystemExit(
            "NS 1e-3 source not found under /data/nestor_moepot_ns/raw; "
            "download NavierStokes_V1e-3_N5000_T50.mat or pass --ns1e3-source"
        )
    split_1e3 = find_splits(large_dir, "1e-3") or convert(source_1e3, "1e-3", large_dir, args.overwrite)

    sources = {
        "ns2d_1e-5_train.hdf5": split_1e5[0],
        "ns2d_1e-5_test.hdf5": split_1e5[1],
        "ns2d_1e-3_train.hdf5": split_1e3[0],
        "ns2d_1e-3_test.hdf5": split_1e3[1],
    }
    registered = {}
    for name, source in sources.items():
        target = args.repo_root / "data" / "large" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        safe_link(source, target, args.repo_root)
        with h5py.File(source, "r") as handle:
            registered[name] = {
                "source": str(source.resolve()),
                "target": str(target),
                "shape": list(handle["data"].shape),
            }
    metadata_path = args.work_root / "metadata" / "repro_data_links.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(registered, indent=2), encoding="utf-8")
    print(json.dumps(registered, indent=2))
    print(f"metadata: {metadata_path}")
    print("PREPARE_OK")


if __name__ == "__main__":
    main()
