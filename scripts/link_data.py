#!/usr/bin/env python3
"""Register canonical NS HDF5 files at paths expected by make_master_file.py."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py


EXPECTED = {
    "ns2d_1e-5_train.hdf5": (1000, 64, 64, 20),
    "ns2d_1e-5_test.hdf5": (200, 64, 64, 20),
    "ns2d_1e-3_train.hdf5": (1000, 64, 64, 50),
    "ns2d_1e-3_test.hdf5": (200, 64, 64, 50),
}


def verify(path: Path, expected: tuple[int, int, int, int]) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if not h5py.is_hdf5(path):
        raise ValueError(f"{path} is not HDF5")
    with h5py.File(path, "r") as handle:
        if "data" not in handle or not isinstance(handle["data"], h5py.Dataset):
            raise ValueError(f"{path} must contain a dataset named 'data'")
        actual = tuple(handle["data"].shape)
    if actual != expected:
        raise ValueError(f"{path}: expected data shape {expected}, got {actual}")


def safe_link(source: Path, target: Path, repo_root: Path) -> None:
    if target.is_symlink():
        target.unlink()
    elif target.exists():
        raise FileExistsError(
            f"refusing to replace non-symlink file {target}; move it manually before linking"
        )
    target.symlink_to(source.resolve(), target_is_directory=False)
    if repo_root not in target.resolve().parents and target.resolve() != source.resolve():
        raise ValueError(f"link {target} does not resolve to {source}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--ns1e5-train", type=Path, required=True)
    parser.add_argument("--ns1e5-test", type=Path, required=True)
    parser.add_argument("--ns1e3-train", type=Path, required=True)
    parser.add_argument("--ns1e3-test", type=Path, required=True)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    large_dir = repo_root / "data" / "large"
    large_dir.mkdir(parents=True, exist_ok=True)

    sources = {
        "ns2d_1e-5_train.hdf5": args.ns1e5_train,
        "ns2d_1e-5_test.hdf5": args.ns1e5_test,
        "ns2d_1e-3_train.hdf5": args.ns1e3_train,
        "ns2d_1e-3_test.hdf5": args.ns1e3_test,
    }
    registered = {}
    for name, source in sources.items():
        expected = EXPECTED[name]
        verify(source, expected)
        target = large_dir / name
        safe_link(source, target, repo_root)
        registered[name] = {
            "source": str(source.resolve()),
            "target": str(target),
            "shape": list(expected),
        }

    manifest = large_dir / "repro_data_links.json"
    manifest.write_text(json.dumps(registered, indent=2), encoding="utf-8")
    print(json.dumps(registered, indent=2))
    print(f"manifest: {manifest}")


if __name__ == "__main__":
    main()
