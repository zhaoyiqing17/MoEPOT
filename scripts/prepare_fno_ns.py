#!/usr/bin/env python3
"""Convert FNO Navier-Stokes data into the HDF5 layout expected by MoE-POT.

The source may be an HDF5-based MAT v7.3 file, a regular HDF5 file, or a MAT
v5 file. Output trajectories always have shape (N, 64, 64, T), matching
MixedTemporalDataset before it adds the singleton channel dimension.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import h5py
except ImportError as exc:
    raise SystemExit("h5py is required; run scripts/setup_env_moepot.sh first") from exc


EXPECTED = {
    "1e-5": {"n_total": 1200, "timesteps": 20},
    "1e-3": {"n_total": 5000, "timesteps": 50},
}


class Source:
    def __init__(self, path: Path, variable: str):
        self.path = path
        self.handle: Any = None
        self.array: np.ndarray | None = None
        self.dataset: Any = None

        if h5py.is_hdf5(path):
            self.handle = h5py.File(path, "r")
            if variable not in self.handle:
                available = ", ".join(self.handle.keys())
                raise ValueError(f"variable {variable!r} not found in {path}; keys: {available}")
            self.dataset = self.handle[variable]
            if not isinstance(self.dataset, h5py.Dataset):
                raise ValueError(f"{variable!r} must be an HDF5 dataset")
            self.shape = tuple(self.dataset.shape)
            self.dtype = self.dataset.dtype
            return

        if path.suffix.lower() != ".mat":
            raise ValueError(f"unsupported source type: {path}")

        from scipy.io import loadmat

        loaded = loadmat(path, variable_names=[variable])
        if variable not in loaded:
            raise ValueError(f"variable {variable!r} not found in MAT file")
        self.array = np.asarray(loaded[variable])
        self.shape = tuple(self.array.shape)
        self.dtype = self.array.dtype

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()

    def read(self, start: int, stop: int, layout: str) -> np.ndarray:
        if layout in {"hwtn", "thwn"}:
            chunk = (
                self.dataset[:, :, :, start:stop]
                if self.dataset is not None
                else self.array[:, :, :, start:stop]
            )
            if chunk.ndim == 5 and chunk.shape[-1] == 1:
                chunk = chunk[..., 0]
            if layout == "thwn":
                chunk = np.transpose(chunk, (3, 1, 2, 0))
            else:
                chunk = np.transpose(chunk, (3, 0, 1, 2))
        else:
            chunk = self.dataset[start:stop] if self.dataset is not None else self.array[start:stop]
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim == 5 and chunk.shape[-1] == 1:
            chunk = chunk[..., 0]
        return np.ascontiguousarray(chunk)


def detect_layout(shape: tuple[int, ...], timesteps: int) -> tuple[str, tuple[int, int, int, int]]:
    if len(shape) == 5 and shape[-1] == 1:
        shape = shape[:-1]
    if len(shape) != 4:
        raise ValueError(f"expected a 4-D source array (or trailing singleton), got {shape}")

    if shape[0] == timesteps and shape[1] == 64 and shape[2] == 64:
        return "thwn", (shape[3], 64, 64, timesteps)
    if shape[0] == 64 and shape[1] == 64 and shape[2] == timesteps:
        return "hwtn", (shape[3], 64, 64, timesteps)
    if shape[1] == timesteps and shape[2:] == (64, 64):
        return "nthw", (shape[0], 64, 64, timesteps)
    if shape[1:3] == (64, 64) and shape[3] == timesteps:
        return "nhwt", (shape[0], 64, 64, timesteps)
    raise ValueError(
        f"cannot infer FNO layout from source shape {shape}; expected "
        f"({timesteps},64,64,N), (64,64,{timesteps},N), "
        f"(N,{timesteps},64,64), or (N,64,64,{timesteps})"
    )


def write_split(
    source: Source,
    output: Path,
    layout: str,
    start: int,
    stop: int,
    timesteps: int,
    chunk_size: int,
    compression: str | None,
    overwrite: bool,
) -> dict[str, Any]:
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)
    expected_count = stop - start
    created = False
    try:
        with h5py.File(output, "w") as target:
            created = True
            data = target.create_dataset(
                "data",
                shape=(0, 64, 64, timesteps),
                maxshape=(None, 64, 64, timesteps),
                dtype=np.float32,
                chunks=(1, 64, 64, timesteps),
                compression=compression,
            )
            data.attrs["source"] = str(source.path)
            data.attrs["source_layout"] = layout
            data.attrs["start_index"] = start
            data.attrs["stop_index_exclusive"] = stop

            for offset in range(start, stop, chunk_size):
                local_stop = min(offset + chunk_size, stop)
                chunk = source.read(offset, local_stop, layout)
                if chunk.shape != (local_stop - offset, 64, 64, timesteps):
                    raise ValueError(
                        f"unexpected converted chunk shape {chunk.shape}; "
                        f"expected {(local_stop - offset, 64, 64, timesteps)}"
                    )
                if not np.isfinite(chunk).all():
                    raise ValueError(f"non-finite values found in source indices [{offset}, {local_stop})")
                old_size = data.shape[0]
                data.resize(old_size + chunk.shape[0], axis=0)
                data[old_size : old_size + chunk.shape[0]] = chunk
    except Exception:
        if created and output.exists():
            output.unlink()
        raise

    with h5py.File(output, "r") as handle:
        dataset = handle["data"]
        if dataset.shape != (expected_count, 64, 64, timesteps):
            raise ValueError(f"written shape mismatch: {dataset.shape}")
        return {
            "path": str(output),
            "shape": list(dataset.shape),
            "dtype": str(dataset.dtype),
            "size_bytes": output.stat().st_size,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--variable", default="u", help="Source variable/dataset name")
    parser.add_argument("--viscosity", choices=sorted(EXPECTED), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--compression", choices=("none", "lzf", "gzip"), default="none")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.n_train <= 0 or args.n_test <= 0 or args.chunk_size <= 0:
        parser.error("sample counts and --chunk-size must be positive")

    expected = EXPECTED[args.viscosity]
    source = Source(args.source, args.variable)
    try:
        layout, normalized_shape = detect_layout(source.shape, expected["timesteps"])
        if normalized_shape[0] != expected["n_total"]:
            raise ValueError(
                f"NS {args.viscosity} expects {expected['n_total']} trajectories, "
                f"but source {args.source} has {normalized_shape[0]}"
            )
        if args.n_train + args.n_test > normalized_shape[0]:
            raise ValueError("requested train/test split exceeds available trajectories")

        output_dir = args.output_dir.resolve()
        tag = "1e-5" if args.viscosity == "1e-5" else "1e-3"
        compression = None if args.compression == "none" else args.compression
        summary = {
            "viscosity": args.viscosity,
            "source": str(args.source.resolve()),
            "source_shape": list(source.shape),
            "source_dtype": str(source.dtype),
            "layout": layout,
            "splits": {
                "train": write_split(
                    source,
                    output_dir / f"ns2d_{tag}_train.hdf5",
                    layout,
                    0,
                    args.n_train,
                    expected["timesteps"],
                    args.chunk_size,
                    compression,
                    args.overwrite,
                ),
                "test": write_split(
                    source,
                    output_dir / f"ns2d_{tag}_test.hdf5",
                    layout,
                    normalized_shape[0] - args.n_test,
                    normalized_shape[0],
                    expected["timesteps"],
                    args.chunk_size,
                    compression,
                    args.overwrite,
                ),
            },
        }
    finally:
        source.close()

    manifest = output_dir / "prepare_manifest.json"
    manifest.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"manifest: {manifest}")


if __name__ == "__main__":
    main()
