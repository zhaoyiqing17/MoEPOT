#!/usr/bin/env python3
"""Check the Python/CUDA environment before launching cloud experiments."""

from __future__ import annotations

import argparse
import importlib
import platform
import sys
from pathlib import Path


REQUIRED = [
    "torch",
    "torchvision",
    "accelerate",
    "einops",
    "h5py",
    "huggingface_hub",
    "matplotlib",
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "tensorboard",
    "timm",
    "yaml",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    if Path.cwd().name != "MoEPOT":
        raise SystemExit(f"run from /code/MoEPOT, current={Path.cwd()}")

    print("python:", sys.version.replace("\n", " "))
    print("platform:", platform.platform())
    missing = []
    versions = {}
    for name in REQUIRED:
        try:
            module = importlib.import_module(name)
            versions[name] = getattr(module, "__version__", "unknown")
            print(f"{name}: {versions[name]}")
        except Exception as exc:
            missing.append(name)
            print(f"{name}: MISSING ({type(exc).__name__}: {exc})")

    import torch

    cuda_ok = torch.cuda.is_available()
    print("torch:", torch.__version__, "cuda_available:", cuda_ok)
    if cuda_ok:
        print("cuda:", torch.version.cuda)
        print("gpu_count:", torch.cuda.device_count())
        for index in range(torch.cuda.device_count()):
            print(f"gpu[{index}]:", torch.cuda.get_device_name(index))
    elif not args.allow_cpu:
        raise SystemExit("CUDA is required for the planned experiment")

    if missing:
        raise SystemExit(f"missing packages: {', '.join(missing)}")
    print("environment check: PASS")


if __name__ == "__main__":
    main()
