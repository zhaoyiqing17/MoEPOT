#!/usr/bin/env python3
"""Cloud preflight for data, Tiny checkpoint and one forward pass."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import h5py
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from MoEPOT.models.moepot import MoEPOTNet
from MoEPOT.utils.griddataset import MixedTemporalDataset
from MoEPOT.utils.utilities import load_model_from_checkpoint


EXPECTED = {
    "ns2d_1e-5_train.hdf5": (1000, 64, 64, 20),
    "ns2d_1e-5_test.hdf5": (200, 64, 64, 20),
    "ns2d_1e-3_train.hdf5": (1000, 64, 64, 50),
    "ns2d_1e-3_test.hdf5": (200, 64, 64, 50),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_tiny() -> MoEPOTNet:
    return MoEPOTNet(
        img_size=128,
        patch_size=8,
        in_channels=4,
        in_timesteps=10,
        out_timesteps=1,
        out_channels=4,
        normalize=0,
        embed_dim=512,
        modes=32,
        depth=4,
        n_blocks=4,
        mlp_ratio=1,
        out_layer_dim=32,
        act="gelu",
        n_cls=6,
        is_finetune=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/model/nestor_moepot_tiny/model_moepot_tiny.pth"),
    )
    parser.add_argument("--skip-forward", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CHECK FAILED: CUDA is not available")
    print("CHECK OK: CUDA available", torch.cuda.get_device_name(0))

    data_dir = ROOT / "data" / "large"
    for name, expected in EXPECTED.items():
        path = data_dir / name
        if not path.is_file():
            raise SystemExit(f"CHECK FAILED: missing {path}")
        with h5py.File(path, "r") as handle:
            if "data" not in handle:
                raise SystemExit(f"CHECK FAILED: {path} has no data dataset")
            shape = tuple(handle["data"].shape)
        if shape != expected:
            raise SystemExit(f"CHECK FAILED: {path} shape={shape}, expected={expected}")
        print(f"{name}: {shape}")

    if not args.checkpoint.is_file():
        raise SystemExit(f"CHECK FAILED: missing checkpoint {args.checkpoint}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"checkpoint_sha256: {sha256(args.checkpoint)}")

    device = torch.device("cuda:0")
    model = build_tiny().to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise SystemExit("CHECK FAILED: checkpoint must contain a model state_dict")
    load_model_from_checkpoint(model, checkpoint["model"])
    model.eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"model parameters: {parameter_count:,}")

    if not args.skip_forward:
        dataset = MixedTemporalDataset(
            "ns2d_fno_1e-5",
            res=128,
            t_in=10,
            t_ar=-1,
            n_channels=4,
            train=False,
        )
        sample, _, _, _ = dataset[0]
        sample = sample.unsqueeze(0).to(device)
        with torch.no_grad():
            prediction, _, _ = model(sample)
        if prediction.shape != (1, 128, 128, 1, 4):
            raise SystemExit(f"CHECK FAILED: unexpected model output {tuple(prediction.shape)}")
        if not torch.isfinite(prediction).all():
            raise SystemExit("CHECK FAILED: non-finite model output")
        print("forward output:", tuple(prediction.shape), "finite=True")
    print("CHECK OK")


if __name__ == "__main__":
    main()
