#!/usr/bin/env python3
"""Download, hash and state-dict-check MoE-POT Tiny from Hugging Face."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download

REPO_ROOT = Path.cwd()
if REPO_ROOT.name != "MoEPOT":
    raise SystemExit(f"run from a checkout directory named MoEPOT, current={REPO_ROOT}")
sys.path.insert(0, str(REPO_ROOT.parent))

from MoEPOT.models.moepot import MoEPOTNet
from MoEPOT.utils.utilities import load_model_from_checkpoint


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_checkpoint(files: list[str], explicit: str | None) -> str:
    if explicit:
        if explicit not in files:
            raise ValueError(f"requested file {explicit!r} is not present in the Hub repository")
        return explicit

    exact = "model_moepot_tiny.pth"
    if exact in files:
        return exact
    candidates = [
        name
        for name in files
        if name.lower().endswith((".pth", ".pt", ".ckpt"))
        and "moepot" in name.lower()
        and "tiny" in name.lower()
    ]
    if len(candidates) != 1:
        raise ValueError(
            "could not select a unique Tiny checkpoint. "
            f"Candidates={candidates}. Available weight files="
            f"{[name for name in files if name.lower().endswith(('.pth', '.pt', '.ckpt'))]}"
        )
    return candidates[0]


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
    parser.add_argument("--repo-id", default="xhy2878/MoEPOT")
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/model/nestor_moepot_tiny/model_moepot_tiny.pth"),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("/data/nestor_moepot_ns/cache/huggingface"))
    parser.add_argument("--filename", help="Explicit Hub filename, e.g. model_moepot_tiny.pth")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-model-check", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        print(f"checkpoint already exists: {args.output}")
    else:
        api = HfApi()
        files = api.list_repo_files(repo_id=args.repo_id, revision=args.revision)
        filename = select_checkpoint(files, args.filename)
        cached = Path(
            hf_hub_download(
                repo_id=args.repo_id,
                filename=filename,
                revision=args.revision,
                cache_dir=args.cache_dir,
            )
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached, args.output)
        print(f"downloaded {args.repo_id}/{filename} -> {args.output}")

    checkpoint_hash = sha256(args.output)
    checkpoint = torch.load(args.output, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("checkpoint must be a dict containing a 'model' state_dict")

    model_check = "skipped"
    if not args.skip_model_check:
        model = build_tiny()
        load_model_from_checkpoint(model, checkpoint["model"])
        model_check = "all Tiny model keys loaded successfully"
        del model

    manifest = {
        "repo_id": args.repo_id,
        "revision": args.revision,
        "path": str(args.output.resolve()),
        "size_bytes": args.output.stat().st_size,
        "sha256": checkpoint_hash,
        "model_check": model_check,
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".sha256.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
