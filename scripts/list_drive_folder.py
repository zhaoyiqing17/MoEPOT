#!/usr/bin/env python3
"""List a public Google Drive folder and resolve one exact filename to a file ID."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gdown import download_folder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Public Google Drive folder URL")
    parser.add_argument("--match", required=True, help="Exact target filename")
    parser.add_argument("--json", type=Path, help="Optional JSON inventory output")
    parser.add_argument("--scratch", type=Path, default=Path("/private/tmp/gdown-index"))
    parser.add_argument("--id-only", action="store_true", help="Print only the matching file ID")
    args = parser.parse_args()

    files = download_folder(
        url=args.url,
        output=str(args.scratch),
        quiet=True,
        remaining_ok=True,
        skip_download=True,
    )
    if files is None:
        raise SystemExit(
            "could not read the Drive folder; confirm the folder is publicly accessible"
        )

    inventory = [
        {
            "id": item.id,
            "path": item.path,
            "filename": Path(item.path).name,
            "view_url": f"https://drive.google.com/file/d/{item.id}/view?usp=sharing",
            "download_url": f"https://drive.google.com/uc?export=download&id={item.id}",
        }
        for item in files
    ]
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(inventory, indent=2, ensure_ascii=False), encoding="utf-8")

    matches = [item for item in inventory if item["filename"] == args.match]
    if len(matches) != 1:
        names = "\n".join(item["path"] for item in inventory)
        raise SystemExit(
            f"expected exactly one {args.match!r}, found {len(matches)}.\n"
            f"Available files:\n{names}"
        )

    match = matches[0]
    if args.id_only:
        print(match["id"])
    else:
        print(json.dumps(match, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
