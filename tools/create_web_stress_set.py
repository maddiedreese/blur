#!/usr/bin/env python3
"""Create deterministic browser-thumbnail stress derivatives from a manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

from PIL import Image


def digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    args = parser.parse_args()
    root = args.manifest.resolve().parent
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    image_dir = args.output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    result: list[dict] = []
    variants = ((256, 75, "thumb-256-jpeg75"), (192, 50, "thumb-192-jpeg50"))
    for row in rows:
        with Image.open(root / row["path"]) as opened:
            image = opened.convert("RGB")
            for max_edge, quality, transform in variants:
                variant = image.copy()
                variant.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
                identifier = f"{row['id']}--{transform}"
                target = image_dir / f"{identifier}.jpg"
                variant.save(target, format="JPEG", quality=quality, optimize=False, progressive=False)
                result.append({
                    **row,
                    "id": identifier,
                    "path": str(target.relative_to(args.output)),
                    "sha256": digest(target),
                    "transform": transform,
                })
    target_manifest = args.output / "manifest.jsonl"
    target_manifest.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in result))
    print(json.dumps({"manifest": str(target_manifest), "count": len(result), "transforms": [item[2] for item in variants]}, indent=2))


if __name__ == "__main__":
    main()
