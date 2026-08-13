#!/usr/bin/env python3
"""Create deterministic browser-thumbnail stress derivatives from a manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import tempfile

from PIL import Image


def digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = args.manifest.resolve().parent
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = pathlib.Path(tempfile.mkdtemp(prefix=f".{args.output.name}-", dir=args.output.parent))
    image_dir = staging / "images"
    image_dir.mkdir()
    result: list[dict] = []
    variants = ((256, 75, "thumb-256-jpeg75"), (192, 50, "thumb-192-jpeg50"))
    try:
        for row in rows:
            with Image.open(root / row["path"]) as opened:
                image = opened.convert("RGB")
                for max_edge, quality, transform in variants:
                    variant = image.copy()
                    variant.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
                    identifier = f"{row['id']}--{transform}"
                    filename = hashlib.sha256(identifier.encode()).hexdigest() + ".jpg"
                    target = image_dir / filename
                    variant.save(target, format="JPEG", quality=quality, optimize=False, progressive=False)
                    result.append({
                        **row,
                        "id": identifier,
                        "path": str(pathlib.Path("images") / filename),
                        "sha256": digest(target),
                        "transform": transform,
                    })
        target_manifest = staging / "manifest.jsonl"
        target_manifest.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in result))
        os.replace(staging, args.output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({"manifest": str(args.output / "manifest.jsonl"), "count": len(result), "transforms": [item[2] for item in variants]}, indent=2))


if __name__ == "__main__":
    main()
