#!/usr/bin/env python3
"""Combine frozen bases with one browser-eligible thumbnail per base."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from PIL import Image


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def verified(row: dict, manifest: Path) -> tuple[dict, int, int]:
    image = (manifest.parent / row["path"]).resolve()
    data = image.read_bytes()
    if hashlib.sha256(data).hexdigest() != row["sha256"]:
        raise ValueError(f"{row['id']}: byte hash mismatch")
    with Image.open(image) as opened:
        width, height = opened.size
    rebased = {**row, "path": os.path.relpath(image, manifest.parent)}
    return rebased, width, height


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("full", type=Path)
    parser.add_argument("stress", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--transform", default="thumb-256-jpeg75")
    parser.add_argument("--min-edge", type=int, default=96)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    result = []
    for row in rows(args.full):
        item, _, _ = verified(row, args.full)
        item["path"] = os.path.relpath((args.full.parent / row["path"]).resolve(), args.output.parent)
        result.append(item)
    eligible = {0: 0, 1: 0}
    for row in rows(args.stress):
        if row.get("transform") != args.transform:
            continue
        item, width, height = verified(row, args.stress)
        if min(width, height) < args.min_edge:
            continue
        item["path"] = os.path.relpath((args.stress.parent / row["path"]).resolve(), args.output.parent)
        result.append(item)
        eligible[int(item["label"])] += 1
    if eligible[0] < 100 or eligible[1] < 100:
        raise ValueError(f"thumbnail breadth floor not met: {eligible}")
    ids = [row["id"] for row in result]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate evaluation ID")
    with tempfile.NamedTemporaryFile("w", dir=args.output.parent, delete=False) as handle:
        temporary = Path(handle.name)
        for row in result:
            handle.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, args.output)
    print(json.dumps({"rows": len(result), "full": len(rows(args.full)), "thumbnail": eligible}, indent=2))


if __name__ == "__main__":
    main()
