#!/usr/bin/env python3
"""Download a small, balanced, test-only Community Forensics evaluation sample.

The source dataset is CC-BY-NC-SA-4.0 and must never be used for training or
shipped in the extension. This helper records provenance in the model manifest
and selects equal real/fake rows from the official Commercial architecture
slice through the Hugging Face Dataset Viewer API.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import pathlib
import time
import urllib.parse
import urllib.request
import urllib.error

DATASET = "OwensLab/CommunityForensics-Eval"
REVISION = "7d4a74a88d2cac93b513c0853bf92c260eaceea0"
LICENSE = "CC-BY-NC-SA-4.0"
API = "https://datasets-server.huggingface.co/filter"


def fetch(where: str, offset: int, length: int) -> dict:
    query = urllib.parse.urlencode({
        "dataset": DATASET,
        "config": "default",
        "split": "CompEval",
        "where": where,
        "offset": offset,
        "length": length,
    })
    request = urllib.request.Request(f"{API}?{query}", headers={"User-Agent": "blur-evaluation/1"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.load(response)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--per-class", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--split-role", choices=("calibration", "test"), default="test")
    parser.add_argument("--ai-architecture", default="Commercial")
    parser.add_argument("--real-source", default="LAION")
    parser.add_argument("--selection", choices=("first", "even"), default="first")
    args = parser.parse_args()
    if args.per_class < 2 or not 1 <= args.batch_size <= 20:
        raise SystemExit("per-class must be >=2 and batch-size must be between 1 and 20")

    image_dir = args.output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output / ".progress.jsonl"
    records: list[dict] = ([json.loads(line) for line in progress_path.read_text().splitlines() if line.strip()]
                           if progress_path.exists() else [])
    if not records:
        # Recover already-downloaded fixed-source real rows from an interrupted run.
        prefix = f"{args.ai_architecture.lower()}-0-"
        for target in sorted(image_dir.glob(f"{prefix}*")):
            stable_id = target.stem
            records.append({
                "id": stable_id,
                "baseId": stable_id,
                "path": str(target.relative_to(args.output)),
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "label": 0,
                "source": f"commfor-{args.split_role}:{args.real_source}",
                "split": args.split_role,
                "license": LICENSE,
                "originUrl": f"https://huggingface.co/datasets/{DATASET}/tree/{REVISION}",
            })
        if records:
            progress_path.write_text("".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records))
    for label in (0, 1):
        downloaded = sum(record["label"] == label for record in records)
        offset = 0
        where = (f'"real_source" = \'{args.real_source}\' AND "label" = 0' if label == 0 else
                 f'"architecture" = \'{args.ai_architecture}\' AND "label" = 1')
        positions: list[int] | None = None
        if args.selection == "even":
            total = int(fetch(where, 0, 1)["num_rows_total"])
            if total < args.per_class:
                raise RuntimeError(f"only {total} rows match label {label}")
            positions = [round(index * (total - 1) / (args.per_class - 1)) for index in range(args.per_class)]
        while downloaded < args.per_class:
            if positions is None:
                requested = min(args.batch_size, args.per_class - downloaded)
                payload = fetch(where, offset, requested)
            else:
                requested = 1
                payload = fetch(where, positions[downloaded], requested)
            rows = payload.get("rows", [])
            if not rows:
                raise RuntimeError(f"dataset ended after {downloaded} label-{label} rows")
            for wrapped in rows:
                row = wrapped["row"]
                encoded = row["image_data"]
                if not isinstance(encoded, str):
                    raise RuntimeError("Dataset Viewer did not return base64 image bytes")
                suffix = "." + str(row["format"]).lower().replace("jpeg", "jpg")
                stable_id = f"{args.ai_architecture.lower()}-{label}-{wrapped['row_idx']}"
                target = image_dir / f"{stable_id}{suffix}"
                target.write_bytes(base64.b64decode(encoded))
                source = str(row["model_name"] if label else row["real_source"])
                records.append({
                    "id": stable_id,
                    "baseId": stable_id,
                    "path": str(target.relative_to(args.output)),
                    "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                    "label": label,
                    "source": f"commfor-{args.split_role}:{source}",
                    **({"generatorFamily": source} if label else {}),
                    "split": args.split_role,
                    "license": LICENSE,
                    "originUrl": f"https://huggingface.co/datasets/{DATASET}/tree/{REVISION}",
                })
                with progress_path.open("a") as progress:
                    progress.write(json.dumps(records[-1], separators=(",", ":")) + "\n")
                downloaded += 1
            offset += len(rows)

    manifest = args.output / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records))
    metadata = {
        "dataset": DATASET,
        "revision": REVISION,
        "license": LICENSE,
        "purpose": "test-only; never training or redistribution",
        "selection": args.selection,
        "ai_architecture": args.ai_architecture,
        "real_source": args.real_source,
        "split_role": args.split_role,
        "per_class": args.per_class,
    }
    (args.output / "sample.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"manifest": str(manifest), "count": len(records), **metadata}, indent=2))


if __name__ == "__main__":
    main()
