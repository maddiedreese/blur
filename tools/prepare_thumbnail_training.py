#!/usr/bin/env python3
"""Build a local, provenance-rich thumbnail-robustness training manifest.

AI positives come from a user-supplied DiffusionDB ZIP (CC0). Real negatives
come from the COCO 2017 train split through the Hugging Face Dataset Viewer;
the per-image COCO/Flickr license identifier and source URL are retained.
Existing Community Forensics rows are copied into the manifest only as the
calibration split and are never used for training.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import pathlib
import shutil
import urllib.parse
import urllib.request
import zipfile


VIEWER = "https://datasets-server.huggingface.co/rows"
COCO_DATASET = "AbdoTW/COCO_2017"
COCO_REVISION = "16c28f4b32df00e5fa71421ca788e379388746dd"
DIFFUSIONDB_ORIGIN = "https://huggingface.co/datasets/poloclub/diffusiondb"


def digest(path: pathlib.Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def get_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "blur-training-prep/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def coco_rows(count: int) -> list[dict]:
    rows: list[dict] = []
    # Spread three blocks across the training split instead of taking one
    # contiguous run. The selection remains deterministic and reproducible.
    for offset in (0, 40_000, 80_000, 110_000):
        query = urllib.parse.urlencode({
            "dataset": COCO_DATASET,
            "config": "default",
            "split": "train",
            "offset": offset,
            "length": min(100, count - len(rows)),
        })
        payload = get_json(f"{VIEWER}?{query}")
        rows.extend(item["row"] for item in payload["rows"])
        if len(rows) >= count:
            break
    if len(rows) < count:
        raise RuntimeError(f"only resolved {len(rows)} of {count} requested COCO rows")
    return rows[:count]


def download_coco(item: tuple[dict, pathlib.Path]) -> dict:
    row, image_dir = item
    target = image_dir / f"coco-{row['id']}.jpg"
    request = urllib.request.Request(row["image"]["src"], headers={"User-Agent": "blur-training-prep/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response, target.open("wb") as output:
        shutil.copyfileobj(response, output)
    return {
        "id": f"coco-{row['id']}",
        "baseId": f"coco-{row['id']}",
        "path": f"images/{target.name}",
        "sha256": digest(target),
        "label": 0,
        "source": "COCO2017-train",
        "split": "train",
        "license": f"COCO-Flickr-license-id-{row['license']}",
        "originUrl": row["flickr_url"] or row["coco_url"],
        "contentGroup": "natural-photograph",
    }


def extract_diffusiondb(archive: pathlib.Path, image_dir: pathlib.Path, count: int) -> list[dict]:
    result: list[dict] = []
    with zipfile.ZipFile(archive) as source:
        members = sorted(name for name in source.namelist() if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")))
        if len(members) < count:
            raise RuntimeError(f"archive has only {len(members)} images; {count} requested")
        # Even spacing avoids depending only on archive ordering while keeping
        # the exact selection deterministic.
        indexes = [index * len(members) // count for index in range(count)]
        for index in indexes:
            member = members[index]
            name = pathlib.PurePosixPath(member).name
            target = image_dir / f"diffusiondb-{name}"
            with source.open(member) as input_file, target.open("wb") as output:
                shutil.copyfileobj(input_file, output)
            identifier = f"diffusiondb-{pathlib.Path(name).stem}"
            result.append({
                "id": identifier,
                "baseId": identifier,
                "path": f"images/{target.name}",
                "sha256": digest(target),
                "label": 1,
                "source": "DiffusionDB-2M",
                "generatorFamily": "StableDiffusion-1.x",
                "split": "train",
                "license": "CC0-1.0",
                "originUrl": DIFFUSIONDB_ORIGIN,
                "contentGroup": "user-prompted-generation",
            })
    return result


def calibration_rows(manifest: pathlib.Path, output: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        source = (manifest.parent / row["path"]).resolve()
        row["path"] = os.path.relpath(source, output.resolve())
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("diffusiondb_zip", type=pathlib.Path)
    parser.add_argument("calibration_manifest", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--per-class", type=int, default=256)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    image_dir = args.output / "images"
    image_dir.mkdir(exist_ok=True)
    ai = extract_diffusiondb(args.diffusiondb_zip, image_dir, args.per_class)
    real_source = coco_rows(args.per_class)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        real = list(executor.map(download_coco, ((row, image_dir) for row in real_source)))
    calibration = calibration_rows(args.calibration_manifest, args.output)
    rows = ai + real + calibration
    target = args.output / "manifest.jsonl"
    target.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows))
    provenance = {
        "manifest": str(target),
        "counts": {"train_ai": len(ai), "train_real": len(real), "calibration": len(calibration)},
        "diffusiondb": {"license": "CC0-1.0", "origin": DIFFUSIONDB_ORIGIN, "archive_sha256": digest(args.diffusiondb_zip)},
        "coco": {"dataset": COCO_DATASET, "revision": COCO_REVISION, "selection_offsets": [0, 40_000, 80_000, 110_000]},
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
