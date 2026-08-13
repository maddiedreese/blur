#!/usr/bin/env python3
"""Download a pinned, licensed FLUX.2 klein subset and convert JXL to JPEG."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import pathlib
import subprocess
import urllib.request


DATASET = "stablellama/FLUX.2-klein-base-9B_samples"
REVISION = "c07dd3cf504b2c4ca67251e21febf3e8b0a46c36"
LICENSE = "CC-BY-4.0"


def download(index: int, output: pathlib.Path) -> dict:
    name = f"animal_{index:05d}_0"
    source_path = f"data/animal/dataset_0/{name}.jxl"
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{source_path}?download=true"
    jxl = output / f"{name}.jxl"
    jpeg = output / f"{name}.jpg"
    if not jpeg.is_file():
        request = urllib.request.Request(url, headers={"User-Agent": "blur-training-prep/1.0"})
        with urllib.request.urlopen(request, timeout=180) as response:
            jxl.write_bytes(response.read())
        subprocess.run(["sips", "-s", "format", "jpeg", str(jxl), "--out", str(jpeg)], check=True, stdout=subprocess.DEVNULL)
        jxl.unlink()
    return {
        "index": index,
        "path": str(jpeg),
        "sha256": hashlib.sha256(jpeg.read_bytes()).hexdigest(),
        "sourcePath": source_path,
        "originUrl": url,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=pathlib.Path)
    # dataset_0 contains exactly 80 numbered animal samples.
    parser.add_argument("--count", type=int, default=80)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.count <= 80:
        parser.error("--count must be between 1 and 80 for dataset_0")
    args.output.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(lambda index: download(index, args.output), range(1, args.count + 1)))
    ledger = {
        "dataset": DATASET,
        "revision": REVISION,
        "license": LICENSE,
        "generatorFamily": "FLUX.2-klein-base-9B",
        "conversion": "macOS ImageIO JXL decode to baseline JPEG",
        "rows": rows,
    }
    target = args.output / "provenance.json"
    target.write_text(json.dumps(ledger, indent=2) + "\n")
    print(json.dumps({"provenance": str(target), "count": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
