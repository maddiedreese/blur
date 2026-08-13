#!/usr/bin/env python3
"""Acquire the pinned X-AIGD AI-only frozen test contract.

The command is intentionally fail-closed and writes through a staging directory.
It never scores images.  Repeated prompts are retained only as a shared
``promptGroup`` inside this one frozen test split.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

from PIL import Image


DATASET = "Coxy7/X-AIGD"
REVISION = "92180f32030507ab54a40d6f1b88f39d6cec8178"
LICENSE = "CC-BY-4.0"
CONFIG = "default"
SPLIT = "labeled_test"
TOTAL_ROWS = 2419
QUOTAS = {"HYDiT_1.2-raw": 181, "Infinity-raw": 198, "Lumina_Next-raw": 173}
FAMILIES = {"HYDiT_1.2-raw": "HYDiT-1.2", "Infinity-raw": "Infinity", "Lumina_Next-raw": "Lumina-Next"}
PROTECTED_TOKENS = {
    "midjourney", "laion", "raise", "dfgan", "coco", "diffusiondb", "flux2-klein",
    "flux.2-klein", "gpt4o", "gpt-4o", "4o-26-3-25", "ideogram",
}
USER_AGENT = "blur-frozen-test-acquisition/1.0"


def normalize_prompt(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value).strip().casefold())


def prompt_group(value: str) -> str:
    return hashlib.sha256(normalize_prompt(value).encode("utf-8")).hexdigest()


def get_bytes(url: str, attempts: int = 6) -> bytes:
    for attempt in range(attempts):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return response.read()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            if attempt + 1 == attempts:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def validate_repository_metadata(metadata: dict) -> None:
    if metadata.get("sha") != REVISION:
        raise ValueError(f"dataset revision moved: {metadata.get('sha')!r}")
    card_license = str((metadata.get("cardData") or {}).get("license", "")).lower()
    if card_license != LICENSE.lower():
        raise ValueError(f"dataset license changed: {card_license!r}")


def select_rows(pages: list[dict], quotas: dict[str, int] = QUOTAS) -> list[dict]:
    selected: list[dict] = []
    seen_rows: set[int] = set()
    seen_model_uid: set[tuple[str, str]] = set()
    counts = {model: 0 for model in quotas}
    for page in pages:
        if page.get("partial"):
            raise ValueError("Dataset Viewer returned a partial page")
        if page.get("num_rows_total") != TOTAL_ROWS:
            raise ValueError("Dataset Viewer row total changed")
        for item in page.get("rows", []):
            row = item.get("row") or {}
            model = str(row.get("generator", ""))
            if model not in quotas:
                continue
            searchable = f"{DATASET} {model}".lower()
            if any(token in searchable for token in PROTECTED_TOKENS):
                raise ValueError(f"protected generator/source token in {model!r}")
            uid = str(row.get("uid", ""))
            row_index = int(item["row_idx"])
            prompt = row.get("original_prompt")
            image = row.get("image")
            if not uid or not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"row {item.get('row_idx')} lacks UID/prompt")
            if not re.fullmatch(r"[A-Za-z0-9_-]+", uid):
                raise ValueError(f"row {item.get('row_idx')} has unsafe UID {uid!r}")
            if row_index in seen_rows:
                raise ValueError(f"duplicate row index {row_index}")
            if (model, uid) in seen_model_uid:
                raise ValueError(f"duplicate generator/UID pair {(model, uid)!r}")
            if not isinstance(image, dict) or not image.get("src"):
                raise ValueError(f"row {item.get('row_idx')} lacks direct image")
            if f"/--/{REVISION}/--/" not in image["src"]:
                raise ValueError(f"row {item.get('row_idx')} image is not revision-pinned")
            seen_rows.add(row_index)
            seen_model_uid.add((model, uid))
            counts[model] += 1
            selected.append({"rowIndex": row_index, **row})
    if counts != quotas:
        raise ValueError(f"generator quota mismatch: observed {counts}, expected {quotas}")
    return selected


def validate_image_bytes(data: bytes, row: dict) -> tuple[str, int, int, str]:
    digest = hashlib.sha256(data).hexdigest()
    with Image.open(io.BytesIO(data)) as image:
        image.verify()
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        width, height = image.size
        fmt = str(image.format or "").upper()
    expected_width, expected_height = int(row["width"]), int(row["height"])
    if (width, height) != (expected_width, expected_height):
        raise ValueError(f"{row['uid']}: decoded dimensions {(width, height)} != {(expected_width, expected_height)}")
    declared = str(row.get("image_format", "")).upper()
    if declared == "JPG":
        declared = "JPEG"
    if declared and fmt != declared:
        raise ValueError(f"{row['uid']}: decoded format {fmt!r} != declared {declared!r}")
    return digest, width, height, fmt


def load_forbidden(paths: list[Path]) -> dict[str, set[str]]:
    values = {"sha256": set(), "promptGroup": set()}
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                for field in values:
                    if row.get(field):
                        values[field].add(str(row[field]))
    return values


def download_one(row: dict, image_dir: Path) -> dict:
    data = get_bytes(row["image"]["src"])
    digest, width, height, fmt = validate_image_bytes(data, row)
    extension = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}.get(fmt)
    if not extension:
        raise ValueError(f"{row['uid']}: unsupported decoded format {fmt!r}")
    base_id = f"xaigd-row-{row['rowIndex']:04d}"
    filename = f"{base_id}-{row['uid']}.{extension}"
    target = image_dir / filename
    temporary = target.with_suffix(target.suffix + ".part")
    temporary.write_bytes(data)
    os.replace(temporary, target)
    model = str(row["generator"])
    locator = f"hf://datasets/{DATASET}@{REVISION}/{CONFIG}/{SPLIT}#row={row['rowIndex']};column=image"
    manifest = {
        "id": base_id, "baseId": base_id, "path": f"images/{filename}",
        "sha256": digest, "label": 1, "source": DATASET, "split": "test", "license": LICENSE,
        "originUrl": f"https://huggingface.co/datasets/{DATASET}/tree/{REVISION}",
        "originLocator": locator, "datasetRevision": REVISION, "datasetConfig": CONFIG,
        "datasetSplit": SPLIT, "rowIndex": row["rowIndex"], "uid": row["uid"],
        "generatorFamily": FAMILIES[model], "generatorModel": model,
        "contentGroup": "prompted-generation", "promptGroup": prompt_group(row["original_prompt"]),
        "width": width, "height": height, "imageFormat": fmt,
        "attribution": f"X-AIGD by Coxy7, CC BY 4.0; source row {locator}",
    }
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--forbid", action="append", type=Path, default=[])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.workers < 1 or args.workers > 16:
        parser.error("--workers must be in [1, 16]")
    metadata_raw = get_bytes(f"https://huggingface.co/api/datasets/{DATASET}")
    metadata = json.loads(metadata_raw)
    validate_repository_metadata(metadata)
    pages: list[dict] = []
    raw_pages: list[bytes] = []
    for offset in range(0, TOTAL_ROWS, 100):
        query = urllib.parse.urlencode({
            "dataset": DATASET, "config": CONFIG, "split": SPLIT,
            "offset": offset, "length": min(100, TOTAL_ROWS - offset),
        })
        raw = get_bytes(f"https://datasets-server.huggingface.co/rows?{query}")
        raw_pages.append(raw)
        pages.append(json.loads(raw))
        time.sleep(0.2)
    selected = select_rows(pages)
    default_forbidden = [
        Path("data/recent-training/manifest.jsonl"), Path("data/thumbnail-training/manifest.jsonl"),
        Path("data/commfor-calibration-64/manifest.jsonl"), Path("data/commfor-calibration-stress/manifest.jsonl"),
        Path("data/commfor-commercial-64/manifest.jsonl"), Path("data/commfor-commercial-stress/manifest.jsonl"),
    ]
    forbidden = load_forbidden(args.forbid or default_forbidden)
    selected_prompt_groups = {prompt_group(row["original_prompt"]) for row in selected}
    if forbidden["promptGroup"].intersection(selected_prompt_groups):
        raise ValueError("X-AIGD prompt group overlaps another split")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output.name}-", dir=args.output.parent))
    try:
        images = staging / "images"; images.mkdir()
        pages_dir = staging / "viewer-pages"; pages_dir.mkdir()
        (staging / "repository-metadata.json").write_bytes(metadata_raw)
        page_ledger = []
        for index, raw in enumerate(raw_pages):
            name = f"rows-{index * 100:04d}.json"
            (pages_dir / name).write_bytes(raw)
            page_ledger.append({"file": f"viewer-pages/{name}", "sha256": hashlib.sha256(raw).hexdigest()})
        manifests = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(download_one, row, images) for row in selected]
            for future in as_completed(futures):
                manifests.append(future.result())
        hashes = [row["sha256"] for row in manifests]
        uids = [row["uid"] for row in manifests]
        row_indices = [row["rowIndex"] for row in manifests]
        model_uids = [(row["generatorModel"], row["uid"]) for row in manifests]
        if len(set(hashes)) != len(hashes):
            raise ValueError("downloaded images contain duplicate byte SHA-256")
        if forbidden["sha256"].intersection(hashes):
            raise ValueError("downloaded image overlaps a training/calibration/protected byte hash")
        if len(set(row_indices)) != len(row_indices):
            raise ValueError("downloaded images contain duplicate row index")
        if len(set(model_uids)) != len(model_uids):
            raise ValueError("downloaded images contain duplicate generator/UID pair")
        manifests.sort(key=lambda row: (row["generatorModel"], row["rowIndex"]))
        (staging / "candidates.jsonl").write_text("".join(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n" for row in manifests))
        ledger = {
            "dataset": DATASET, "revision": REVISION, "license": LICENSE, "config": CONFIG, "split": SPLIT,
            "rows": len(manifests), "generatorCounts": QUOTAS,
            "uniqueBases": len(set(row_indices)), "uniqueByteHashes": len(set(hashes)),
            "uniqueUids": len(set(uids)), "uniqueGeneratorUidPairs": len(set(model_uids)),
            "uniquePromptGroups": len({row["promptGroup"] for row in manifests}),
            "expectedUniquePromptGroups": 375, "viewerPages": page_ledger,
            "repositoryMetadata": {
                "file": "repository-metadata.json", "sha256": hashlib.sha256(metadata_raw).hexdigest(),
            },
        }
        if ledger["rows"] != 552 or ledger["uniqueBases"] != 552:
            raise ValueError("frozen AI breadth floor not met")
        if ledger["uniquePromptGroups"] != 375:
            raise ValueError(f"prompt-group cardinality changed: {ledger['uniquePromptGroups']}")
        (staging / "ledger.json").write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
        os.replace(staging, args.output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({"output": str(args.output), "rows": 552, "uniquePromptGroups": 375}, indent=2))


if __name__ == "__main__":
    main()
