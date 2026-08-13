#!/usr/bin/env python3
"""Acquire license-pinned real-image candidates without scoring them.

PxHere and museum art are read from pinned Hugging Face WebDataset shards.
CORD is paginated through Dataset Viewer after the Hub revision is verified.
The complete output directory is staged beside its destination and renamed only
after every source reaches quota plus margin and every image validates.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import shutil
import tarfile
import tempfile
from typing import BinaryIO, Iterator
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
USER_AGENT = "blur-frozen-test-acquisition/1"


class HttpClient:
    def json(self, url: str) -> dict:
        with urlopen(Request(url, headers={"User-Agent": USER_AGENT}), timeout=60) as response:
            return json.load(response)

    def open(self, url: str) -> BinaryIO:
        return urlopen(Request(url, headers={"User-Agent": USER_AGENT}), timeout=120)

    def bytes(self, url: str) -> bytes:
        with self.open(url) as response:
            return response.read()


def validate_image(data: bytes) -> tuple[str, int, int]:
    with Image.open(io.BytesIO(data)) as image:
        image.verify()
    with Image.open(io.BytesIO(data)) as image:
        width, height = image.size
        image_format = (image.format or "").lower()
    if width < 128 or height < 128:
        raise ValueError(f"image is too small: {width}x{height}")
    if image_format not in {"jpeg", "png", "webp", "tiff"}:
        raise ValueError(f"unsupported image format: {image_format}")
    extension = {"jpeg": ".jpg", "png": ".png", "webp": ".webp", "tiff": ".tiff"}[image_format]
    return extension, width, height


def iter_webdataset(stream: BinaryIO) -> Iterator[tuple[str, bytes, dict]]:
    """Yield grouped image/JSON samples from a WebDataset tar stream."""
    pending: dict[str, dict[str, bytes]] = {}
    with tarfile.open(fileobj=stream, mode="r|*") as archive:
        for member in archive:
            if not member.isfile():
                continue
            path = PurePosixPath(member.name)
            suffix = path.suffix.lower()
            if suffix not in IMAGE_EXTENSIONS | {".json"}:
                continue
            key = str(path.with_suffix(""))
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            pending.setdefault(key, {})[suffix] = extracted.read()
            parts = pending[key]
            image_suffix = next((item for item in IMAGE_EXTENSIONS if item in parts), None)
            if image_suffix and ".json" in parts:
                metadata = json.loads(parts[".json"])
                yield key, parts[image_suffix], metadata
                del pending[key]


def verify_revision(client: HttpClient, dataset: str, revision: str, expected_license: str) -> dict:
    metadata = client.json(f"https://huggingface.co/api/datasets/{dataset}")
    if metadata.get("sha") != revision:
        raise ValueError(f"{dataset}: expected revision {revision}, got {metadata.get('sha')}")
    if metadata.get("gated") or metadata.get("private") or metadata.get("disabled"):
        raise ValueError(f"{dataset}: repository is not publicly usable")
    declared_license = str((metadata.get("cardData") or {}).get("license", "")).lower()
    if declared_license != expected_license.lower():
        raise ValueError(f"{dataset}: expected license {expected_license}, got {declared_license or '<missing>'}")
    return metadata


def atomic_image(staging: Path, source_slug: str, sample_id: str, data: bytes) -> tuple[str, str, int, int]:
    extension, width, height = validate_image(data)
    digest = hashlib.sha256(data).hexdigest()
    safe_id = hashlib.sha256(sample_id.encode()).hexdigest()[:24]
    relative = Path("images") / source_slug / f"{safe_id}{extension}"
    destination = staging / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.write_bytes(data)
    os.replace(temporary, destination)
    return relative.as_posix(), digest, width, height


def source_target(spec: dict, margin: float) -> int:
    return spec["count"] + max(10, math.ceil(spec["count"] * margin))


def webdataset_candidates(client: HttpClient, spec: dict, staging: Path, margin: float, seed: str) -> list[dict]:
    repository = verify_revision(client, spec["source"], spec["datasetRevision"], spec["license"])
    prefix = "pxhere-" if spec["source"] == "nyuuzyou/pxhere" else "ArtMuseumsPD_"
    shards = sorted(item["rfilename"] for item in repository.get("siblings", []) if item["rfilename"].startswith(prefix) and item["rfilename"].endswith(".tar"))
    random.Random(f"{seed}:{spec['source']}").shuffle(shards)
    goal = source_target(spec, margin)
    rows = []
    seen = set()
    seen_digests = set()
    for shard in shards:
        locator = f"https://huggingface.co/datasets/{spec['source']}/resolve/{spec['datasetRevision']}/{shard}"
        with client.open(locator) as stream:
            for key, data, metadata in iter_webdataset(stream):
                original_id = str(metadata.get("image_id") or metadata.get("id") or key)
                origin = metadata.get("download_url") or metadata.get("url") or metadata.get("source_url") or f"hf://datasets/{spec['source']}@{spec['datasetRevision']}/{shard}#{key}"
                identity = f"{spec['source']}:{original_id}"
                if identity in seen:
                    continue
                try:
                    path, digest, width, height = atomic_image(staging, spec["source"].replace("/", "--"), identity, data)
                except (OSError, ValueError):
                    continue
                if digest in seen_digests:
                    (staging / path).unlink()
                    continue
                seen.add(identity)
                seen_digests.add(digest)
                is_photo = spec["source"] == "nyuuzyou/pxhere"
                rows.append({
                    "id": identity, "baseId": identity, "path": path, "sha256": digest,
                    "label": 0, "source": spec["source"], "split": "test",
                    "license": spec["license"], "originUrl": origin,
                    "datasetRevision": spec["datasetRevision"],
                    "contentGroup": "camera-photograph" if is_photo else "human-art",
                    "attribution": "PxHere image; CC0-1.0" if is_photo else "Mitsua Art Museums PD dataset; CC BY 4.0; underlying work declared CC0/public domain",
                    "upstreamLocator": f"{shard}#{key}", "width": width, "height": height,
                })
                if len(rows) >= goal:
                    return rows
    raise ValueError(f"{spec['source']}: only {len(rows)} valid unique images; requires {goal}")


def cord_candidates(client: HttpClient, spec: dict, staging: Path, margin: float) -> list[dict]:
    verify_revision(client, spec["source"], spec["datasetRevision"], spec["license"])
    goal = source_target(spec, margin)
    rows = []
    seen_digests = set()
    for upstream_split in ("test", "validation"):
        offset = 0
        while len(rows) < goal:
            query = urlencode({"dataset": spec["source"], "config": "default", "split": upstream_split, "offset": offset, "length": 100})
            payload = client.json(f"https://datasets-server.huggingface.co/rows?{query}")
            page = payload.get("rows", [])
            if not page:
                break
            for item in page:
                index = int(item["row_idx"])
                image = item["row"]["image"]
                ground_truth = item["row"].get("ground_truth", {})
                if isinstance(ground_truth, str):
                    try:
                        ground_truth = json.loads(ground_truth)
                    except json.JSONDecodeError:
                        ground_truth = {}
                source_record_id = str((ground_truth.get("meta") or {}).get("image_id") or f"{upstream_split}:{index}")
                identity = f"cord-v2:{upstream_split}:{index}"
                try:
                    data = client.bytes(image["src"])
                    path, digest, width, height = atomic_image(staging, "cord-v2", identity, data)
                except (OSError, ValueError):
                    continue
                if digest in seen_digests:
                    (staging / path).unlink()
                    continue
                seen_digests.add(digest)
                rows.append({
                    "id": identity, "baseId": identity, "path": path, "sha256": digest,
                    "label": 0, "source": spec["source"], "split": "test",
                    "license": spec["license"],
                    "originUrl": f"hf://datasets/{spec['source']}@{spec['datasetRevision']}/default/{upstream_split}/{index}/image",
                    "datasetRevision": spec["datasetRevision"], "contentGroup": "text-heavy-receipt",
                    "attribution": "CORD v2 by NAVER CLOVA; CC BY 4.0",
                    "upstreamLocator": f"default/{upstream_split}/{index}/image",
                    "sourceRecordId": source_record_id, "width": width, "height": height,
                })
                if len(rows) >= goal:
                    # Viewer has no revision parameter. Rechecking the Hub head
                    # closes the ordinary update race and fails rather than
                    # claiming that rows from a changed head came from the pin.
                    verify_revision(client, spec["source"], spec["datasetRevision"], spec["license"])
                    return rows
            offset += len(page)
            if offset >= int(payload.get("num_rows_total", offset)):
                break
    raise ValueError(f"CORD: only {len(rows)} valid images; requires {goal}")


def acquire(spec_path: Path, output: Path, margin: float, seed: str, client: HttpClient) -> dict:
    if output.exists():
        raise FileExistsError(output)
    config = json.loads(spec_path.read_text())
    real_specs = [item for item in config["sources"] if item["label"] == 0]
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        rows = []
        for spec in real_specs:
            if spec["source"] == "naver-clova-ix/cord-v2":
                rows.extend(cord_candidates(client, spec, staging, margin))
            elif spec["source"] in {"nyuuzyou/pxhere", "Mitsua/art-museums-pd-440k"}:
                rows.extend(webdataset_candidates(client, spec, staging, margin, seed))
            else:
                raise ValueError(f"unsupported real acquisition source: {spec['source']}")
        duplicate_hashes = [digest for digest in {row["sha256"] for row in rows} if sum(item["sha256"] == digest for item in rows) > 1]
        if duplicate_hashes:
            raise ValueError(f"cross-source duplicate image bytes: {len(duplicate_hashes)}")
        manifest = staging / "candidates.jsonl"
        manifest.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows))
        provenance = {
            "schemaVersion": 1, "selectionSeed": seed, "margin": margin,
            "sourceCounts": {spec["source"]: source_target(spec, margin) for spec in real_specs},
            "candidateManifestSha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        }
        (staging / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        os.replace(staging, output)
        return provenance
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=ROOT / "tools/frozen_test_sources.json")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--seed", default="blur-real-acquisition-v1")
    args = parser.parse_args()
    if not 0 < args.margin <= 1:
        raise SystemExit("margin must be in (0, 1]")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps(acquire(args.spec, args.output, args.margin, args.seed, HttpClient()), indent=2))


if __name__ == "__main__":
    main()
