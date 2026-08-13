#!/usr/bin/env python3
"""Extract only pinned target-generator columns from local HF Parquet shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib

import pyarrow.parquet as pq


SOURCES = (
    ("gpt4o", "OpenAI-4o_t2i_human_preference", "4o-26-3-25", "image2", "model2", "9fafb39b4bb3bac6e2fbabd13503fa1199fde400"),
    ("ideogram", "Ideogram-V2_t2i_human_preference", "ideogram", "image2", "model2", "9d9bb0aa365e9fbc77e865731ec96655a10e0990"),
)


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_file(path: pathlib.Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load_jsonl(path: pathlib.Path, output: pathlib.Path, split: str | None = None) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if split is not None and row["split"] != split:
            continue
        source = (path.parent / row["path"]).resolve()
        row["path"] = os.path.relpath(source, output.resolve())
        rows.append(row)
    return rows


def extract(
    parquet: pathlib.Path,
    *,
    slug: str,
    repository: str,
    model: str,
    image_column: str,
    model_column: str,
    revision: str,
    image_dir: pathlib.Path,
    output: pathlib.Path,
    limit: int,
) -> list[dict]:
    table = pq.read_table(parquet, columns=["prompt", image_column, model_column])
    result: list[dict] = []
    seen_hashes: set[str] = set()
    for index in range(table.num_rows):
        found_model = table[model_column][index].as_py()
        if found_model != model:
            continue
        prompt = " ".join(str(table["prompt"][index].as_py()).lower().split())
        image = table[image_column][index].as_py()
        value = image["bytes"]
        checksum = digest_bytes(value)
        if checksum in seen_hashes:
            continue
        seen_hashes.add(checksum)
        extension = pathlib.Path(image.get("path") or "image.jpg").suffix.lower()
        if extension not in {".jpg", ".jpeg", ".png", ".webp"}:
            extension = ".jpg"
        identifier = f"{slug}-{index:05d}"
        target = image_dir / f"{identifier}{extension}"
        target.write_bytes(value)
        result.append({
            "id": identifier,
            "baseId": identifier,
            "path": os.path.relpath(target, output.resolve()),
            "sha256": checksum,
            "label": 1,
            "source": f"Rapidata/{repository}",
            "generatorFamily": model,
            "split": "train",
            "license": "CDLA-Permissive-2.0",
            "originUrl": f"https://huggingface.co/datasets/Rapidata/{repository}/tree/{revision}",
            "contentGroup": "human-preference-generation",
            "splitGroup": hashlib.sha256(f"{revision}\0{model}\0{prompt}".encode()).hexdigest(),
        })
        if len(result) >= limit:
            break
    if not result:
        raise RuntimeError(f"{parquet} yielded no unique rows for target model {model}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--flux2-directory", type=pathlib.Path, required=True)
    parser.add_argument("--gpt4o", type=pathlib.Path, required=True)
    parser.add_argument("--ideogram", type=pathlib.Path, required=True)
    parser.add_argument("--real-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--calibration-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--per-family", type=int, default=88)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    image_dir = args.output / "images"
    image_dir.mkdir(exist_ok=True)
    parquet_paths = {"gpt4o": args.gpt4o, "ideogram": args.ideogram}
    ai: list[dict] = []
    provenance_sources = []
    flux_ledger = json.loads((args.flux2_directory / "provenance.json").read_text())
    for item in flux_ledger["rows"]:
        source = pathlib.Path(item["path"]).resolve()
        identifier = f"flux2-klein-{item['index']:05d}"
        ai.append({
            "id": identifier,
            "baseId": identifier,
            "path": os.path.relpath(source, args.output.resolve()),
            "sha256": item["sha256"],
            "label": 1,
            "source": flux_ledger["dataset"],
            "generatorFamily": flux_ledger["generatorFamily"],
            "split": "train",
            "license": flux_ledger["license"],
            "originUrl": item["originUrl"],
            "contentGroup": "generated-animal",
        })
    provenance_sources.append({
        "dataset": flux_ledger["dataset"], "revision": flux_ledger["revision"],
        "license": flux_ledger["license"], "count": len(flux_ledger["rows"]),
    })
    for slug, repository, model, image_column, model_column, revision in SOURCES:
        parquet = parquet_paths[slug]
        rows = extract(
            parquet, slug=slug, repository=repository, model=model, image_column=image_column,
            model_column=model_column, revision=revision, image_dir=image_dir,
            output=args.output, limit=args.per_family,
        )
        ai.extend(rows)
        provenance_sources.append({
            "slug": slug, "model": model, "revision": revision,
            "targetColumn": image_column, "parquetSha256": digest_file(parquet),
            "count": len(rows),
        })
    real = load_jsonl(args.real_manifest, args.output, "train")
    real = [row for row in real if row["label"] == 0][:len(ai)]
    if len(real) != len(ai):
        raise RuntimeError(f"need {len(ai)} real rows, found {len(real)}")
    calibration = load_jsonl(args.calibration_manifest, args.output, "calibration")
    manifest = args.output / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in real + ai + calibration))
    provenance = {
        "manifest": str(manifest),
        "counts": {"trainReal": len(real), "trainAi": len(ai), "calibration": len(calibration)},
        "sources": provenance_sources,
        "notes": "Only the declared target image side was extracted; comparison-side images were not written.",
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
