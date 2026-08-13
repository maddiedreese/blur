#!/usr/bin/env python3
"""Build a frozen test manifest from already acquired, license-audited rows.

This tool deliberately does not download or score images. Acquisition emits a
candidate JSONL with local paths and byte hashes; this command verifies every
byte, rejects protected/training/calibration overlap, enforces pinned source
contracts and deterministically selects the configured quotas.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTECTED_TOKENS = {
    "laion", "midjourney", "raise", "dfgan", "commfor", "coco", "diffusiondb",
    "flux.2-klein", "flux2-klein", "gpt4o", "gpt-4o", "4o-26-3-25", "ideogram",
}
REQUIRED_FIELDS = {
    "id", "baseId", "path", "sha256", "label", "source", "split",
    "license", "originUrl", "datasetRevision", "contentGroup",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stable_key(row: dict, seed: str) -> str:
    material = f"{seed}\0{row['source']}\0{row['baseId']}\0{row['sha256']}"
    return hashlib.sha256(material.encode()).hexdigest()


def contains_protected_token(value: str) -> bool:
    """Match protected provenance identifiers, not substrings such as coconut."""
    lowered = value.lower()
    return any(
        re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", lowered)
        for token in PROTECTED_TOKENS
    )


def load_forbidden(paths: list[Path]) -> dict[str, set[str]]:
    values = {field: set() for field in ("id", "baseId", "sha256", "promptGroup")}
    for path in paths:
        for row in read_jsonl(path):
            for field in values:
                if row.get(field):
                    values[field].add(str(row[field]))
    return values


def validate_candidate(row: dict, root: Path, source_specs: dict, forbidden: dict) -> list[str]:
    errors = []
    missing = sorted(REQUIRED_FIELDS - row.keys())
    if missing:
        errors.append(f"{row.get('id', '<missing>')}: missing {', '.join(missing)}")
        return errors
    if row["split"] != "test":
        errors.append(f"{row['id']}: split must be test")
    if row["id"] != row["baseId"]:
        errors.append(f"{row['id']}: acquisition rows must be unique bases, not derivatives")
    if row["label"] not in (0, 1):
        errors.append(f"{row['id']}: invalid label")
    spec = source_specs.get(row["source"])
    if not spec:
        errors.append(f"{row['id']}: unapproved source {row['source']}")
    else:
        for field in ("label", "datasetRevision", "license"):
            if row[field] != spec[field]:
                errors.append(f"{row['id']}: {field} does not match pinned source contract")
        if row["contentGroup"] not in spec["contentGroups"]:
            errors.append(f"{row['id']}: unapproved contentGroup {row['contentGroup']}")
        if row["label"] == 1:
            generators = spec.get("generators") or [spec]
            generator = next((item for item in generators if item.get("generatorModel") == row.get("generatorModel")), None)
            if not generator:
                errors.append(f"{row['id']}: generatorModel does not match the exact source identifier")
            elif row.get("generatorFamily") != generator.get("generatorFamily"):
                errors.append(f"{row['id']}: generatorFamily does not match source contract")
            if spec.get("datasetConfig") and row.get("datasetConfig") != spec["datasetConfig"]:
                errors.append(f"{row['id']}: datasetConfig does not match source contract")
            if spec.get("datasetSplit") and row.get("datasetSplit") != spec["datasetSplit"]:
                errors.append(f"{row['id']}: datasetSplit does not match source contract")
            if not isinstance(row.get("uid"), str) or not row["uid"]:
                errors.append(f"{row['id']}: source UID is required")
            if not isinstance(row.get("promptGroup"), str) or not re.fullmatch(r"[0-9a-f]{64}", row["promptGroup"]):
                errors.append(f"{row['id']}: promptGroup SHA-256 is required")
            if spec.get("datasetSplit"):
                if not isinstance(row.get("rowIndex"), int) or row["rowIndex"] < 0:
                    errors.append(f"{row['id']}: non-negative rowIndex is required")
                expected_locator = (
                    f"hf://datasets/{row['source']}@{row['datasetRevision']}/"
                    f"{row.get('datasetConfig')}/{row.get('datasetSplit')}#row={row.get('rowIndex')};column=image"
                )
                if row.get("originLocator") != expected_locator:
                    errors.append(f"{row['id']}: originLocator does not match pinned row")
    searchable = " ".join(
        str(row.get(key, "")) for key in ("source", "generatorFamily", "generatorModel", "originUrl")
    ).lower()
    if contains_protected_token(searchable):
        errors.append(f"{row['id']}: protected source token")
    if not isinstance(row["sha256"], str) or len(row["sha256"]) != 64:
        errors.append(f"{row['id']}: invalid SHA-256")
    else:
        image = (root / row["path"]).resolve()
        try:
            image.relative_to(root.resolve())
        except ValueError:
            errors.append(f"{row['id']}: path escapes candidate root")
        if not image.is_file():
            errors.append(f"{row['id']}: missing image")
        elif hashlib.sha256(image.read_bytes()).hexdigest() != row["sha256"]:
            errors.append(f"{row['id']}: byte hash mismatch")
    for field, values in forbidden.items():
        if row.get(field) and str(row[field]) in values:
            errors.append(f"{row['id']}: forbidden {field} overlap")
    if not row.get("attribution"):
        errors.append(f"{row['id']}: attribution/public-domain statement is required")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidates", type=Path)
    parser.add_argument("--spec", type=Path, default=ROOT / "tools/frozen_test_sources.json")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", default="blur-frozen-test-v1")
    parser.add_argument("--forbid", action="append", type=Path, default=[])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    config = json.loads(args.spec.read_text())
    specs = {item["source"]: item for item in config["sources"]}
    forbidden_paths = args.forbid or [
        ROOT / "data/recent-training/manifest.jsonl",
        ROOT / "data/thumbnail-training/manifest.jsonl",
        ROOT / "data/commfor-calibration-64/manifest.jsonl",
        ROOT / "data/commfor-calibration-stress/manifest.jsonl",
        ROOT / "data/commfor-commercial-64/manifest.jsonl",
        ROOT / "data/commfor-commercial-stress/manifest.jsonl",
    ]
    forbidden = load_forbidden(forbidden_paths)
    candidate_root = args.candidates.parent
    candidates = read_jsonl(args.candidates)
    errors = []
    seen = defaultdict(set)
    seen_generator_uid: set[tuple[str, str, str]] = set()
    for row in candidates:
        errors.extend(validate_candidate(row, candidate_root, specs, forbidden))
        for field in ("id", "baseId", "sha256", "originLocator"):
            if not row.get(field):
                continue
            if row.get(field) in seen[field]:
                errors.append(f"{row.get('id', '<missing>')}: duplicate {field}")
            seen[field].add(row.get(field))
        if row.get("label") == 1 and row.get("uid"):
            generator_uid = (str(row.get("source")), str(row.get("generatorModel")), str(row["uid"]))
            if generator_uid in seen_generator_uid:
                errors.append(f"{row.get('id', '<missing>')}: duplicate generatorModel/uid")
            seen_generator_uid.add(generator_uid)
    if errors:
        raise SystemExit("candidate manifest rejected:\n- " + "\n- ".join(errors))

    selected = []
    for source, spec in specs.items():
        pool = [row for row in candidates if row["source"] == source]
        generators = spec.get("generators")
        if generators:
            for generator in generators:
                generator_pool = [row for row in pool if row.get("generatorModel") == generator["generatorModel"]]
                generator_pool.sort(key=lambda row: stable_key(row, args.seed))
                if len(generator_pool) < generator["count"]:
                    raise SystemExit(
                        f"source {source}/{generator['generatorModel']} has {len(generator_pool)} valid rows; "
                        f"requires {generator['count']}"
                    )
                selected.extend(generator_pool[:generator["count"]])
        else:
            pool.sort(key=lambda row: stable_key(row, args.seed))
            if len(pool) < spec["count"]:
                raise SystemExit(f"source {source} has {len(pool)} valid rows; requires {spec['count']}")
            selected.extend(pool[:spec["count"]])
    selected.sort(key=lambda row: (row["label"], row["source"], stable_key(row, args.seed)))

    counts = Counter(row["label"] for row in selected)
    real_sources = {row["source"] for row in selected if row["label"] == 0}
    generators = {row["generatorFamily"] for row in selected if row["label"] == 1}
    if counts[0] < 500 or counts[1] < 500 or len(real_sources) < 3 or len(generators) < 3:
        raise SystemExit("configured selection does not satisfy the frozen-test breadth floor")
    args.output.write_text("".join(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n" for row in selected))
    print(json.dumps({
        "rows": len(selected), "real": counts[0], "ai": counts[1],
        "realSources": sorted(real_sources), "generatorFamilies": sorted(generators),
        "manifestSha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }, indent=2))


if __name__ == "__main__":
    main()
