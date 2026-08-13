#!/usr/bin/env python3
"""Deterministically merge separately audited frozen-test candidate ledgers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

REQUIRED_FIELDS = {"id", "baseId", "sha256", "path", "label", "source"}


def merged_rows(inputs: list[Path], output: Path) -> list[dict]:
    if len(inputs) < 2:
        raise ValueError("at least two independently acquired candidate manifests are required")
    rows = []
    seen = {field: set() for field in ("id", "baseId", "sha256", "originLocator")}
    seen_generator_uid: set[tuple[str, str, str]] = set()
    seen_manifests = set()
    for manifest in inputs:
        manifest = manifest.resolve()
        if manifest in seen_manifests:
            raise ValueError(f"duplicate input manifest {manifest}")
        seen_manifests.add(manifest)
        for line_number, line in enumerate(manifest.read_text().splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = sorted(field for field in REQUIRED_FIELDS if row.get(field) in (None, ""))
            if missing:
                raise ValueError(f"{manifest}:{line_number}: missing {', '.join(missing)}")
            for field in ("id", "baseId", "path", "source"):
                if not isinstance(row[field], str):
                    raise ValueError(f"{manifest}:{line_number}: {field} must be a string")
            if not isinstance(row["label"], int) or isinstance(row["label"], bool) or row["label"] not in (0, 1):
                raise ValueError(f"{manifest}:{line_number}: label must be 0 or 1")
            if not isinstance(row["sha256"], str) or len(row["sha256"]) != 64:
                raise ValueError(f"{manifest}:{line_number}: invalid SHA-256")
            source_path = (manifest.parent / row["path"]).resolve()
            try:
                source_path.relative_to(manifest.parent)
            except ValueError as error:
                raise ValueError(f"{manifest}:{line_number}: image path escapes candidate directory") from error
            if not source_path.is_file():
                raise ValueError(f"{manifest}:{line_number}: missing image {source_path}")
            if hashlib.sha256(source_path.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError(f"{manifest}:{line_number}: byte hash mismatch")
            row["path"] = os.path.relpath(source_path, output.parent.resolve())
            for field in seen:
                value = row.get(field)
                if not value:
                    continue
                if value in seen[field]:
                    raise ValueError(f"{manifest}:{line_number}: duplicate {field} {value!r}")
                seen[field].add(value)
            if row.get("label") == 1 and row.get("uid"):
                generator_uid = (str(row.get("source")), str(row.get("generatorModel")), str(row["uid"]))
                if generator_uid in seen_generator_uid:
                    raise ValueError(f"{manifest}:{line_number}: duplicate generatorModel/uid {generator_uid!r}")
                seen_generator_uid.add(generator_uid)
            rows.append(row)
    rows.sort(key=lambda row: (int(row["label"]), str(row["source"]), str(row["baseId"])))
    return rows


def merge(inputs: list[Path], output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = merged_rows(inputs, output)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=output.parent, prefix=f".{output.name}-", delete=False
        ) as handle:
            temporary = Path(handle.name)
            for row in rows:
                handle.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {
        "output": str(output),
        "rows": len(rows),
        "manifestSha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(merge(args.inputs, args.output), indent=2))


if __name__ == "__main__":
    main()
