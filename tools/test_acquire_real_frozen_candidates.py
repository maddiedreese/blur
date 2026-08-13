#!/usr/bin/env python3
"""Offline tests for the frozen real-candidate acquisition adapters."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from urllib.parse import parse_qs, urlparse

from PIL import Image

import acquire_real_frozen_candidates as acquisition
import build_frozen_test_manifest as builder


def image_bytes(index: int, size: tuple[int, int] = (160, 144)) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", size, (index % 251, (index * 3) % 251, (index * 7) % 251)).save(output, "PNG")
    return output.getvalue()


def webdataset_tar(count: int, *, duplicate_first: bool = False, offset: int = 0) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        for index in range(count):
            key = f"sample-{index:03d}"
            pixels = image_bytes(offset + (0 if duplicate_first and index == 1 else index))
            metadata = json.dumps({
                "image_id": key,
                "download_url": f"https://origin.example/{key}",
            }).encode()
            for name, data in ((f"{key}.png", pixels), (f"{key}.json", metadata)):
                member = tarfile.TarInfo(name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
    return output.getvalue()


class FakeClient:
    def __init__(self, revisions: dict[str, str], shards: dict[str, bytes] | None = None, licenses: dict[str, str] | None = None):
        self.revisions = revisions
        self.shards = shards or {}
        self.licenses = licenses or {source: ("cc0-1.0" if source == "nyuuzyou/pxhere" else "cc-by-4.0") for source in revisions}
        self.byte_payloads: dict[str, bytes] = {}
        self.viewer_rows: dict[str, list[dict]] = {}
        self.opened: list[str] = []

    def json(self, url: str) -> dict:
        if url.startswith("https://huggingface.co/api/datasets/"):
            dataset = url.removeprefix("https://huggingface.co/api/datasets/")
            return {
                "sha": self.revisions[dataset],
                "gated": False,
                "private": False,
                "disabled": False,
                "cardData": {"license": self.licenses[dataset]},
                "siblings": [{"rfilename": name} for name in self.shards],
            }
        query = parse_qs(urlparse(url).query)
        split = query["split"][0]
        offset = int(query["offset"][0])
        length = int(query["length"][0])
        source = self.viewer_rows.get(split, [])
        return {"rows": source[offset:offset + length], "num_rows_total": len(source)}

    def open(self, url: str) -> io.BytesIO:
        self.opened.append(url)
        name = url.rsplit("/", 1)[-1]
        return io.BytesIO(self.shards[name])

    def bytes(self, url: str) -> bytes:
        self.opened.append(url)
        return self.byte_payloads[url]


class AcquisitionTests(unittest.TestCase):
    def test_validate_image_rejects_corrupt_and_tiny_data(self) -> None:
        with self.assertRaises(Exception):
            acquisition.validate_image(b"not an image")
        with self.assertRaisesRegex(ValueError, "too small"):
            acquisition.validate_image(image_bytes(1, (64, 64)))

    def test_webdataset_emits_exact_margin_with_pin_and_provenance(self) -> None:
        revision = "a" * 40
        spec = {
            "source": "nyuuzyou/pxhere", "datasetRevision": revision,
            "license": "CC0-1.0", "label": 0, "count": 1,
            "contentGroups": ["camera-photograph"],
        }
        shard = "pxhere-00000.tar"
        client = FakeClient({spec["source"]: revision}, {shard: webdataset_tar(12, duplicate_first=True)})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = acquisition.webdataset_candidates(client, spec, root, 0.1, "fixed-seed")
            self.assertEqual(len(rows), 11)
            self.assertEqual(len({row["sha256"] for row in rows}), 11)
            self.assertTrue(all(row["datasetRevision"] == revision for row in rows))
            self.assertTrue(all(row["license"] == "CC0-1.0" for row in rows))
            self.assertTrue(all(row["originUrl"].startswith("https://origin.example/") for row in rows))
            self.assertTrue(all(row["upstreamLocator"].startswith(shard + "#") for row in rows))
            self.assertTrue(all((root / row["path"]).is_file() for row in rows))
            self.assertTrue(all(hashlib.sha256((root / row["path"]).read_bytes()).hexdigest() == row["sha256"] for row in rows))
            self.assertFalse(any(path.suffix == ".part" for path in root.rglob("*")))
            source_specs = {spec["source"]: spec}
            forbidden = {field: set() for field in ("id", "baseId", "sha256")}
            self.assertEqual(builder.validate_candidate(rows[0], root, source_specs, forbidden), [])

    def test_cord_uses_test_then_validation_and_never_records_signed_url(self) -> None:
        source = "naver-clova-ix/cord-v2"
        revision = "b" * 40
        spec = {
            "source": source, "datasetRevision": revision,
            "license": "CC-BY-4.0", "label": 0, "count": 1,
            "contentGroups": ["text-heavy-receipt"],
        }
        client = FakeClient({source: revision})
        for split, count in (("test", 6), ("validation", 6)):
            rows = []
            for index in range(count):
                signed = f"https://signed.example/{split}/{index}?token=secret"
                client.byte_payloads[signed] = image_bytes(index + (0 if split == "test" else 100))
                rows.append({
                    "row_idx": index,
                    "row": {
                        "image": {"src": signed},
                        "ground_truth": json.dumps({"meta": {"image_id": f"receipt-{split}-{index}"}}),
                    },
                })
            client.viewer_rows[split] = rows
        with tempfile.TemporaryDirectory() as directory:
            rows = acquisition.cord_candidates(client, spec, Path(directory), 0.1)
        self.assertEqual(len(rows), 11)
        self.assertEqual(sum("/test/" in row["originUrl"] for row in rows), 6)
        self.assertEqual(sum("/validation/" in row["originUrl"] for row in rows), 5)
        self.assertTrue(all(row["originUrl"].startswith(f"hf://datasets/{source}@{revision}/") for row in rows))
        self.assertFalse(any("token=" in row["originUrl"] for row in rows))
        self.assertTrue(all(row["sourceRecordId"].startswith("receipt-") for row in rows))

    def test_acquire_rolls_back_output_on_failure(self) -> None:
        source = "naver-clova-ix/cord-v2"
        revision = "c" * 40
        config = {"sources": [{
            "source": source, "datasetRevision": revision,
            "license": "CC-BY-4.0", "label": 0, "count": 1,
            "contentGroups": ["text-heavy-receipt"],
        }]}
        client = FakeClient({source: revision})
        client.viewer_rows["test"] = [{"row_idx": 0, "row": {"image": {"src": "missing"}}}]
        client.viewer_rows["validation"] = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec_path = root / "sources.json"
            spec_path.write_text(json.dumps(config))
            output = root / "acquired"
            with self.assertRaises(KeyError):
                acquisition.acquire(spec_path, output, 0.1, "seed", client)
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".acquired-*")), [])

    def test_acquire_installs_complete_directory_and_manifest_atomically(self) -> None:
        revisions = {
            "nyuuzyou/pxhere": "1" * 40,
            "Mitsua/art-museums-pd-440k": "2" * 40,
            "naver-clova-ix/cord-v2": "3" * 40,
        }
        sources = [
            {"source": source, "datasetRevision": revision,
             "license": "CC0-1.0" if source == "nyuuzyou/pxhere" else "CC-BY-4.0",
             "label": 0, "count": 1,
             "contentGroups": ["camera-photograph" if source == "nyuuzyou/pxhere" else
                               "human-art" if source.startswith("Mitsua/") else "text-heavy-receipt"]}
            for source, revision in revisions.items()
        ]
        client = FakeClient(revisions, {
            "pxhere-00000.tar": webdataset_tar(11),
            "ArtMuseumsPD_00000.tar": webdataset_tar(11, offset=50),
        })
        cord_rows = []
        for index in range(11):
            url = f"https://signed.example/test/{index}"
            client.byte_payloads[url] = image_bytes(150 + index)
            cord_rows.append({"row_idx": index, "row": {"image": {"src": url}}})
        client.viewer_rows["test"] = cord_rows
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec_path = root / "sources.json"
            spec_path.write_text(json.dumps({"sources": sources}))
            output = root / "acquired"
            provenance = acquisition.acquire(spec_path, output, 0.1, "seed", client)
            lines = (output / "candidates.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), 33)
            self.assertEqual(provenance["sourceCounts"], {source: 11 for source in revisions})
            self.assertEqual(
                hashlib.sha256((output / "candidates.jsonl").read_bytes()).hexdigest(),
                provenance["candidateManifestSha256"],
            )
            self.assertTrue((output / "provenance.json").is_file())
            self.assertEqual(list(root.glob(".acquired-*")), [])

    def test_revision_mismatch_fails_before_opening_shards(self) -> None:
        spec = {
            "source": "nyuuzyou/pxhere", "datasetRevision": "d" * 40,
            "license": "CC0-1.0", "label": 0, "count": 1,
        }
        client = FakeClient({spec["source"]: "e" * 40}, {"pxhere-00000.tar": webdataset_tar(11)})
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "expected revision"):
                acquisition.webdataset_candidates(client, spec, Path(directory), 0.1, "seed")
        self.assertEqual(client.opened, [])


if __name__ == "__main__":
    unittest.main()
