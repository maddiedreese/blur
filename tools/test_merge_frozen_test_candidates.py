from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from merge_frozen_test_candidates import merge, merged_rows


def candidate(identifier: str, path: str, data: bytes, label: int, source: str, **extra: str) -> dict:
    return {
        "id": identifier, "baseId": identifier, "sha256": hashlib.sha256(data).hexdigest(),
        "path": path, "label": label, "source": source, **extra,
    }


class MergeFrozenCandidatesTests(unittest.TestCase):
    def test_rebases_paths_and_sorts_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); a = root / "a"; b = root / "b"; a.mkdir(); b.mkdir()
            (a / "one.jpg").write_bytes(b"one"); (b / "two.jpg").write_bytes(b"two")
            (a / "rows.jsonl").write_text(json.dumps(candidate("z", "one.jpg", b"one", 1, "ai", uid="u")) + "\n")
            (b / "rows.jsonl").write_text(json.dumps(candidate("a", "two.jpg", b"two", 0, "real")) + "\n")
            output = root / "combined.jsonl"
            rows = merged_rows([a / "rows.jsonl", b / "rows.jsonl"], output)
            self.assertEqual([row["id"] for row in rows], ["a", "z"])
            self.assertEqual((output.parent / rows[0]["path"]).resolve(), (b / "two.jpg").resolve())

    def test_rejects_duplicate_base_across_ledgers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "x.jpg").write_bytes(b"x"); (root / "y.jpg").write_bytes(b"y")
            first = root / "one.jsonl"; second = root / "two.jsonl"
            first.write_text(json.dumps(candidate("one", "x.jpg", b"x", 1, "ai") | {"baseId": "same"}) + "\n")
            second.write_text(json.dumps(candidate("two", "y.jpg", b"y", 1, "ai") | {"baseId": "same"}) + "\n")
            with self.assertRaisesRegex(ValueError, "duplicate baseId"):
                merged_rows([first, second], root / "combined.jsonl")

    def test_rejects_duplicate_byte_hash_across_ledgers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); a = root / "a"; b = root / "b"; a.mkdir(); b.mkdir()
            (a / "x.png").write_bytes(b"same"); (b / "y.png").write_bytes(b"same")
            (a / "rows.jsonl").write_text(json.dumps(candidate("a", "x.png", b"same", 0, "real")) + "\n")
            (b / "rows.jsonl").write_text(json.dumps(candidate("b", "y.png", b"same", 1, "ai")) + "\n")
            with self.assertRaisesRegex(ValueError, "duplicate sha256"):
                merged_rows([a / "rows.jsonl", b / "rows.jsonl"], root / "combined.jsonl")

    def test_rejects_duplicate_id_even_when_bases_and_bytes_differ(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); a = root / "a"; b = root / "b"; a.mkdir(); b.mkdir()
            (a / "x.png").write_bytes(b"x"); (b / "y.png").write_bytes(b"y")
            first = candidate("same", "x.png", b"x", 0, "real")
            second = candidate("same", "y.png", b"y", 1, "ai") | {"baseId": "different"}
            (a / "rows.jsonl").write_text(json.dumps(first) + "\n")
            (b / "rows.jsonl").write_text(json.dumps(second) + "\n")
            with self.assertRaisesRegex(ValueError, "duplicate id"):
                merged_rows([a / "rows.jsonl", b / "rows.jsonl"], root / "combined.jsonl")

    def test_rejects_hash_mismatch_and_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); a = root / "a"; b = root / "b"; a.mkdir(); b.mkdir()
            (a / "x.png").write_bytes(b"x"); (b / "y.png").write_bytes(b"y")
            second = b / "rows.jsonl"
            second.write_text(json.dumps(candidate("b", "y.png", b"y", 1, "ai")) + "\n")
            first = a / "rows.jsonl"
            first.write_text(json.dumps(candidate("a", "x.png", b"wrong", 0, "real")) + "\n")
            with self.assertRaisesRegex(ValueError, "byte hash mismatch"):
                merged_rows([first, second], root / "combined.jsonl")
            first.write_text(json.dumps(candidate("a", "../b/y.png", b"y", 0, "real")) + "\n")
            with self.assertRaisesRegex(ValueError, "path escapes"):
                merged_rows([first, second], root / "combined.jsonl")

    def test_merge_writes_complete_hashed_output_and_will_not_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); a = root / "a"; b = root / "b"; a.mkdir(); b.mkdir()
            (a / "x.png").write_bytes(b"x"); (b / "y.png").write_bytes(b"y")
            first = a / "rows.jsonl"; second = b / "rows.jsonl"
            first.write_text(json.dumps(candidate("a", "x.png", b"x", 0, "real")) + "\n")
            second.write_text(json.dumps(candidate("b", "y.png", b"y", 1, "ai")) + "\n")
            output = root / "combined" / "candidates.jsonl"
            result = merge([first, second], output)
            self.assertEqual(result["rows"], 2)
            self.assertEqual(result["manifestSha256"], hashlib.sha256(output.read_bytes()).hexdigest())
            self.assertEqual(len(output.read_text().splitlines()), 2)
            self.assertEqual(list(output.parent.glob(f".{output.name}-*")), [])
            with self.assertRaises(FileExistsError):
                merge([first, second], output)


if __name__ == "__main__":
    unittest.main()
