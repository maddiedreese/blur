from __future__ import annotations

import io
import unittest

from PIL import Image

from acquire_xaigd_frozen_test import (
    LICENSE, REVISION, normalize_prompt, prompt_group, select_rows,
    validate_image_bytes, validate_repository_metadata,
)


def viewer_item(index: int, uid: str, model: str, prompt: str) -> dict:
    return {
        "row_idx": index,
        "row": {
            "generator": model, "uid": uid, "original_prompt": prompt,
            "image": {"src": f"https://datasets-server.huggingface.co/cached/--/{REVISION}/--/{index}/image.jpg"},
            "width": 16, "height": 12, "image_format": "JPEG",
        },
    }


class XAIGDAcquisitionTests(unittest.TestCase):
    def test_normalization_and_prompt_group_are_stable(self) -> None:
        self.assertEqual(normalize_prompt("  Café\n  STREET  "), "café street")
        self.assertEqual(prompt_group("Café  street"), prompt_group("  Café\nSTREET "))

    def test_repository_pin_and_license_fail_closed(self) -> None:
        validate_repository_metadata({"sha": REVISION, "cardData": {"license": LICENSE.lower()}})
        with self.assertRaisesRegex(ValueError, "revision moved"):
            validate_repository_metadata({"sha": "0" * 40, "cardData": {"license": LICENSE.lower()}})
        with self.assertRaisesRegex(ValueError, "license changed"):
            validate_repository_metadata({"sha": REVISION, "cardData": {"license": "unknown"}})

    def test_selection_keeps_same_prompt_group_inside_one_test_split(self) -> None:
        quotas = {"HYDiT_1.2-raw": 1, "Infinity-raw": 1}
        page = {
            "partial": False, "num_rows_total": 2419,
            "rows": [
                viewer_item(1, "uid-a", "HYDiT_1.2-raw", "Shared prompt"),
                viewer_item(2, "uid-b", "Infinity-raw", " shared  PROMPT "),
            ],
        }
        rows = select_rows([page], quotas)
        self.assertEqual(len(rows), 2)
        self.assertEqual(prompt_group(rows[0]["original_prompt"]), prompt_group(rows[1]["original_prompt"]))

    def test_selection_allows_cross_family_uid_but_rejects_same_model_uid(self) -> None:
        cross_family = {
            "partial": False, "num_rows_total": 2419,
            "rows": [
                viewer_item(1, "same", "HYDiT_1.2-raw", "one"),
                viewer_item(2, "same", "Infinity-raw", "two"),
            ],
        }
        self.assertEqual(len(select_rows([cross_family], {"HYDiT_1.2-raw": 1, "Infinity-raw": 1})), 2)
        same_model = {
            "partial": False, "num_rows_total": 2419,
            "rows": [
                viewer_item(1, "same", "HYDiT_1.2-raw", "one"),
                viewer_item(2, "same", "HYDiT_1.2-raw", "two"),
            ],
        }
        with self.assertRaisesRegex(ValueError, "duplicate generator/UID pair"):
            select_rows([same_model], {"HYDiT_1.2-raw": 2})

    def test_selection_rejects_duplicate_row_quota_drift_and_unsafe_uid(self) -> None:
        duplicate_row = {
            "partial": False, "num_rows_total": 2419,
            "rows": [
                viewer_item(1, "a", "HYDiT_1.2-raw", "one"),
                viewer_item(1, "b", "Infinity-raw", "two"),
            ],
        }
        with self.assertRaisesRegex(ValueError, "duplicate row index"):
            select_rows([duplicate_row], {"HYDiT_1.2-raw": 1, "Infinity-raw": 1})
        with self.assertRaisesRegex(ValueError, "quota mismatch"):
            select_rows([{"partial": False, "num_rows_total": 2419, "rows": duplicate_row["rows"][:1]}], {"HYDiT_1.2-raw": 2})
        unsafe = {"partial": False, "num_rows_total": 2419, "rows": [viewer_item(1, "../escape", "HYDiT_1.2-raw", "one")]}
        with self.assertRaisesRegex(ValueError, "unsafe UID"):
            select_rows([unsafe], {"HYDiT_1.2-raw": 1})

    def test_image_validation_checks_decode_dimensions_and_format(self) -> None:
        buffer = io.BytesIO()
        Image.new("RGB", (16, 12), "red").save(buffer, "JPEG")
        row = {"uid": "x", "width": 16, "height": 12, "image_format": "JPEG"}
        digest, width, height, fmt = validate_image_bytes(buffer.getvalue(), row)
        self.assertEqual(len(digest), 64)
        self.assertEqual((width, height, fmt), (16, 12, "JPEG"))
        with self.assertRaisesRegex(ValueError, "decoded dimensions"):
            validate_image_bytes(buffer.getvalue(), {**row, "width": 17})
