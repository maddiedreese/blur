#!/usr/bin/env python3
"""Focused fail-closed tests for the frozen-test manifest builder."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from build_frozen_test_manifest import contains_protected_token, validate_candidate


class FrozenManifestValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "image.jpg").write_bytes(b"not-a-real-jpeg-but-stable-test-bytes")
        self.digest = hashlib.sha256((self.root / "image.jpg").read_bytes()).hexdigest()
        self.specs = {
            "ai-source": {
                "label": 1,
                "datasetRevision": "a" * 40,
                "license": "CC0-1.0",
                "contentGroups": ["prompted-generation"],
                "generatorFamily": "family-a",
                "generatorModel": "model-a-exact",
            }
        }
        self.row = {
            "id": "sample", "baseId": "sample", "path": "image.jpg",
            "sha256": self.digest, "label": 1, "source": "ai-source",
            "split": "test", "license": "CC0-1.0", "originUrl": "https://example.test/object",
            "datasetRevision": "a" * 40, "contentGroup": "prompted-generation",
            "generatorFamily": "family-a", "generatorModel": "model-a-exact",
            "uid": "upstream-uid", "promptGroup": "b" * 64,
            "attribution": "CC0 source statement",
        }
        self.forbidden = {field: set() for field in ("id", "baseId", "sha256")}

    def tearDown(self):
        self.temp.cleanup()

    def errors(self, **changes):
        return validate_candidate({**self.row, **changes}, self.root, self.specs, self.forbidden)

    def test_accepts_a_fully_pinned_hash_verified_base(self):
        self.assertEqual(self.errors(), [])

    def test_rejects_revision_license_and_exact_model_mismatch(self):
        self.assertTrue(any("datasetRevision" in item for item in self.errors(datasetRevision="wrong")))
        self.assertTrue(any("license" in item for item in self.errors(license="unknown")))
        self.assertTrue(any("generatorModel" in item for item in self.errors(generatorModel="family-alias")))

    def test_rejects_hash_path_and_attribution_failures(self):
        self.assertTrue(any("hash mismatch" in item for item in self.errors(sha256="0" * 64)))
        self.assertTrue(any("missing image" in item for item in self.errors(path="absent.jpg")))
        self.assertTrue(any("attribution" in item for item in self.errors(attribution="")))
        outside = self.root.parent / "outside.jpg"
        outside.write_bytes(b"outside")
        try:
            digest = hashlib.sha256(outside.read_bytes()).hexdigest()
            self.assertTrue(any("escapes" in item for item in self.errors(path="../outside.jpg", sha256=digest)))
        finally:
            outside.unlink()

    def test_rejects_derivatives_protected_tokens_and_overlap(self):
        self.assertTrue(any("unique bases" in item for item in self.errors(baseId="other")))
        self.assertTrue(any("protected source token" in item for item in self.errors(originUrl="https://example.test/laion/x")))
        forbidden = {field: set(values) for field, values in self.forbidden.items()}
        forbidden["sha256"].add(self.digest)
        errors = validate_candidate(self.row, self.root, self.specs, forbidden)
        self.assertTrue(any("forbidden sha256 overlap" in item for item in errors))

    def test_protected_identifiers_use_token_boundaries(self):
        self.assertTrue(contains_protected_token("https://example.test/coco/image.jpg"))
        self.assertTrue(contains_protected_token("generator=flux.2-klein"))
        self.assertFalse(contains_protected_token("https://example.test/coconut-cake.jpg"))


if __name__ == "__main__":
    unittest.main()
