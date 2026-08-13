from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from model_pipeline import Record, balanced_accuracy, fixed_threshold_metrics, preprocess_image, read_manifest, validate_source_separation
from train_detector import (
    ManifestDataset,
    assert_no_protected_training_sources,
    configure_determinism,
    configure_partial_backbone,
    family_balanced_sample_weights,
    optimizer_parameter_groups,
    passes_constraints,
)


class TinyVisionTransformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = torch.nn.Linear(2, 2)
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(4)])
        self.norm = torch.nn.LayerNorm(2)
        self.head = torch.nn.Linear(2, 1)


class ModelPipelineTests(unittest.TestCase):
    def test_family_balanced_weights_split_class_and_ai_family_mass(self) -> None:
        records = [
            Record("r1", "r1", pathlib.Path("r1"), 0, "real", None, "train", "MIT", "https://x/r1", "0" * 64),
            Record("r2", "r2", pathlib.Path("r2"), 0, "real", None, "train", "MIT", "https://x/r2", "1" * 64),
            Record("a1", "a1", pathlib.Path("a1"), 1, "ai", "family-a", "train", "MIT", "https://x/a1", "2" * 64),
            Record("a2", "a2", pathlib.Path("a2"), 1, "ai", "family-a", "train", "MIT", "https://x/a2", "3" * 64),
            Record("b1", "b1", pathlib.Path("b1"), 1, "ai", "family-b", "train", "MIT", "https://x/b1", "4" * 64),
        ]
        dataset = ManifestDataset(records, train=True, paired_degradations=True)
        weights = family_balanced_sample_weights(dataset).numpy()
        self.assertAlmostEqual(float(weights[:4].sum()), 0.5)
        self.assertAlmostEqual(float(weights[4:8].sum()), 0.25)
        self.assertAlmostEqual(float(weights[8:].sum()), 0.25)

    def test_partial_backbone_unfreezes_only_tail_norm_and_head(self) -> None:
        model = TinyVisionTransformer()
        report = configure_partial_backbone(model, 2)
        self.assertEqual(report["unfrozen_block_indices"], [2, 3])
        trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        self.assertTrue(any(name.startswith("blocks.2.") for name in trainable))
        self.assertTrue(any(name.startswith("blocks.3.") for name in trainable))
        self.assertTrue(any(name.startswith("norm.") for name in trainable))
        self.assertTrue(any(name.startswith("head.") for name in trainable))
        self.assertFalse(any(name.startswith("patch_embed.") for name in trainable))
        self.assertFalse(any(name.startswith("blocks.1.") for name in trainable))

    def test_head_only_mode_preserves_frozen_final_norm(self) -> None:
        model = TinyVisionTransformer()
        configure_partial_backbone(model, 0)
        trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        self.assertEqual(trainable, {"head.weight", "head.bias"})

    def test_partial_backbone_rejects_invalid_block_count(self) -> None:
        model = TinyVisionTransformer()
        with self.assertRaisesRegex(ValueError, r"\[0, 4\]"):
            configure_partial_backbone(model, 5)

    def test_discriminative_optimizer_groups_use_requested_rates(self) -> None:
        model = TinyVisionTransformer()
        configure_partial_backbone(model, 1)
        groups = optimizer_parameter_groups(
            model, head_learning_rate=2e-5, backbone_learning_rate=2e-6, weight_decay=0.05,
        )
        rates = {str(group["group_name"]): group["lr"] for group in groups}
        self.assertEqual(rates, {"backbone": 2e-6, "head": 2e-5})
        grouped_ids = [id(parameter) for group in groups for parameter in group["params"]]  # type: ignore[union-attr]
        self.assertEqual(len(grouped_ids), len(set(grouped_ids)))

    def test_determinism_configuration_reseeds_all_rngs(self) -> None:
        configure_determinism(123)
        first = (np.random.random(), torch.rand(1).item())
        configure_determinism(123)
        second = (np.random.random(), torch.rand(1).item())
        self.assertEqual(first, second)
        self.assertTrue(torch.are_deterministic_algorithms_enabled())

    def test_preprocess_matches_model_contract(self) -> None:
        image = Image.new("RGB", (640, 480), (100, 120, 140))
        tensor = preprocess_image(image)
        self.assertEqual(tuple(tensor.shape), (3, 384, 384))
        self.assertEqual(str(tensor.dtype), "torch.float32")

    def test_thumbnail_degradation_is_deterministic_and_distinct(self) -> None:
        pixels = np.tile(np.arange(640, dtype=np.uint8), (480, 1))
        image = Image.fromarray(np.stack([pixels, np.flipud(pixels), pixels], axis=2), mode="RGB")
        original = preprocess_image(image)
        first = preprocess_image(image, augment_seed=123)
        second = preprocess_image(image, augment_seed=123)
        self.assertTrue(np.array_equal(first.numpy(), second.numpy()))
        self.assertFalse(np.array_equal(original.numpy(), first.numpy()))

    def test_rejects_source_leakage(self) -> None:
        records = [
            Record("a", "a", pathlib.Path("a"), 1, "collection-a", "generator-x", "train", "MIT", "https://example/a", "0" * 64),
            Record("b", "b", pathlib.Path("b"), 1, "collection-b", "generator-x", "calibration", "MIT", "https://example/b", "1" * 64),
        ]
        with self.assertRaisesRegex(ValueError, "source groups cross splits"):
            validate_source_separation(records)

    def test_rejects_derivative_leakage(self) -> None:
        records = [
            Record("a", "original", pathlib.Path("a"), 0, "camera-a", None, "train", "MIT", "https://example/a", "0" * 64),
            Record("b", "original", pathlib.Path("b"), 0, "camera-b", None, "test", "MIT", "https://example/b", "1" * 64),
        ]
        with self.assertRaisesRegex(ValueError, "lineage crosses splits"):
            validate_source_separation(records)

    def test_manifest_requires_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = pathlib.Path(directory) / "manifest.jsonl"
            manifest.write_text(json.dumps({"id": "x"}) + "\n")
            with self.assertRaisesRegex(ValueError, "missing fields"):
                read_manifest(manifest, require_files=False)

    def test_balanced_accuracy_uses_fixed_threshold(self) -> None:
        labels = np.asarray([0, 0, 1, 1])
        scores = np.asarray([0.1, 0.8, 0.7, 0.9])
        self.assertEqual(balanced_accuracy(labels, scores, threshold=0.65), 0.75)

    def test_fixed_threshold_metrics_report_recall_and_fpr(self) -> None:
        labels = np.asarray([0, 0, 1, 1])
        logits = np.asarray([-2.0, 2.0, 1.0, 2.0])
        metrics = fixed_threshold_metrics(labels, logits, threshold=0.65)
        self.assertEqual(metrics["ai_recall"], 1.0)
        self.assertEqual(metrics["real_recall"], 0.5)
        self.assertEqual(metrics["false_positive_rate"], 0.5)
        self.assertEqual(metrics["balanced_accuracy"], 0.75)

    def test_protected_sources_are_rejected_from_training(self) -> None:
        protected = Record(
            "x", "x", pathlib.Path("x"), 1, "commercial-eval", "MidjourneyV6", "train",
            "evaluation-only", "https://example/x", "0" * 64,
        )
        with self.assertRaisesRegex(ValueError, "protected"):
            assert_no_protected_training_sources([protected])

    def test_selection_constraints_prevent_full_resolution_regression(self) -> None:
        baseline = {"balanced_accuracy": 0.9, "real_recall": 0.9}
        full = {"balanced_accuracy": 0.85, "ai_recall": 0.9, "real_recall": 0.8, "false_positive_rate": 0.2}
        stress = {"balanced_accuracy": 0.85, "ai_recall": 0.85, "real_recall": 0.85, "false_positive_rate": 0.15}
        self.assertFalse(passes_constraints(
            full, stress, baseline, min_ai_recall=0.75, min_real_recall=0.75, max_fpr=0.25,
            max_full_ba_drop=0.01, max_full_real_recall_drop=0.02,
        ))


if __name__ == "__main__":
    unittest.main()
