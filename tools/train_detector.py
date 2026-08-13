#!/usr/bin/env python3
"""Fine-tune the pinned detector using a provenance-rich, source-separated manifest."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random

import numpy as np
from PIL import Image
from safetensors.torch import load_file, save_file
import timm
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from model_pipeline import Record, fixed_threshold_metrics, preprocess_image, read_manifest, sha256

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE_CHECKPOINT = ROOT / "models" / "commfor-model-384.safetensors"
SEED = 11997733
PROTECTED_TRAIN_TOKENS = ("midjourney", "laion", "raise", "dfgan")


def configure_determinism(seed: int = SEED) -> None:
    """Configure all RNGs and deterministic kernels before model construction."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def configure_partial_backbone(model: torch.nn.Module, last_n_blocks: int) -> dict[str, object]:
    """Freeze the backbone, then expose the head and optionally its final blocks.

    The final normalization is adapted with the blocks because its learned scale
    and bias are part of the representation delivered to the classifier.
    """
    blocks = getattr(model, "blocks", None)
    head = getattr(model, "head", None)
    if not isinstance(blocks, (torch.nn.ModuleList, torch.nn.Sequential)) or not isinstance(head, torch.nn.Module):
        raise ValueError("partial adaptation requires a timm ViT with an ordered block container and a head")
    if not 0 <= last_n_blocks <= len(blocks):
        raise ValueError(f"last_n_blocks must be in [0, {len(blocks)}]")

    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in head.parameters():
        parameter.requires_grad = True

    unfrozen_block_indices: list[int] = []
    if last_n_blocks:
        unfrozen_block_indices = list(range(len(blocks) - last_n_blocks, len(blocks)))
        for index in unfrozen_block_indices:
            for parameter in blocks[index].parameters():
                parameter.requires_grad = True
        for name in ("norm", "fc_norm"):
            module = getattr(model, name, None)
            if isinstance(module, torch.nn.Module):
                for parameter in module.parameters():
                    parameter.requires_grad = True

    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    return {
        "total_blocks": len(blocks),
        "unfrozen_block_indices": unfrozen_block_indices,
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "trainable_parameter_names": trainable,
    }


def optimizer_parameter_groups(
    model: torch.nn.Module,
    *,
    head_learning_rate: float,
    backbone_learning_rate: float,
    weight_decay: float,
) -> list[dict[str, object]]:
    if head_learning_rate <= 0 or backbone_learning_rate <= 0:
        raise ValueError("learning rates must be positive")
    head = getattr(model, "head", None)
    if not isinstance(head, torch.nn.Module):
        raise ValueError("model has no classification head")
    head_ids = {id(parameter) for parameter in head.parameters()}
    head_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad and id(parameter) in head_ids]
    backbone_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad and id(parameter) not in head_ids]
    groups: list[dict[str, object]] = []
    if backbone_parameters:
        groups.append({"params": backbone_parameters, "lr": backbone_learning_rate, "weight_decay": weight_decay, "group_name": "backbone"})
    if head_parameters:
        groups.append({"params": head_parameters, "lr": head_learning_rate, "weight_decay": weight_decay, "group_name": "head"})
    if not groups:
        raise ValueError("model has no trainable parameters")
    return groups


class ManifestDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, records: list[Record], train: bool, *, paired_degradations: bool = False, stress_only: bool = False) -> None:
        self.records = records
        self.train = train
        self.paired_degradations = paired_degradations
        self.stress_only = stress_only
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.records) * (2 if self.train and self.paired_degradations else 1)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record_index = index // 2 if self.train and self.paired_degradations else index
        record = self.records[record_index]
        with Image.open(record.path) as image:
            degraded_pair = self.train and self.paired_degradations and index % 2 == 1
            seed = SEED + self.epoch * len(self) + index if degraded_pair or self.stress_only else None
            pixels = preprocess_image(image, augment_seed=seed)
        return pixels, torch.tensor(float(record.label), dtype=torch.float32)


def family_balanced_sample_weights(dataset: ManifestDataset) -> torch.Tensor:
    """Balance classes 50/50 and AI records equally across generator families."""
    counts: dict[tuple[int, str], int] = {}
    for record in dataset.records:
        family = record.source if record.label == 0 else str(record.generator_family)
        counts[(record.label, family)] = counts.get((record.label, family), 0) + 1
    ai_families = {family for label, family in counts if label == 1}
    if not ai_families or not any(label == 0 for label, _ in counts):
        raise ValueError("family-balanced sampling requires both labels and at least one AI family")
    copies = 2 if dataset.train and dataset.paired_degradations else 1
    weights: list[float] = []
    for index in range(len(dataset)):
        record = dataset.records[index // copies]
        family = record.source if record.label == 0 else str(record.generator_family)
        class_mass = 0.5 if record.label == 0 else 0.5 / len(ai_families)
        weights.append(class_mass / (counts[(record.label, family)] * copies))
    return torch.tensor(weights, dtype=torch.double)


def load_model(checkpoint: pathlib.Path) -> torch.nn.Module:
    model = timm.create_model("vit_small_patch16_384.augreg_in21k_ft_in1k", pretrained=False, num_classes=1)
    state = load_file(str(checkpoint))
    if all(key.startswith("vit.") for key in state):
        state = {key.removeprefix("vit."): value for key, value in state.items()}
    model.load_state_dict(state)
    return model


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, torch.Tensor, torch.Tensor]:
    model.eval()
    total_loss = 0.0
    criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    with torch.no_grad():
        for pixels, labels in loader:
            logits = model(pixels.to(device)).squeeze(1)
            total_loss += float(criterion(logits, labels.to(device)))
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    return total_loss / len(loader.dataset), torch.cat(all_logits), torch.cat(all_labels)  # type: ignore[arg-type]


def assert_no_protected_training_sources(records: list[Record]) -> None:
    violations = []
    for record in records:
        fields = " ".join(filter(None, (record.source, record.generator_family))).lower()
        if any(token in fields for token in PROTECTED_TRAIN_TOKENS):
            violations.append(f"{record.id}:{record.source}:{record.generator_family or ''}")
    if violations:
        raise ValueError(f"protected held-out/calibration source present in training rows: {violations[:5]}")


def passes_constraints(
    full: dict[str, float | int],
    stress: dict[str, float | int],
    baseline: dict[str, float | int],
    *,
    min_ai_recall: float,
    min_real_recall: float,
    max_fpr: float,
    max_full_ba_drop: float,
    max_full_real_recall_drop: float,
) -> bool:
    return bool(
        full["ai_recall"] >= min_ai_recall
        and stress["ai_recall"] >= min_ai_recall
        and full["real_recall"] >= min_real_recall
        and stress["real_recall"] >= min_real_recall
        and full["false_positive_rate"] <= max_fpr
        and stress["false_positive_rate"] <= max_fpr
        and full["balanced_accuracy"] >= baseline["balanced_accuracy"] - max_full_ba_drop
        and full["real_recall"] >= baseline["real_recall"] - max_full_real_recall_drop
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=pathlib.Path)
    parser.add_argument("--checkpoint", type=pathlib.Path, default=BASE_CHECKPOINT)
    parser.add_argument("--output", type=pathlib.Path, default=ROOT / "models" / "detector-finetuned.safetensors")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    adaptation = parser.add_mutually_exclusive_group()
    adaptation.add_argument("--freeze-backbone", action="store_true", help="train only the classification head (useful for smoke tests)")
    adaptation.add_argument("--unfreeze-last-n-blocks", type=int, help="train the head, final norm, and last N ViT blocks")
    parser.add_argument("--backbone-learning-rate", type=float, help="LR for partially unfrozen blocks/norm (default: 0.1x --learning-rate)")
    parser.add_argument("--no-paired-degradations", action="store_true")
    parser.add_argument("--family-balanced-sampling", action="store_true")
    parser.add_argument("--min-ai-recall", type=float, default=0.75)
    parser.add_argument("--min-real-recall", type=float, default=0.75)
    parser.add_argument("--max-fpr", type=float, default=0.25)
    parser.add_argument("--max-full-ba-drop", type=float, default=0.01)
    parser.add_argument("--max-full-real-recall-drop", type=float, default=0.02)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    for name in ("min_ai_recall", "min_real_recall", "max_fpr", "max_full_ba_drop", "max_full_real_recall_drop"):
        value = getattr(args, name)
        if not 0 <= value <= 1:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1]")

    if args.unfreeze_last_n_blocks is not None and args.unfreeze_last_n_blocks < 1:
        parser.error("--unfreeze-last-n-blocks must be at least 1")
    if args.backbone_learning_rate is not None and args.unfreeze_last_n_blocks is None:
        parser.error("--backbone-learning-rate requires --unfreeze-last-n-blocks")
    if args.learning_rate <= 0 or (args.backbone_learning_rate is not None and args.backbone_learning_rate <= 0):
        parser.error("learning rates must be positive")

    configure_determinism()
    records = read_manifest(args.manifest)
    train_records = [record for record in records if record.split == "train"]
    validation_records = [record for record in records if record.split == "calibration"]
    if not train_records or not validation_records:
        raise SystemExit("manifest must contain non-empty train and calibration splits")
    assert_no_protected_training_sources(train_records)

    device = torch.device(args.device)
    model = load_model(args.checkpoint).to(device)
    adaptation_report: dict[str, object] = {
        "mode": "full",
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    if args.freeze_backbone:
        adaptation_report = {"mode": "head_only", **configure_partial_backbone(model, 0)}
    elif args.unfreeze_last_n_blocks is not None:
        try:
            adaptation_report = {
                "mode": "partial_backbone",
                **configure_partial_backbone(model, args.unfreeze_last_n_blocks),
            }
        except ValueError as error:
            parser.error(str(error))
    train_data = ManifestDataset(train_records, train=True, paired_degradations=not args.no_paired_degradations)
    validation_data = ManifestDataset(validation_records, train=False)
    stress_data = ManifestDataset(validation_records, train=False, stress_only=True)
    generator = torch.Generator().manual_seed(SEED)
    sampler = None
    if args.family_balanced_sampling:
        sampler = WeightedRandomSampler(
            family_balanced_sample_weights(train_data),
            num_samples=len(train_data), replacement=True, generator=generator,
        )
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=sampler is None,
        sampler=sampler, generator=generator, num_workers=0,
    )
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    stress_loader = DataLoader(stress_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    backbone_learning_rate = args.backbone_learning_rate
    if backbone_learning_rate is None:
        backbone_learning_rate = args.learning_rate * (0.1 if args.unfreeze_last_n_blocks is not None else 1.0)
    parameter_groups = optimizer_parameter_groups(
        model,
        head_learning_rate=args.learning_rate,
        backbone_learning_rate=backbone_learning_rate,
        weight_decay=args.weight_decay,
    )
    optimizer = torch.optim.AdamW(parameter_groups)
    adaptation_report["head_learning_rate"] = args.learning_rate
    adaptation_report["backbone_learning_rate"] = backbone_learning_rate
    print(json.dumps({"training_adaptation": adaptation_report}, sort_keys=True))
    criterion = torch.nn.BCEWithLogitsLoss()

    _, baseline_logits, baseline_labels = evaluate(model, validation_loader, device)
    baseline_metrics = fixed_threshold_metrics(baseline_labels.numpy(), baseline_logits.numpy(), threshold=0.65)
    best_selection: tuple[float, float, float] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_report: dict[str, object] | None = None
    for epoch in range(args.epochs):
        train_data.set_epoch(epoch)
        model.train()
        for pixels, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(pixels.to(device)).squeeze(1)
            loss = criterion(logits, labels.to(device))
            loss.backward()
            optimizer.step()
        validation_loss, full_logits, full_labels = evaluate(model, validation_loader, device)
        _, stress_logits, stress_labels = evaluate(model, stress_loader, device)
        full_metrics = fixed_threshold_metrics(full_labels.numpy(), full_logits.numpy(), threshold=0.65)
        stress_metrics = fixed_threshold_metrics(stress_labels.numpy(), stress_logits.numpy(), threshold=0.65)
        eligible = passes_constraints(
            full_metrics, stress_metrics, baseline_metrics,
            min_ai_recall=args.min_ai_recall, min_real_recall=args.min_real_recall,
            max_fpr=args.max_fpr, max_full_ba_drop=args.max_full_ba_drop,
            max_full_real_recall_drop=args.max_full_real_recall_drop,
        )
        report = {"epoch": epoch + 1, "calibration_bce": validation_loss, "full": full_metrics, "thumbnail_stress": stress_metrics, "eligible": eligible}
        print(json.dumps(report))
        selection = (
            min(float(full_metrics["balanced_accuracy"]), float(stress_metrics["balanced_accuracy"])),
            (float(full_metrics["balanced_accuracy"]) + float(stress_metrics["balanced_accuracy"])) / 2,
            -validation_loss,
        )
        if eligible and (best_selection is None or selection > best_selection):
            best_selection = selection
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_report = report

    if best_state is None or best_report is None:
        raise SystemExit("no checkpoint satisfied the fixed-0.65 full-resolution/thumbnail recall and FPR constraints")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(best_state, str(args.output), metadata={
        "base_checkpoint_sha256": sha256(args.checkpoint),
        "manifest_sha256": sha256(args.manifest),
        "seed": str(SEED),
        "adaptation": json.dumps(adaptation_report, sort_keys=True),
        "selection_report": json.dumps(best_report, sort_keys=True),
    })
    run = {
        "checkpoint": str(args.output),
        "checkpoint_sha256": sha256(args.output),
        "base_checkpoint_sha256": sha256(args.checkpoint),
        "manifest_sha256": sha256(args.manifest),
        "seed": SEED,
        "train_sources": sorted({record.source for record in train_records}),
        "calibration_sources": sorted({record.source for record in validation_records}),
        "baseline_full_metrics": baseline_metrics,
        "selected": best_report,
        "adaptation": adaptation_report,
        "hyperparameters": {key: getattr(args, key) for key in (
            "epochs", "batch_size", "learning_rate", "backbone_learning_rate", "weight_decay", "freeze_backbone", "unfreeze_last_n_blocks", "no_paired_degradations",
            "family_balanced_sampling",
            "min_ai_recall", "min_real_recall", "max_fpr", "max_full_ba_drop", "max_full_real_recall_drop",
        )},
    }
    args.output.with_suffix(".run.json").write_text(json.dumps(run, indent=2) + "\n")
    print(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
