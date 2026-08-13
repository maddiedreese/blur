#!/usr/bin/env python3
"""Isolated frozen-DINOv2 detector experiment.

This script deliberately consumes only ``split=train`` from the recent-source
manifest.  Protected calibration and commercial evaluation manifests are read
only after the head has been selected on a deterministic, source-stratified
holdout from that training split.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import pathlib
import random
from dataclasses import dataclass

import numpy as np
from PIL import Image
import torch
from torch import nn
import timm


MODEL_ID = "vit_small_patch14_dinov2.lvd142m"
HF_MODEL = "timm/vit_small_patch14_dinov2.lvd142m"
HF_REVISION = "4610ca143709d58a633b6397a74412c2c3842454"
IMAGE_SIZE = 224
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
THRESHOLD = 0.65


@dataclass(frozen=True)
class Row:
    id: str
    base_id: str
    path: pathlib.Path
    label: int
    source: str
    family: str
    split: str
    transform: str | None


def read_rows(path: pathlib.Path) -> list[Row]:
    root = path.resolve().parent
    result = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        image = pathlib.Path(value["path"])
        if not image.is_absolute():
            image = root / image
        result.append(Row(
            id=str(value["id"]), base_id=str(value["baseId"]), path=image.resolve(),
            label=int(value["label"]), source=str(value["source"]),
            family=str(value.get("generatorFamily") or value["source"]),
            split=str(value["split"]), transform=value.get("transform"),
        ))
    return result


def partition(rows: list[Row]) -> tuple[list[Row], list[Row]]:
    groups: dict[tuple[int, str], list[Row]] = {}
    for row in rows:
        groups.setdefault((row.label, row.family), []).append(row)
    train, validation = [], []
    for key, values in sorted(groups.items()):
        values.sort(key=lambda r: hashlib.sha256(r.base_id.encode()).hexdigest())
        cut = max(1, round(len(values) * 0.2))
        validation.extend(values[:cut])
        train.extend(values[cut:])
    return train, validation


def degrade(image: Image.Image, variant: str) -> Image.Image:
    if variant == "full":
        return image
    size, quality = ((256, 75) if variant == "thumb256-jpeg75" else (192, 50))
    scale = size / min(image.size)
    resized = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    resized.save(buffer, "JPEG", quality=quality, optimize=False)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def pixels(path: pathlib.Path, variant: str = "full") -> torch.Tensor:
    with Image.open(path) as source:
        image = degrade(source.convert("RGB"), variant)
    scale = IMAGE_SIZE / min(image.size)
    image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.BICUBIC)
    left, top = (image.width - IMAGE_SIZE) // 2, (image.height - IMAGE_SIZE) // 2
    image = image.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))
    value = np.asarray(image, dtype=np.float32) / 255.0
    value = (value - MEAN) / STD
    return torch.from_numpy(value.transpose(2, 0, 1).copy())


def embeddings(backbone: nn.Module, rows: list[Row], variants: tuple[str, ...], batch: int) -> tuple[torch.Tensor, torch.Tensor, list[tuple[Row, str]]]:
    items = [(row, variant) for row in rows for variant in variants]
    output = []
    with torch.inference_mode():
        for start in range(0, len(items), batch):
            tensor = torch.stack([pixels(row.path, variant) for row, variant in items[start:start + batch]])
            feature = backbone.forward_head(backbone.forward_features(tensor), pre_logits=True)
            output.append(feature.cpu())
            print(f"features {min(start + batch, len(items))}/{len(items)}", flush=True)
    return torch.cat(output), torch.tensor([row.label for row, _ in items], dtype=torch.float32), items


class Detector(nn.Module):
    def __init__(self, backbone: nn.Module, mean: torch.Tensor, scale: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
        super().__init__()
        self.backbone = backbone
        self.register_buffer("feature_mean", mean)
        self.register_buffer("feature_scale", scale)
        self.head = nn.Linear(weight.numel(), 1)
        self.head.weight.data.copy_(weight.reshape(1, -1))
        self.head.bias.data.copy_(bias.reshape(1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        feature = self.backbone.forward_head(self.backbone.forward_features(value), pre_logits=True)
        return self.head((feature - self.feature_mean) / self.feature_scale)


def fit_head(x: torch.Tensor, y: torch.Tensor, items: list[tuple[Row, str]], xv: torch.Tensor, yv: torch.Tensor) -> tuple[nn.Linear, torch.Tensor, torch.Tensor, dict]:
    mean, scale = x.mean(0), x.std(0).clamp_min(1e-5)
    x, xv = (x - mean) / scale, (xv - mean) / scale
    group_counts: dict[tuple[int, str], int] = {}
    for row, _ in items:
        group_counts[(row.label, row.family)] = group_counts.get((row.label, row.family), 0) + 1
    ai_groups = len({row.family for row, _ in items if row.label == 1})
    weights = []
    for row, _ in items:
        total = 0.5 if row.label == 0 else 0.5 / ai_groups
        weights.append(total / group_counts[(row.label, row.family)])
    sample_weight = torch.tensor(weights) * len(weights)
    best = None
    for decay in (0.0, 1e-5, 1e-4, 1e-3, 1e-2):
        torch.manual_seed(17)
        head = nn.Linear(x.shape[1], 1)
        optimizer = torch.optim.AdamW(head.parameters(), lr=0.02, weight_decay=decay)
        for _ in range(700):
            logits = head(x).squeeze(1)
            loss = (nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none") * sample_weight).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        metrics = measures(yv.numpy(), torch.sigmoid(head(xv).squeeze(1)).detach().numpy())
        rank = (metrics["real_recall"] >= 0.90, metrics["balanced_accuracy"], metrics["ai_recall"])
        if best is None or rank > best[0]:
            best = (rank, head, metrics, decay)
    assert best is not None
    return best[1], mean, scale, {"weight_decay": best[3], "validation": best[2]}


def measures(labels: np.ndarray, scores: np.ndarray) -> dict:
    pred = scores >= THRESHOLD
    tp = int(np.sum((labels == 1) & pred)); fn = int(np.sum((labels == 1) & ~pred))
    tn = int(np.sum((labels == 0) & ~pred)); fp = int(np.sum((labels == 0) & pred))
    ai = tp / (tp + fn); real = tn / (tn + fp)
    return {"threshold": THRESHOLD, "tp": tp, "fn": fn, "tn": tn, "fp": fp,
            "ai_recall": ai, "real_recall": real, "false_positive_rate": fp / (fp + tn),
            "balanced_accuracy": (ai + real) / 2}


def evaluate(head: nn.Linear, mean: torch.Tensor, scale: torch.Tensor, x: torch.Tensor, y: torch.Tensor, items: list[tuple[Row, str]]) -> dict:
    with torch.inference_mode():
        scores = torch.sigmoid(head((x - mean) / scale).squeeze(1)).numpy()
    result = {"all": measures(y.numpy(), scores)}
    for transform in sorted({(row.transform or variant) for row, variant in items}):
        mask = np.asarray([(row.transform or variant) == transform for row, variant in items])
        result[transform] = measures(y.numpy()[mask], scores[mask])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("recent_manifest", type=pathlib.Path)
    parser.add_argument("commercial_full", type=pathlib.Path)
    parser.add_argument("commercial_stress", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    recent = [r for r in read_rows(args.recent_manifest) if r.split == "train"]
    train, validation = partition(recent)
    assert all(r.split == "train" for r in train + validation)
    random.Random(17).shuffle(train); random.Random(23).shuffle(validation)
    backbone = timm.create_model(MODEL_ID, pretrained=True, num_classes=0, img_size=IMAGE_SIZE).eval()
    for parameter in backbone.parameters(): parameter.requires_grad_(False)
    variants = ("full", "thumb256-jpeg75", "thumb192-jpeg50")
    x, y, train_items = embeddings(backbone, train, variants, args.batch)
    xv, yv, _ = embeddings(backbone, validation, variants, args.batch)
    head, mean, scale, selection = fit_head(x, y, train_items, xv, yv)
    # Protected commercial data is first opened here, after model selection.
    full_rows = [r for r in read_rows(args.commercial_full) if r.split == "test"]
    stress_rows = [r for r in read_rows(args.commercial_stress) if r.split == "test"]
    xf, yf, full_items = embeddings(backbone, full_rows, ("full",), args.batch)
    xs, ys, stress_items = embeddings(backbone, stress_rows, ("full",), args.batch)
    detector = Detector(backbone, mean, scale, head.weight.detach().flatten(), head.bias.detach()).eval()
    state = args.output / "dinov2-frozen-detector.pt"
    torch.save(detector.state_dict(), state)
    onnx_path = args.output / "dinov2-frozen-detector.onnx"
    parity = {"exported": False}
    try:
        torch.onnx.export(detector, torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE), onnx_path,
                          input_names=["pixel_values"], output_names=["logits"], opset_version=18,
                          dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}})
        import onnxruntime as ort
        sample = torch.stack([pixels(r.path) for r in full_rows[:3]])
        with torch.inference_mode(): expected = detector(sample).numpy()
        actual = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"]).run(None, {"pixel_values": sample.numpy()})[0]
        parity = {"exported": True, "max_abs_error": float(np.max(np.abs(expected - actual)))}
    except Exception as error:
        parity = {"exported": False, "error": repr(error)}
    report = {
        "model": MODEL_ID, "upstream": HF_MODEL, "upstream_revision": HF_REVISION,
        "license": "Apache-2.0 (per official timm Hugging Face model card)", "image_size": IMAGE_SIZE,
        "training": {"train_base_rows": len(train), "validation_base_rows": len(validation),
                     "paired_variants": list(variants), **selection},
        "commercial_full": evaluate(head, mean, scale, xf, yf, full_items),
        "commercial_stress": evaluate(head, mean, scale, xs, ys, stress_items),
        "artifacts": {"state_bytes": state.stat().st_size, "onnx_bytes": onnx_path.stat().st_size if onnx_path.exists() else None,
                      "onnx_parity": parity},
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
