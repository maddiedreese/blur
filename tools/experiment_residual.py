#!/usr/bin/env python3
"""Evaluate a tiny residual/frequency branch; never mutates production artifacts."""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
from PIL import Image
import torch

from model_pipeline import Record, balanced_accuracy, read_manifest, sigmoid


def residual_features(record: Record) -> np.ndarray:
    """Return 24 deterministic, browser-reproducible scalar features."""
    with Image.open(record.path) as image:
        rgb = np.asarray(image.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    gray = rgb @ np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
    residual = gray[1:-1, 1:-1] * 4 - gray[:-2, 1:-1] - gray[2:, 1:-1] - gray[1:-1, :-2] - gray[1:-1, 2:]
    abs_residual = np.abs(residual)
    residual_stats = [abs_residual.mean(), residual.std(), np.mean(residual**2), np.mean(residual**4)]

    spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray))))
    yy, xx = np.indices(spectrum.shape)
    radius = np.sqrt((xx - 127.5) ** 2 + (yy - 127.5) ** 2) / np.sqrt(2 * 127.5**2)
    radial = []
    for low, high in zip(np.linspace(0, 1, 13)[:-1], np.linspace(0, 1, 13)[1:]):
        values = spectrum[(radius >= low) & (radius < high)]
        radial.append(float(values.mean()) if values.size else 0.0)
    radial = (np.asarray(radial) - np.mean(radial)) / (np.std(radial) + 1e-6)

    channel_stats = []
    for channel in range(3):
        horizontal = np.diff(rgb[:, :, channel], axis=1)
        vertical = np.diff(rgb[:, :, channel], axis=0)
        channel_stats.extend([np.mean(np.abs(horizontal)), np.mean(np.abs(vertical))])
    boundaries = np.arange(8, 256, 8)
    vertical_boundary = np.mean(np.abs(gray[:, boundaries] - gray[:, boundaries - 1]))
    horizontal_boundary = np.mean(np.abs(gray[boundaries, :] - gray[boundaries - 1, :]))
    return np.asarray(residual_stats + radial.tolist() + channel_stats + [vertical_boundary, horizontal_boundary], dtype=np.float32)


def load_logits(path: pathlib.Path) -> dict[str, float]:
    values: dict[str, float] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                values[str(item["id"])] = float(item["logit"])
    return values


def fit_logistic(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, float]:
    x = torch.tensor(features, dtype=torch.float64)
    y = torch.tensor(labels, dtype=torch.float64)
    weights = torch.nn.Parameter(torch.zeros(x.shape[1], dtype=torch.float64))
    bias = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.LBFGS([weights, bias], max_iter=100, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(x @ weights + bias, y)
        loss = loss + 1e-3 * torch.sum(weights**2)
        loss.backward()
        return loss

    optimizer.step(closure)
    return weights.detach().numpy(), float(bias.detach())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=pathlib.Path)
    parser.add_argument("base_scores", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    records = read_manifest(args.manifest)
    logits_by_id = load_logits(args.base_scores)
    if any(record.id not in logits_by_id for record in records):
        raise SystemExit("base score file does not cover every manifest id")
    features = np.stack([residual_features(record) for record in records])
    labels = np.asarray([record.label for record in records], dtype=np.int64)
    logits = np.asarray([logits_by_id[record.id] for record in records], dtype=np.float32)
    splits = np.asarray([record.split for record in records])
    train = splits == "train"
    validation = splits == "calibration"
    if not np.any(train) or not np.any(validation):
        raise SystemExit("experiment requires train and calibration splits")
    mean = features[train].mean(axis=0)
    std = features[train].std(axis=0) + 1e-6
    normalized = (features - mean) / std
    weights, bias = fit_logistic(normalized[train], labels[train])
    residual_logits = normalized @ weights + bias

    best = None
    for alpha in np.linspace(0, 1, 21):
        fused = sigmoid((1 - alpha) * logits[validation] + alpha * residual_logits[validation])
        accuracy = balanced_accuracy(labels[validation], fused)
        candidate = (accuracy, -float(alpha), float(alpha))
        if best is None or candidate > best:
            best = candidate
    assert best is not None
    alpha = best[2]
    base_validation = balanced_accuracy(labels[validation], sigmoid(logits[validation]))
    lift = best[0] - base_validation
    test_available = np.any(splits == "test")
    result = {
        "feature_version": "residual-24-v1",
        "feature_count": 24,
        "fusion_alpha": alpha,
        "base_calibration_balanced_accuracy": base_validation,
        "fused_calibration_balanced_accuracy": best[0],
        "calibration_lift": lift,
        "recommend_integration": False,
        "integration_gate": "at least 100 calibration samples, calibration lift >= 0.02, and untouched-test lift >= 0.02",
        "normalization_mean": mean.tolist(),
        "normalization_std": std.tolist(),
        "weights": weights.tolist(),
        "bias": bias,
    }
    if test_available:
        test = splits == "test"
        test_base = balanced_accuracy(labels[test], sigmoid(logits[test]))
        test_fused = balanced_accuracy(
            labels[test], sigmoid((1 - alpha) * logits[test] + alpha * residual_logits[test])
        )
        result["test_base_balanced_accuracy"] = test_base
        result["test_fused_balanced_accuracy"] = test_fused
        result["test_lift"] = test_fused - test_base
        result["recommend_integration"] = bool(np.sum(validation) >= 100 and lift >= 0.02 and test_fused - test_base >= 0.02)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
