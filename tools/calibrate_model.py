#!/usr/bin/env python3
"""Fit temperature and bias on validation logits; test data is explicitly rejected."""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch

from model_pipeline import balanced_accuracy, calibrated_scores, sha256


def read_scores(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    labels: list[int] = []
    logits: list[float] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("split") != "calibration":
                raise ValueError(f"{path}:{line_number}: calibration accepts calibration rows only")
            labels.append(int(item["label"]))
            logits.append(float(item["logit"]))
    if set(labels) != {0, 1}:
        raise ValueError("calibration requires both labels")
    return np.asarray(labels, dtype=np.int64), np.asarray(logits, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scores", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--allow-small-smoke", action="store_true")
    parser.add_argument("--max-short-side", type=int)
    args = parser.parse_args()
    labels, logits = read_scores(args.scores)
    if len(labels) < 100 and not args.allow_small_smoke:
        raise SystemExit("calibration requires at least 100 calibration rows (or --allow-small-smoke for plumbing only)")
    logits_tensor = torch.tensor(logits, dtype=torch.float64)
    labels_tensor = torch.tensor(labels, dtype=torch.float64)
    log_temperature = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    bias = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.LBFGS([log_temperature, bias], max_iter=100, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        calibrated = (logits_tensor + bias) / log_temperature.exp()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(calibrated, labels_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_temperature.exp().detach())
    fitted_bias = float(bias.detach())
    scores = calibrated_scores(logits, temperature, fitted_bias)
    artifact = {
        "method": "temperature-plus-bias",
        "temperature": temperature,
        "bias": fitted_bias,
        "decision_threshold": 0.65,
        "validation_balanced_accuracy": balanced_accuracy(labels, scores),
        "validation_scores_sha256": sha256(args.scores),
        "sample_count": len(labels),
        "smoke_only": bool(args.allow_small_smoke),
        **({"max_short_side": args.max_short_side} if args.max_short_side else {}),
    }
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
