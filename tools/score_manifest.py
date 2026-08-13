#!/usr/bin/env python3
"""Write raw detector logits for a manifest without calibrating or thresholding them."""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
from PIL import Image
import torch

from model_pipeline import calibrated_scores, preprocess_image, read_manifest
from train_detector import BASE_CHECKPOINT, load_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=pathlib.Path)
    parser.add_argument("--checkpoint", type=pathlib.Path, default=BASE_CHECKPOINT)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--split", choices=("train", "calibration", "test"))
    parser.add_argument("--calibration", type=pathlib.Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    records = read_manifest(args.manifest)
    if args.split:
        records = [record for record in records if record.split == args.split]
        if not records:
            raise SystemExit(f"manifest has no {args.split} rows")
    device = torch.device(args.device)
    model = load_model(args.checkpoint).to(device).eval()
    calibration = json.loads(args.calibration.read_text()) if args.calibration else None
    with args.output.open("w", encoding="utf-8") as output, torch.no_grad():
        for record in records:
            with Image.open(record.path) as image:
                short_side = min(image.size)
                pixels = preprocess_image(image).unsqueeze(0).to(device)
            logit = float(model(pixels).squeeze().cpu())
            raw_score = float(torch.sigmoid(torch.tensor(logit)))
            applies = bool(calibration and short_side <= calibration.get("max_short_side", 0))
            score = (float(calibrated_scores(
                np.asarray([logit]), float(calibration["temperature"]), float(calibration["bias"])
            )[0]) if applies else raw_score)
            output.write(json.dumps({
                "id": record.id,
                "baseId": record.base_id,
                "source": record.source,
                **({"generatorFamily": record.generator_family} if record.generator_family else {}),
                **({"contentGroup": record.content_group} if record.content_group else {}),
                **({"transform": record.transform} if record.transform else {}),
                "split": record.split,
                "label": record.label,
                "logit": logit,
                "rawScore": raw_score,
                "score": score,
                "shortSide": short_side,
                "calibrationApplied": applies,
            }, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
