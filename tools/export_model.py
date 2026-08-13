#!/usr/bin/env python3
"""Reproducibly download, verify, and export the pinned detector to ONNX."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

import torch
from safetensors.torch import load_file
import timm

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
REVISION = "6076002bf0d9dd37537f965ee2f06f826c333b61"
SOURCE_URL = f"https://huggingface.co/OwensLab/commfor-model-384/resolve/{REVISION}/model.safetensors"
SOURCE_SHA256 = "b89f36275f3bf5e2b040eee36597a8f19db051bff9a473a9cf7b2466284fb387"
PRODUCTION_CHECKPOINT = MODEL_DIR / "detector-thumbnail-head.safetensors"
PRODUCTION_SHA256 = "9cb5b56d44fff294e2f52c49bddea51d30b9d88fa29d4b4eaf4095753c9ceb36"


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, default=MODEL_DIR / "detector.onnx")
    parser.add_argument("--calibration", type=pathlib.Path)
    args = parser.parse_args()
    MODEL_DIR.mkdir(exist_ok=True)
    source = args.checkpoint or PRODUCTION_CHECKPOINT
    if not source.exists():
        raise SystemExit(f"checkpoint is missing: {source}")
    actual = sha256(source)
    if args.checkpoint is None and actual != PRODUCTION_SHA256:
        raise SystemExit(f"checkpoint checksum mismatch: {actual}")

    model = timm.create_model(
        "vit_small_patch16_384.augreg_in21k_ft_in1k",
        pretrained=False,
        num_classes=1,
    )
    state = load_file(str(source))
    # PyTorchModelHubMixin saves the enclosing module's `vit.*` keys.
    if all(key.startswith("vit.") for key in state):
        state = {key.removeprefix("vit."): value for key, value in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"state mismatch; missing={missing}, unexpected={unexpected}")
    model.eval()

    target = args.output
    target.parent.mkdir(parents=True, exist_ok=True)
    example = torch.zeros(1, 3, 384, 384, dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            model,
            example,
            target,
            input_names=["pixel_values"],
            output_names=["fake_logit"],
            opset_version=18,
            dynamo=False,
            do_constant_folding=True,
        )
    metadata = {
        "source": "repository production checkpoint" if args.checkpoint is None else str(source.resolve()),
        "source_sha256": actual,
        "base_source": SOURCE_URL,
        "base_source_sha256": SOURCE_SHA256,
        "onnx_sha256": sha256(target),
        "input": [1, 3, 384, 384],
        "output": "fake_logit",
        "opset": 18,
    }
    if args.checkpoint is None:
        metadata.update({
            "training_manifest_sha256": "d925a86940a80ab4d93320e2406368f1875a8fd7e0a2c2caade3a05c0c0ce4f5",
            "adaptation": "frozen ViT backbone; binary classifier head fine-tune",
        })
    if args.calibration:
        calibration = json.loads(args.calibration.read_text())
        if calibration.get("decision_threshold") != 0.65:
            raise SystemExit("calibration artifact must preserve decision_threshold=0.65")
        metadata["calibration"] = calibration
    metadata_target = MODEL_DIR / "model.json" if target == MODEL_DIR / "detector.onnx" else target.with_suffix(".json")
    metadata_target.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
