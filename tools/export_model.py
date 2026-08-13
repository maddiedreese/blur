#!/usr/bin/env python3
"""Reproducibly download, verify, and export the pinned detector to ONNX."""

from __future__ import annotations

import hashlib
import json
import pathlib
import urllib.request

import torch
from safetensors.torch import load_file
import timm

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
REVISION = "6076002bf0d9dd37537f965ee2f06f826c333b61"
SOURCE_URL = f"https://huggingface.co/OwensLab/commfor-model-384/resolve/{REVISION}/model.safetensors"
SOURCE_SHA256 = "b89f36275f3bf5e2b040eee36597a8f19db051bff9a473a9cf7b2466284fb387"


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    source = MODEL_DIR / "commfor-model-384.safetensors"
    if not source.exists():
        print(f"downloading pinned checkpoint to {source}")
        urllib.request.urlretrieve(SOURCE_URL, source)
    actual = sha256(source)
    if actual != SOURCE_SHA256:
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

    target = MODEL_DIR / "detector.onnx"
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
        "source": SOURCE_URL,
        "source_sha256": actual,
        "onnx_sha256": sha256(target),
        "input": [1, 3, 384, 384],
        "output": "fake_logit",
        "opset": 18,
    }
    (MODEL_DIR / "model.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
