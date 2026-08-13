#!/usr/bin/env python3
"""Check graph validity and PyTorch/ONNX output parity on deterministic tensors."""

from __future__ import annotations

import json
import pathlib

import numpy as np
import onnx
import onnxruntime as ort
import torch
from safetensors.torch import load_file
import timm

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_torch_model() -> torch.nn.Module:
    model = timm.create_model("vit_small_patch16_384.augreg_in21k_ft_in1k", pretrained=False, num_classes=1)
    state = load_file(str(ROOT / "models/commfor-model-384.safetensors"))
    if all(key.startswith("vit.") for key in state):
        state = {key.removeprefix("vit."): value for key, value in state.items()}
    model.load_state_dict(state)
    return model.eval()


def main() -> None:
    model_path = ROOT / "models/detector.onnx"
    onnx.checker.check_model(onnx.load(model_path))
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    torch_model = load_torch_model()
    generator = np.random.default_rng(11997733)
    max_error = 0.0
    for sample in [np.zeros((1, 3, 384, 384), np.float32), generator.standard_normal((1, 3, 384, 384), dtype=np.float32)]:
        with torch.no_grad():
            expected = torch_model(torch.from_numpy(sample)).numpy()
        actual = session.run(None, {"pixel_values": sample})[0]
        max_error = max(max_error, float(np.max(np.abs(actual - expected))))
    if max_error > 1e-4:
        raise SystemExit(f"ONNX parity failure: max absolute error {max_error}")
    print(json.dumps({"valid": True, "max_absolute_error": max_error, "providers": session.get_providers()}, indent=2))


if __name__ == "__main__":
    main()
