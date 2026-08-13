#!/usr/bin/env python3
"""Safely inventory the pinned official GAPL checkpoint without importing its code."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

import torch

HF_REVISION = "ea0341c1ca59862508a1a621fb8072c274bc31dd"
EXPECTED_SHA256 = "ffbcb5eb526f0df0fd197d7266bdd0325b66813e95010f1285685acf2d267235"


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=pathlib.Path)
    args = parser.parse_args()
    actual_sha256 = sha256(args.checkpoint)
    if actual_sha256 != EXPECTED_SHA256:
        raise SystemExit(f"checkpoint checksum mismatch: {actual_sha256}")

    # weights_only blocks arbitrary pickle globals; mmap avoids duplicating the
    # 1.2 GB checkpoint in process memory merely to inspect its tensors.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    model = checkpoint["model"]
    groups: dict[str, dict[str, int]] = {}
    for name, tensor in model.items():
        if ".lora_" in name:
            group = "lora_adapters"
        elif ".base_layer." in name:
            group = "peft_wrapped_base_layers"
        elif name.startswith("feature_extractor."):
            group = "clip_vision_backbone"
        else:
            group = name.split(".", 1)[0]
        entry = groups.setdefault(group, {"parameters": 0, "bytes": 0, "tensors": 0})
        entry["parameters"] += tensor.numel()
        entry["bytes"] += tensor.numel() * tensor.element_size()
        entry["tensors"] += 1

    total_parameters = sum(tensor.numel() for tensor in model.values())
    total_bytes = sum(tensor.numel() * tensor.element_size() for tensor in model.values())
    prototype = checkpoint["prototype"]
    report = {
        "source": f"https://huggingface.co/AbyssLumine/GAPL/tree/{HF_REVISION}",
        "sha256": actual_sha256,
        "file_bytes": args.checkpoint.stat().st_size,
        "epoch": checkpoint["epoch"],
        "model_parameters": total_parameters,
        "model_tensor_bytes": total_bytes,
        "tensor_groups": groups,
        "prototype_shape": list(prototype.shape),
        "prototype_bytes": prototype.numel() * prototype.element_size(),
        "estimated_fp16_tensor_bytes": total_parameters * 2 + prototype.numel() * 2,
        "estimated_int8_weight_floor_bytes": total_parameters + prototype.numel(),
        "architecture": "OpenAI CLIP ViT-L/14 vision encoder + q/k/v LoRA + 1024-to-128 projection + 64x128 prototypes + 4-head cross-attention + linear head",
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
