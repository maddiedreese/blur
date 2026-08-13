#!/usr/bin/env python3
"""Isolated scorer for the pinned xRayon ConvNeXtV2 candidate.

This is evaluation-only. It does not export, copy, or integrate the model into
the extension. The preprocessing reproduces the upstream test transform.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from PIL import Image
import timm
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


MODEL_REVISION = "3fecc2d10e3812bc6d26de0c0fe359f219101e2c"
CHECKPOINT_SHA256 = "37f31776a241b575dc034ddded7afd12014ba453ac07cbe3725f808787717f0e"


class ManifestImages(Dataset):
    def __init__(self, manifest: Path):
        self.root = manifest.parent
        self.rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        self.transform = transforms.Compose([
            transforms.Resize(288),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = self.root / row["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {row['id']}")
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return index, tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if checkpoint_hash != CHECKPOINT_SHA256:
        raise ValueError(f"unexpected checkpoint SHA-256: {checkpoint_hash}")
    if args.output.exists():
        raise FileExistsError(args.output)

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = timm.create_model("convnextv2_base", pretrained=False, num_classes=2)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()

    dataset = ManifestImages(args.manifest)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    scored = [None] * len(dataset)
    started = time.perf_counter()
    with torch.inference_mode():
        for indices, images in loader:
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)[:, 1]
            for index, logit, probability in zip(indices.tolist(), logits[:, 1].tolist(), probabilities.tolist()):
                row = dict(dataset.rows[index])
                row.update({
                    "score": probability,
                    "rawFakeLogit": logit,
                    "candidate": "xRayon/convnext-ai-images-detector",
                    "candidateRevision": MODEL_REVISION,
                    "checkpointSha256": checkpoint_hash,
                })
                scored[index] = row
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in scored))
    elapsed = time.perf_counter() - started
    print(json.dumps({"rows": len(scored), "seconds": elapsed, "rowsPerSecond": len(scored) / elapsed}, indent=2))


if __name__ == "__main__":
    main()
