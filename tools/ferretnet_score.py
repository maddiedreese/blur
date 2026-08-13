#!/usr/bin/env python3
"""Isolated scorer for the pinned FerretNet-B candidate.

Evaluation only: the checkpoint and upstream source stay under ignored
``artifacts/`` and are never copied into the browser extension.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


UPSTREAM_REVISION = "e92796c1a2fb07ccc57ecd7e718e6dce067be5fa"
CHECKPOINT_SHA256 = "fe755d78370bb6547070329553572405b4ecebd23382c9a6cbb11c4ab85a82c2"
NORMALIZE_MEAN = (0.48145466, 0.4578275, 0.40821073)
NORMALIZE_STD = (0.26862954, 0.26130258, 0.27577711)


class ManifestImages(Dataset):
    def __init__(self, manifest: Path):
        self.root = manifest.parent
        self.rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        self.transform = transforms.Compose([
            transforms.CenterCrop((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
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


def load_model(upstream: Path, checkpoint: Path):
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
        raise ValueError("unexpected FerretNet-B checkpoint SHA-256")

    sys.path.insert(0, str(upstream.resolve()))
    from src.model.ferretnet import Ferret  # pylint: disable=import-outside-toplevel
    from src.model.lpd import get_lpd_dict  # pylint: disable=import-outside-toplevel

    model = Ferret(
        in_channels=3,
        num_classes=1,
        dim=96,
        depths=[2, 2],
        lpd_func="median",
        window_size=3,
        lpd_dict=get_lpd_dict(),
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(args.output)

    model = load_model(args.upstream, args.checkpoint)
    dataset = ManifestImages(args.manifest)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    scored = [None] * len(dataset)
    started = time.perf_counter()
    with torch.inference_mode():
        for indices, images in loader:
            logits = model(images).flatten()
            probabilities = torch.sigmoid(logits)
            for index, logit, probability in zip(indices.tolist(), logits.tolist(), probabilities.tolist()):
                row = dict(dataset.rows[index])
                row.update({
                    "score": probability,
                    "rawLogit": logit,
                    "candidate": "FerretNet-B-Median-3",
                    "candidateRevision": UPSTREAM_REVISION,
                    "checkpointSha256": CHECKPOINT_SHA256,
                })
                scored[index] = row

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in scored))
    elapsed = time.perf_counter() - started
    print(json.dumps({"rows": len(scored), "seconds": elapsed, "rowsPerSecond": len(scored) / elapsed}, indent=2))


if __name__ == "__main__":
    main()
