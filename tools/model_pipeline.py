"""Shared, dependency-light primitives for detector training and evaluation."""

from __future__ import annotations

import hashlib
import io
import json
import pathlib
import random
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch

IMAGE_SIZE = 384
RESIZE_SIZE = 440
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
REQUIRED_FIELDS = {"id", "baseId", "path", "sha256", "label", "source", "split", "license", "originUrl"}
VALID_SPLITS = {"train", "calibration", "test"}


@dataclass(frozen=True)
class Record:
    id: str
    base_id: str
    path: pathlib.Path
    label: int
    source: str
    generator_family: str | None
    split: str
    license: str
    origin_url: str
    expected_sha256: str
    content_group: str | None = None
    transform: str | None = None


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: pathlib.Path, *, require_files: bool = True) -> list[Record]:
    records: list[Record] = []
    manifest_root = path.resolve().parent
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            missing = REQUIRED_FIELDS - raw.keys()
            if missing:
                raise ValueError(f"{path}:{line_number}: missing fields {sorted(missing)}")
            image_path = pathlib.Path(raw["path"])
            if not image_path.is_absolute():
                image_path = manifest_root / image_path
            record = Record(
                id=str(raw["id"]),
                base_id=str(raw["baseId"]),
                path=image_path.resolve(),
                label=int(raw["label"]),
                source=str(raw["source"]),
                generator_family=str(raw["generatorFamily"]) if raw.get("generatorFamily") else None,
                split=str(raw["split"]),
                license=str(raw["license"]),
                origin_url=str(raw["originUrl"]),
                expected_sha256=str(raw["sha256"]),
                content_group=str(raw["contentGroup"]) if raw.get("contentGroup") else None,
                transform=str(raw["transform"]) if raw.get("transform") else None,
            )
            if record.label not in (0, 1):
                raise ValueError(f"{path}:{line_number}: label must be 0 or 1")
            if record.split not in VALID_SPLITS:
                raise ValueError(f"{path}:{line_number}: invalid split {record.split!r}")
            if record.label == 1 and not record.generator_family:
                raise ValueError(f"{path}:{line_number}: AI rows require generatorFamily")
            if require_files:
                if not record.path.is_file():
                    raise ValueError(f"{path}:{line_number}: missing image {record.path}")
                actual_sha256 = sha256(record.path)
                if actual_sha256 != record.expected_sha256:
                    raise ValueError(f"{path}:{line_number}: image checksum mismatch for {record.path}")
            records.append(record)
    validate_source_separation(records)
    if not records:
        raise ValueError(f"{path}: manifest is empty")
    return records


def validate_source_separation(records: Iterable[Record]) -> None:
    """Reject leakage by both base image lineage and dataset/generator source."""
    base_splits: dict[str, set[str]] = {}
    source_splits: dict[tuple[int, str], set[str]] = {}
    ids: set[str] = set()
    for record in records:
        if record.id in ids:
            raise ValueError(f"duplicate record id {record.id!r}")
        ids.add(record.id)
        base_splits.setdefault(record.base_id, set()).add(record.split)
        held_out_source = record.source if record.label == 0 else record.generator_family
        source_splits.setdefault((record.label, str(held_out_source)), set()).add(record.split)
    leaking_bases = sorted(key for key, splits in base_splits.items() if len(splits) > 1)
    leaking_sources = sorted(key for key, splits in source_splits.items() if len(splits) > 1)
    if leaking_bases:
        raise ValueError(f"base image lineage crosses splits: {leaking_bases[:5]}")
    if leaking_sources:
        raise ValueError(f"label/source groups cross splits: {leaking_sources[:5]}")


def _resize_short_side(image: Image.Image, size: int = RESIZE_SIZE) -> Image.Image:
    width, height = image.size
    scale = size / min(width, height)
    target = (round(width * scale), round(height * scale))
    return image.resize(target, Image.Resampling.BICUBIC)


def _center_crop(image: Image.Image, size: int = IMAGE_SIZE) -> Image.Image:
    left = max(0, (image.width - size) // 2)
    top = max(0, (image.height - size) // 2)
    return image.crop((left, top, left + size, top + size))


def deterministic_degradation(image: Image.Image, seed: int) -> Image.Image:
    """Web-realistic augmentation; deterministic for an item/epoch seed."""
    rng = random.Random(seed)
    # Force a realistic thumbnail path for the paired stress example. Retain
    # aspect ratio, then upscale so the model sees the information actually
    # available to a browser displaying a small/recompressed source image.
    short_target = rng.choice((112, 160, 192, 256, 320))
    scale = short_target / min(image.size)
    thumbnail_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    image = image.resize(thumbnail_size, Image.Resampling.LANCZOS)
    if rng.random() < 0.85:
        quality = rng.choice((30, 40, 50, 65, 75, 85, 95))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=False)
        buffer.seek(0)
        image = Image.open(buffer).convert("RGB")
    if rng.random() < 0.25:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.1, 1.5)))
    if rng.random() < 0.25:
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.8, 1.2))
    if rng.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return image


def preprocess_image(image: Image.Image, *, augment_seed: int | None = None) -> torch.Tensor:
    image = image.convert("RGB")
    if augment_seed is not None:
        image = deterministic_degradation(image, augment_seed)
    image = _center_crop(_resize_short_side(image))
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    pixels = (pixels - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.transpose(pixels, (2, 0, 1)).copy())


def sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.clip(logits, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-logits))


def calibrated_scores(logits: np.ndarray, temperature: float, bias: float) -> np.ndarray:
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    return sigmoid((logits + bias) / temperature)


def balanced_accuracy(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.65) -> float:
    labels = labels.astype(np.int64)
    predictions = (scores >= threshold).astype(np.int64)
    recalls = []
    for label in (0, 1):
        mask = labels == label
        if not np.any(mask):
            raise ValueError(f"balanced accuracy requires label {label}")
        recalls.append(float(np.mean(predictions[mask] == label)))
    return sum(recalls) / 2.0


def fixed_threshold_metrics(labels: np.ndarray, logits: np.ndarray, threshold: float = 0.65) -> dict[str, float | int]:
    labels = labels.astype(np.int64)
    scores = sigmoid(logits.astype(np.float64))
    predictions = (scores >= threshold).astype(np.int64)
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    if tp + fn == 0 or tn + fp == 0:
        raise ValueError("fixed-threshold metrics require both labels")
    ai_recall = tp / (tp + fn)
    real_recall = tn / (tn + fp)
    return {
        "threshold": threshold,
        "balanced_accuracy": (ai_recall + real_recall) / 2,
        "ai_recall": ai_recall,
        "real_recall": real_recall,
        "false_positive_rate": fp / (fp + tn),
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
    }
