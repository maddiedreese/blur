# Rejected candidate: xRayon ConvNeXtV2-Base

This is an isolated audit, not an integration or an accuracy claim.

## Pinned provenance

- Repository: `xRayon/convnext-ai-images-detector`
- Revision: `3fecc2d10e3812bc6d26de0c0fe359f219101e2c`
- Phase-2 checkpoint SHA-256: `37f31776a241b575dc034ddded7afd12014ba453ac07cbe3725f808787717f0e`
- Phase-2 checkpoint size: 1,052,834,809 bytes
- License file: MIT, copyright 2026 Toufik Abbadi
- Architecture: timm `convnextv2_base`, two output classes, 87,694,850 parameters
- Test preprocessing: resize shorter side to 288, center crop 256, ImageNet normalization

The checkpoint contains the model plus optimizer, scheduler, and scaler state. The model tensors alone require 350,779,400 bytes (334.5 MiB) in FP32. An idealized all-int8 weight representation would still be approximately 83.6 MiB before quantization metadata and runtime assets. ConvNeXtV2-Base is reported at roughly 15.4 GMAC at 224px; spatial scaling gives an approximate 20.1 GMAC at this checkpoint's 256px input. This is substantially heavier than the current browser candidate.

The model card labels the repository MIT, but it lists eleven phase-1 datasets and two phase-2 datasets without recording each dataset's license or a complete sample-level provenance manifest. Redistribution and derivative-weight provenance therefore require further review despite the repository license.

## Published OOD caveat

The model card reports 90.40% “accuracy” on EvalGen. The pinned `test/test.py` actually calls `test_1_class` on an AI-only EvalGen directory and prints percent predicted fake. The real-negative dataset is commented out. This is fake-only recall, not two-class accuracy, balanced accuracy, specificity, or false-positive evidence.

## Protected full-resolution diagnostic

The isolated scorer reproduced the upstream preprocessing and loaded the checkpoint with `torch.load(..., weights_only=True)`. Every protected image was SHA-256 checked against its manifest. At the fixed `score >= 0.65` decision rule on 32 LAION real images and 32 Midjourney images:

| TP | TN | FP | FN | AI recall | Real recall | False-positive rate | Balanced accuracy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 17 | 15 | 0 | 100% | 53.125% | 46.875% | 76.5625% |

The score file is retained locally at `artifacts/convnext-xrayon/full-scores.jsonl` with SHA-256 `7c5a330dee99a7ba98605b6f4819616de631222fd17744a6c27d8da315f7bd3b`.

The stress run was stopped after the full-resolution false-positive rate decisively failed the release policy. No thumbnail metric is reported for this candidate.

## High-confidence sweep

Using this same small diagnostic set only, the lowest threshold allowing at most one of 32 real images to be called AI was just above `0.975399374961853`. It retained 6 of 32 AI images (18.75% AI recall). At `0.99` and every higher threshold tested, it retained zero AI images and zero real false positives.

All 32 protected Midjourney base images correspond to cases the current base detector missed after thumbnail degradation. ConvNeXt marks only 6 of their full-resolution originals above the one-false-positive cutoff. Because ConvNeXt thumbnail inference was not completed, this does not show that it can recover the actual thumbnail failures. A “very high confidence” gate at 0.99 recovers none even at full resolution.

## Decision

Rejected for integration. At the required threshold it would add fifteen false positives to only 32 protected real examples. As a high-confidence rescue signal it offers weak recall, no demonstrated thumbnail recovery, and prohibitive browser size/compute. Its weights and scores remain under ignored `artifacts/`; the release build does not reference them.
