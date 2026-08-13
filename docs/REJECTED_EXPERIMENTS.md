# Rejected diagnostic experiments

The following runs are retained locally for failure analysis only. They are too small and too narrow to support an accuracy claim, checkpoint choice, or release.

## Current checkpoint diagnostics

The 64-row full-resolution commercial sample contained only 32 real and 32 AI images. It produced 29 true positives, 32 true negatives, and 3 false negatives at the fixed threshold. One nominal generator subgroup contained only two images. This is useful as a smoke result, not evidence of generalization.

Web-stress behavior rejects the current release candidate:

| Diagnostic slice | Rows | AI recall | Real recall | Balanced accuracy |
|---|---:|---:|---:|---:|
| Calibration stress, 256px JPEG 75 | 64 | 90.625% | 100% | 95.3125% |
| Calibration stress, 192px JPEG 50 | 64 | 0% | 100% | 50% |
| Commercial stress, both thumbnail transforms before calibration | 128 | 0% | 100% | 50% |
| Commercial stress after rejected calibration | 128 | 25% | 95.3125% | 60.15625% |

The rejected calibration introduced three false positives while recovering only sixteen of sixty-four AI examples. Its per-transform observed false-positive rates were 3.125% and 6.25%, on only 32 real examples per transform. It is quarantined as `quarantine/rejected-models/calibration-small-stress-rejected.json` and must not be loaded, exported into a model, or packaged.

The pre-gate extension ZIP is quarantined as `quarantine/rejected-packages/blur-chrome-pre-release-gate.zip`. These quarantine paths are ignored except for their README and are not build inputs.

## Thumbnail-aware head fine-tune

A frozen-backbone head fine-tune used 256 CC0 DiffusionDB images and 256
COCO 2017 training images, paired with deterministic browser-like thumbnail
degradations. Its calibration split was source-separated RAISE/DFGAN, while
the untouched diagnostic test remained LAION/Midjourney.

The selected epoch looked promising on calibration (thumbnail balanced
accuracy 0.8594, AI recall 0.7500, real recall 0.9688), but did not transfer:

- Untouched full resolution: TP 29, TN 32, FP 0, FN 3; balanced accuracy
  0.9531.
- Held-out thumbnails: TP 16, TN 62, FP 2, FN 48; balanced accuracy 0.6094,
  AI recall 0.2500, real recall 0.9688.

The checkpoint is quarantined under ignored `artifacts/` and is not exported,
packaged, or used by the extension.

These figures describe rejected diagnostic samples, not product accuracy. A future release must satisfy the independently enforced gate in [BENCHMARK.md](BENCHMARK.md).
