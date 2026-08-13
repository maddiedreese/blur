# Rejected candidate: FerretNet-B

This is an isolated diagnostic, not an integration or an accuracy claim.

## Pinned provenance

- Official repository: `xigua7105/FerretNet`
- Audited revision: `e92796c1a2fb07ccc57ecd7e718e6dce067be5fa`
- License: Apache-2.0
- Architecture: `FerretNet-B-Median-3`, 1,063,837 checkpoint elements
- Checkpoint size: 4.1 MiB
- Checkpoint SHA-256: `fe755d78370bb6547070329553572405b4ecebd23382c9a6cbb11c4ab85a82c2`
- Test preprocessing: RGB, 256px center crop, tensor conversion, and the
  upstream normalization constants. The model computes its upstream median-3
  local-pixel-difference map internally.
- Official decision rule: `sigmoid(logit) > 0.5`

The 192px stress images are smaller than the upstream 256px crop. Torchvision's
official `CenterCrop(256)` behavior pads those images before cropping; the
scorer does not invent a resize that the checkpoint was not evaluated with.

## Protected diagnostics

Every input was SHA-256 checked against its source-separated manifest. These
sets contain only 32 LAION real images and 32 Midjourney images per resolution
group, so the results are release diagnostics rather than general performance
estimates.

| Resolution group | TP | TN | FP | FN | AI recall | Real recall | Balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full resolution | 32 | 30 | 2 | 0 | 100% | 93.75% | 96.875% |
| 256px, JPEG 75 | 0 | 32 | 0 | 32 | 0% | 100% | 50% |
| 192px, JPEG 50 | 0 | 32 | 0 | 32 | 0% | 100% | 50% |

The score files remain ignored under `artifacts/ferretnet-audit/`:

- `full-scores.jsonl`: SHA-256
  `768cc5b97c08902db155292fcbc718bd6b4e2a5160bec2e9471e055f1cb69e20`
- `stress-scores.jsonl`: SHA-256
  `a4c51616e11429379dcf252324091a55078e4e7aee9dbd1e238da4f1aab72181`

## Positive-only rescue check

At the official threshold FerretNet recovers none of the current model's
thumbnail false negatives. A threshold chosen after inspecting this protected
set is not a valid calibration, but an oracle sweep shows the best possible
case under a limit of one additional real false positive out of 32:

| Transform | Oracle strict threshold | Recovered base misses | Added real false positives |
|---|---:|---:|---:|
| 256px, JPEG 75 | `> 0.0037024813` | 3/32 | 1/32 |
| 192px, JPEG 50 | `> 0.0031710302` | 0/32 | 1/32 |

A single threshold satisfying the one-false-positive limit on both transforms
is `> 0.0037024813`; it recovers the same three 256px cases and zero 192px
cases. These extremely low thresholds are not high-confidence votes and their
selection on the protected test rows would be leakage. They are reported only
as an upper bound on complementarity.

For context, a full-resolution threshold `> 0.9330577254` permits one real
false positive and retains 31/32 AI images. That separation disappears after
thumbnail recompression.

## Decision

Rejected for browser integration. The candidate reproduces its paper's stated
compression weakness on the exact path that needs improvement. Its tiny model
size does not compensate for zero thumbnail recall, and no defensible
positive-only gate recovers the failures while preserving the real-image
false-positive budget.
