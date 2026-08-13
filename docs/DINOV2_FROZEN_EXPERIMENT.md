# Frozen DINOv2 ViT-S/14 experiment — rejected

This is an isolated experiment and is not integrated into the extension.

## Reproducibility and isolation

- Backbone: [`timm/vit_small_patch14_dinov2.lvd142m`](https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m), revision `4610ca143709d58a633b6397a74412c2c3842454`.
- License: Apache-2.0, as declared by the official timm model card and Meta DINOv2 repository.
- Architecture: 22.1M-parameter DINOv2 ViT-S/14; frozen backbone and one trained 384-dimensional linear binary head.
- Input: 224×224 ImageNet-normalized RGB. The timm card's native evaluation size is 518×518; 224 was intentionally tested to make browser execution less expensive.
- Training input: only `split=train` records from `data/recent-training/manifest.jsonl`. The 276 records were deterministically split within each source family into 220 fit and 56 validation base images.
- Every fit and validation image was represented as full resolution, 256px/JPEG75, and 192px/JPEG50. Loss weight was balanced 50/50 between real and AI, then equally across AI generator families.
- No Community Forensics calibration or commercial image was opened while fitting or choosing the head. Checkpoint selection used the fixed 0.65 threshold on the recent-source validation partition. Commercial full/stress manifests were opened only after selection.

Reproduce with:

```sh
HF_HOME=artifacts/dinov2-frozen/hf-home .venv311/bin/python \
  tools/dinov2_frozen_experiment.py \
  data/recent-training/manifest.jsonl \
  data/commfor-commercial-64/manifest.jsonl \
  data/commfor-commercial-stress/manifest.jsonl \
  artifacts/dinov2-frozen --batch 8
```

The machine-readable result is `artifacts/dinov2-frozen/report.json`.

## Fixed-threshold results

All results use score ≥ 0.65 as AI-generated.

| Untouched set | TP | FN | TN | FP | AI recall | Real recall | FPR | Balanced accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Commercial full | 22 | 10 | 20 | 12 | 68.75% | 62.50% | 37.50% | 65.63% |
| 192px JPEG50 | 19 | 13 | 20 | 12 | 59.38% | 62.50% | 37.50% | 60.94% |
| 256px JPEG75 | 19 | 13 | 21 | 11 | 59.38% | 65.63% | 34.38% | 62.50% |

The stress AI recall materially exceeds the existing head-only 25% reference, but the false-positive rate is unacceptable at every resolution. This candidate is therefore **hard rejected**. It must not replace or ensemble with the shipped detector based on these measurements.

## Export and browser feasibility

The full frozen detector exported to ONNX opset 18 successfully:

- PyTorch state: 86,591,203 bytes.
- ONNX: 86,662,140 bytes.
- ONNX Runtime CPU parity over three commercial samples: maximum absolute logit error `5.2928924560546875e-05`.

The graph uses standard ViT operations and is technically suitable for ONNX Runtime Web/WASM or WebGPU. However, its 86.7 MB FP32 payload is large for a Chrome extension, and ViT attention remains substantially more expensive than the current compact runtime. Quantization could reduce transfer size but would require a new parity and accuracy evaluation. Browser feasibility does not override the failed real-image safety gate.

## Limitations

- The safe recent-source corpus is small, particularly for Ideogram, and the real training side is COCO-only.
- The commercial test has 32 images per class, so confidence intervals are wide; the measured 34–38% FPR is nevertheless far beyond any reasonable safety gate.
- Last-block tuning was not attempted because the frozen candidate already failed decisively on real recall. Tuning against the protected test would be leakage; doing it responsibly requires a larger, diverse, separately licensed real training/validation corpus.
