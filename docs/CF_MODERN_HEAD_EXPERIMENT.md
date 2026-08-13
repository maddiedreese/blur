# Modern-source Community Forensics head — rejected

This candidate is quarantined and is not integrated into the extension.

## Method

- Starting checkpoint: production Community Forensics head `models/detector-thumbnail-head.safetensors`, SHA-256 `9cb5b56d44fff294e2f52c49bddea51d30b9d88fa29d4b4eaf4095753c9ceb36`.
- Architecture: `vit_small_patch16_384.augreg_in21k_ft_in1k`; the complete backbone was frozen and only the 385-parameter binary head was trainable.
- Training data: the 276 `split=train` rows in `data/recent-training/manifest.jsonl` (138 COCO real; 80 FLUX.2 Klein, 43 GPT-4o, and 15 Ideogram AI).
- Augmentation: every training record contributed an original and a deterministic web-style degraded view.
- Sampling: 50% real and 50% AI probability mass; the AI half was divided equally between the three generator families. Sampling was deterministic and with replacement.
- Selection: two epochs at learning rate `1e-4`, batch size 32, AdamW weight decay 0.05. Selection used only the DFGAN/RAISE calibration full and deterministic stress views at fixed threshold 0.65. The untouched commercial manifests were scored only after epoch 2 was selected.
- Candidate SHA-256: `d961c3f46263259f477b220dd411b9e108d80897c5f866e41dba63e396b90b36`.

Reproduction command:

```sh
.venv311/bin/python tools/train_detector.py data/recent-training/manifest.jsonl \
  --checkpoint models/detector-thumbnail-head.safetensors \
  --output artifacts/cf-modern-head-fb-lr1e4.safetensors \
  --epochs 2 --batch-size 32 --learning-rate 0.0001 --weight-decay 0.05 \
  --freeze-backbone --family-balanced-sampling \
  --min-ai-recall 0.25 --min-real-recall 0.9 --max-fpr 0.1 \
  --max-full-ba-drop 0 --max-full-real-recall-drop 0 --device cpu
```

## Untouched commercial results at 0.65

| Checkpoint / transform | TP | FN | TN | FP | AI recall | Real recall | FPR | Balanced accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Production / full | 29 | 3 | 32 | 0 | 90.625% | 100% | 0% | 95.3125% |
| Candidate / full | 32 | 0 | 31 | 1 | 100% | 96.875% | 3.125% | 98.4375% |
| Production / 256px JPEG75 | 9 | 23 | 31 | 1 | 28.125% | 96.875% | 3.125% | 62.5% |
| Candidate / 256px JPEG75 | 12 | 20 | 30 | 2 | 37.5% | 93.75% | 6.25% | 65.625% |
| Production / 192px JPEG50 | 7 | 25 | 31 | 1 | 21.875% | 96.875% | 3.125% | 59.375% |
| Candidate / 192px JPEG50 | 16 | 16 | 30 | 2 | 50% | 93.75% | 6.25% | 71.875% |

## Decision

**Hard reject.** The candidate improves AI recall and balanced accuracy at all three resolutions, but the acceptance rule forbids any FPR regression. Full-resolution FPR changes from 0/32 to 1/32; both thumbnail transforms change from 1/32 to 2/32. The candidate must not replace or be ensembled with the production detector based on this experiment.

The result suggests that modern-family training is useful, but the next iteration needs a broader independent real-photo corpus and a real-safety-aware training objective. The commercial test cannot be used to tune that objective.
