# Model improvement pipeline

This pipeline updates the pinned Community Forensics ViT-small detector without changing its browser input or output contract. It deliberately separates pipeline verification from evidence of improved detection quality.

## Dataset manifest

Training data is local and is never downloaded implicitly. Supply JSON Lines with one row per image:

```json
{"id":"unique-row","baseId":"original-image-lineage","path":"images/example.webp","sha256":"...","label":1,"source":"collection","generatorFamily":"generator-family","split":"train","license":"SPDX-or-exact-terms","originUrl":"https://authoritative.example/item"}
```

- `label` is `0` for real and `1` for generated.
- `baseId` must be shared by an original and all crops, recompressions, screenshots, or other derivatives.
- `sha256` binds the manifest row to exact bytes and is checked before training.
- `source` identifies the originating collection; AI rows additionally require `generatorFamily`.
- Entire generator families and real-image sources must belong to one split. The loader rejects source or lineage leakage.
- Every row requires license and origin provenance. A URL is not itself proof of redistribution permission; review the actual terms before use.

Recommended source groups include recent generators absent from the base checkpoint and hard real negatives: human digital art, CGI, screenshots, memes, highly processed photography, phone computational photography, and scanned film. Keep at least one recent generator family and one real collection untouched as test-only sources.

## Reproducible run

```sh
.venv311/bin/python tools/train_detector.py data/manifest.jsonl \
  --output models/detector-finetuned.safetensors \
  --min-ai-recall 0.75 \
  --min-real-recall 0.75 \
  --max-fpr 0.25 \
  --max-full-ba-drop 0.01 \
  --max-full-real-recall-drop 0.02

.venv311/bin/python tools/score_manifest.py data/manifest.jsonl \
  --checkpoint models/detector-finetuned.safetensors \
  --split calibration \
  --output artifacts/raw-logits.jsonl

.venv311/bin/python tools/calibrate_model.py artifacts/calibration-logits.jsonl \
  --output models/calibration.json

.venv311/bin/python tools/export_model.py \
  --checkpoint models/detector-finetuned.safetensors \
  --calibration models/calibration.json \
  --output models/detector-finetuned.onnx

.venv311/bin/python tools/verify_model.py \
  --checkpoint models/detector-finetuned.safetensors \
  --model models/detector-finetuned.onnx
```

`score_manifest.py` writes raw logits. Select calibration rows with `--split calibration` before calibration. `calibrate_model.py` rejects any non-calibration row and preserves the fixed `0.65` decision threshold. Training uses seed `11997733`; the output run record includes hashes, hyperparameters, and the exact train/calibration sources.

### Guarded partial-backbone adaptation

The default command preserves the existing full-model fine-tuning behavior, and `--freeze-backbone` remains a head-only smoke-test mode. For a lower-risk accuracy experiment, adapt only the final ViT blocks:

```sh
.venv311/bin/python tools/train_detector.py data/training/manifest.jsonl \
  --unfreeze-last-n-blocks 2 \
  --learning-rate 2e-5 \
  --backbone-learning-rate 2e-6
```

This freezes patch embedding and all earlier transformer blocks, trains the classification head at `--learning-rate`, and trains the final `N` blocks plus the final representation norm at the lower backbone rate. When `--backbone-learning-rate` is omitted in partial mode it defaults to one tenth of the head rate. The block count is checked against the instantiated timm ViT and cannot be combined with `--freeze-backbone`. Start with one or two blocks; increasing the count materially raises overfitting and catastrophic-forgetting risk and is justified only by source-separated calibration results.

Runs record the adaptation mode, exact unfrozen block indices, trainable parameter names/count, and both learning rates. Python, NumPy, CPU/CUDA Torch RNGs, the CUDA BLAS workspace, and strict deterministic Torch algorithms are configured before model construction. A nondeterministic kernel fails the run instead of silently weakening reproducibility. Reproducibility still depends on the pinned hardware/software environment.

The calibration schema matches the extension formula `sigmoid((rawLogit + bias) / temperature)`. For the final release, collect raw scores through the built extension so browser decoding and multi-crop aggregation are represented; offline center-crop logits are suitable for model iteration but are not a substitute for deployed-path calibration.

Every training image is paired with its untouched full-resolution version and a deterministic browser-stress version. Stress examples downsample the short side to 112, 160, 192, 256, or 320 pixels; apply JPEG quality 30–95; and optionally add blur, contrast variation, and a horizontal flip. This pairing prevents thumbnail robustness from replacing the full-resolution signal. Evaluation preprocessing stays identical to production: RGB, shortest-side resize to 440, center crop to 384, `[0,1]`, ImageNet normalization, NCHW float32.

Checkpoint selection uses the deployed decision rule `sigmoid(rawLogit) >= 0.65`, never validation loss alone. Each epoch reports balanced accuracy, AI recall, real recall, false-positive rate, and exact confusion counts for both untouched calibration images and deterministic thumbnail/compression stress images. An epoch is eligible only if both paths meet the configured recall/FPR floors and its untouched balanced accuracy and real recall remain within the configured drops from the pinned base checkpoint. Among eligible epochs, selection maximizes the worse of untouched and stress balanced accuracy, then their mean, then lower BCE. If no epoch is eligible, training exits without writing a checkpoint.

`Midjourney`, `LAION`, `RAISE`, and `DFGAN` are fail-closed protected tokens in training rows. They cover the currently held-out Midjourney/LAION test sets and RAISE/DFGAN calibration sets and cannot be admitted through a mislabeled `train` row. They remain available only for their assigned measurement roles.

There is currently no safe local training manifest. The available local image corpora are exclusively the protected test/calibration sources above. Once a licensed, source-separated training corpus exists, use:

```sh
.venv311/bin/python tools/train_detector.py data/training/manifest.jsonl \
  --checkpoint models/commfor-model-384.safetensors \
  --output models/detector-thumbnail-robust.safetensors \
  --epochs 3 --batch-size 8 --learning-rate 2e-5 \
  --min-ai-recall 0.75 --min-real-recall 0.75 --max-fpr 0.25 \
  --max-full-ba-drop 0.01 --max-full-real-recall-drop 0.02
```

## Residual experiment gate

`experiment_residual.py` evaluates a 24-feature Laplacian, radial-spectrum, channel-gradient, and JPEG-block-boundary branch. It writes an experiment artifact and cannot alter the exported production model. Integration is recommended only if it adds at least 0.02 source-separated calibration balanced accuracy and the frozen fusion also improves the untouched test set. Until both conditions are measured, the production extension remains visual-model-only.

## Acceptance evidence

- Report balanced accuracy at exactly `score >= 0.65`.
- Report real and generated recall separately, overall and by source/degradation.
- Preserve the untouched test split until training, model selection, fusion selection, and calibration are frozen.
- Run ONNX parity after every export. Maximum absolute raw-logit error must remain at or below `1e-4`.
- Do not describe a successful smoke run as an accuracy improvement.

## Plumbing smoke test

`create_smoke_dataset.py` generates twelve deterministic procedural images with disjoint synthetic source IDs. They are deliberately not representative of photographs or modern generators and must never be used to report detector accuracy. Their only purpose is exercising the complete artifact chain on machines without licensed datasets.

Pass `--allow-small-smoke` to calibration only for this procedural check. The resulting calibration artifact is marked `"smoke_only": true` and must never ship.
