# Reproducible benchmark contract

The benchmark is source-separated and has three immutable roles:

- `train`: model fitting only.
- `calibration`: probability calibration and fusion weights only.
- `test`: one-time final measurement only.

The test split must never select a checkpoint, threshold, calibration parameter, preprocessing choice, or metadata weight. The deployed classification rule is always `score >= 0.65`.

## Dataset manifest

Use JSON Lines with one record per original or deterministic derivative:

```json
{"id":"stable-id","baseId":"original-id","sha256":"...","label":1,"split":"test","source":"collection","generatorFamily":"held-out-family","generatorModel":"exact-model","contentGroup":"photo","transform":"jpeg-75","license":"..."}
```

`baseId` ties every crop, resize, screenshot, and recompression to its original. AI rows require `generatorFamily`; real rows use `source` as their held-out acquisition domain. `splitGroup` may add a stronger grouping key for prompt siblings or a collection batch.

Validate before training or measurement:

```sh
npm run benchmark:validate -- dataset.jsonl
```

Validation rejects duplicate IDs and any cross-split reuse of bytes, base images, real sources, generator families, or explicit split groups. Perceptual near-duplicate clustering must happen during dataset preparation; assign each cluster one `splitGroup`.

## Calibration

Generate raw, uncalibrated visual scores for the calibration split only:

```json
{"id":"stable-id","split":"calibration","label":1,"rawScore":0.82}
```

Fit the deterministic regularized Platt calibrator:

```sh
npm run benchmark:calibrate -- calibration-scores.jsonl calibrator.json
```

The command refuses any row not explicitly marked `calibration` and refuses to overwrite an existing calibrator. Commit the calibrator with the checkpoint SHA-256 and preprocessing version. Apply it identically in the reference and browser runtimes, then freeze it before test inference.

A deployable calibration artifact additionally requires at least 100 unique
base images per class, at least three real-image acquisition sources, at least
three generator families, and at least 25 unique base images in every included
source or family. Deterministic derivatives retain their original `baseId` and
cannot inflate these counts. Choose identity versus temperature or Platt
mapping using deterministic `baseId`-grouped cross-validation entirely inside
the calibration split. A non-identity mapping is eligible only when it lowers
out-of-fold Brier score without reducing balanced accuracy or increasing the
false-positive rate at the frozen `score >= 0.65` rule. The final test split
remains untouched until the mapping and checkpoint are frozen.

The current local calibration smoke set is not adequate for a deployable
mapping: its 32 real bases all come from RAISE and its 32 AI bases all come from
DFGAN. It is useful for detecting obvious regressions, but its transformations
are correlated derivatives of those same 64 bases and are not additional
calibration samples.

## Evaluation

Final score files contain the frozen calibrated score plus slice metadata:

```json
{"id":"stable-id","label":1,"score":0.87,"source":"collection","generatorFamily":"held-out-family","contentGroup":"photo","transform":"metadata-stripped"}
```

Run:

```sh
npm run benchmark:evaluate -- test-scores.jsonl > report.json
```

The report includes exact confusion counts, AI recall, real recall, balanced accuracy, false-positive rate, precision, Brier score, and source/generator/content/corruption slices. A slice containing only one class reports balanced accuracy as `null`; it does not invent a missing-class recall.

Balanced accuracy is:

```text
0.5 * (TP / (TP + FN) + TN / (TN + FP))
```

No accuracy claim belongs in project documentation until it is backed by a committed machine-readable report from real labeled rows.

### Deployed Chrome scoring

The reference Python scorer is useful for training loops, but it does not exercise browser decoding, spatial multi-crop aggregation, C2PA fusion, or the selected ONNX execution provider. For final evidence, launch a clean Chrome-for-Testing profile with `dist/` as the only unpacked extension and run:

```sh
CDP_PORT=9226 node tools/deployed_score.mjs dataset/manifest.jsonl artifacts/deployed-scores.jsonl
```

The collector serves only the supplied local labeled images over loopback during the test, reads the exact score/runtime exposed by the extension content script, and records the model SHA-256, preprocessing version, and `evaluationMode: "deployed"`. It accepts at most 64 rows per run to stay within the runtime queue bound. `SCORE_TRANSFORM=...` selects a transform slice. The local server is evaluation tooling and is never part of extension inference or the packaged product.

## Release gate

Small samples are diagnostic only. In particular, a clean full-resolution result does not override a thumbnail failure, and calibration that increases false positives while leaving AI recall weak is rejected.

`npm run package:release` cannot create a release ZIP until `artifacts/release-test-scores.jsonl` passes the release gate. `npm run package` remains available for reproducible local-review builds and is not release approval. Every evidence row must be frozen test data scored through the deployed browser pipeline (`evaluationMode: "deployed"`) and identify `baseId`, the exact model SHA-256, preprocessing version, and `resolutionGroup` (`full-res` or `thumbnail`). The gate requires:

- At least 500 unique real and 500 unique AI base images, three held-out real sources, and three held-out generator families. Derivatives cannot inflate these counts.
- Overall balanced accuracy of at least 0.90 and observed real recall of at least 0.98.
- A 95% Wilson lower bound of at least 0.95 for real recall and at least 0.80 for AI recall.
- A 95% Wilson upper bound no greater than 0.02 for false-positive rate.
- At least 100 samples per class in each resolution group, with balanced accuracy at least 0.85 and real recall at least 0.97 in both.
- At least 50 real examples per source and no real source with observed false-positive rate above 0.05.

Run it directly with:

```sh
npm run release:gate -- artifacts/release-test-scores.jsonl
```

These are packaging safeguards, not reported accuracy. Passing them does not justify a public claim unless the underlying labeled rows, source separation, and report are independently auditable.

## Required stress sets

- Held-out current generator families, with metadata both intact and stripped.
- Camera photography from held-out acquisition sources.
- Human digital art, CGI/game renders, edited photos, product photos.
- Screenshots, memes, text-heavy graphics, and small thumbnails.
- Seeded JPEG 95/75/50/30, WebP 80/50, resize 1024/512/256/112, crop 90%/75%, blur, sharpening, alpha compositing, and realistic combined transforms.

Derivatives remain in the same split as their `baseId`. Report pixels-only and provenance-assisted results separately so provenance-rich files cannot conceal weak visual generalization.
