# Detector model

The repository includes the pinned production checkpoint at `models/detector-thumbnail-head.safetensors`. `npm run model:export` verifies its SHA-256 and reproducibly exports `models/detector.onnx`; the generated ONNX file is intentionally ignored.

The production candidate is a frozen-backbone, thumbnail-robust classifier-head fine-tune of the official `OwensLab/commfor-model-384` Community Forensics ViT-small checkpoint at revision `6076002bf0d9dd37537f965ee2f06f826c333b61`.

- Checkpoint license: MIT
- Backbone: `timm/vit_small_patch16_384.augreg_in21k_ft_in1k`, Apache-2.0
- Source weight SHA-256: `b89f36275f3bf5e2b040eee36597a8f19db051bff9a473a9cf7b2466284fb387`
- Fine-tuned checkpoint SHA-256: `9cb5b56d44fff294e2f52c49bddea51d30b9d88fa29d4b4eaf4095753c9ceb36`
- Training-manifest SHA-256: `d925a86940a80ab4d93320e2406368f1875a8fd7e0a2c2caade3a05c0c0ce4f5`
- Fine-tuning sources: COCO 2017 real images and DiffusionDB-2M generated images, with deterministic paired resize/JPEG/blur/contrast degradations. RAISE/DFGAN were calibration-only; the protected Midjourney/LAION test was not used for training or selection.
- Input: resize shortest side to 440, center-crop 384×384, RGB, ImageNet mean/std, NCHW float32
- Output: one fake-image logit; apply sigmoid

- Exported ONNX SHA-256: `c376ed8cf38f2aedd21c74cfaf51ee9fec7936efd332f54d76bf630a14e2d3bf`
- ONNX opset: 18
- PyTorch ↔ ONNX maximum absolute error on deterministic parity inputs: `4.649162292480469e-06`

Run `npm run model:verify` after export. It checks the graph and rejects maximum absolute logit error above `1e-4`.

The clean-room training and acceptance pipeline is documented in [`docs/MODEL_TRAINING.md`](../docs/MODEL_TRAINING.md). The fixed-threshold protected diagnostic measured 29/32 AI and 32/32 real images at full resolution, and 16/64 AI with 62/64 real images under two severe thumbnail transformations. These small diagnostics guide candidate selection; they are not a claim about the private bounty benchmark.
