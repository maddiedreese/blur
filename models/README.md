# Detector model

Place the pinned, exported model at `models/detector.onnx`.

The selected clean-room foundation is the official `OwensLab/commfor-model-384` Community Forensics ViT-small checkpoint at revision `6076002bf0d9dd37537f965ee2f06f826c333b61`.

- Checkpoint license: MIT
- Backbone: `timm/vit_small_patch16_384.augreg_in21k_ft_in1k`, Apache-2.0
- Source weight SHA-256: `b89f36275f3bf5e2b040eee36597a8f19db051bff9a473a9cf7b2466284fb387`
- Input: resize shortest side to 440, center-crop 384×384, RGB, ImageNet mean/std, NCHW float32
- Output: one fake-image logit; apply sigmoid

- Exported ONNX SHA-256: `6a6d40b50f1b7469c8c3bdc918f87f77cffd6754c486773b8b632985374dab8b`
- ONNX opset: 18
- PyTorch ↔ ONNX maximum absolute error on deterministic parity inputs: `8.58306884765625e-06`

Run `npm run model:verify` after export. It checks the graph and rejects maximum absolute logit error above `1e-4`.
