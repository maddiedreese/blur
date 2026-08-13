# Blur

Blur is a privacy-preserving Manifest V3 Chrome extension that identifies AI-generated images directly in the browser. It never sends images, URLs, or inference results to the developer or an inference service.

## Features

- Automatically discovers responsive, lazy-loaded, and dynamically inserted webpage images.
- Displays an AI/real score on every analyzed image at the fixed 65% bounty threshold.
- Runs a pinned Community Forensics ViT-small detector with WebGPU and a WASM fallback.
- Inspects bounded JPEG, PNG, and WebP metadata as a conservative supporting signal.
- Bundles all inference code and model assets in the packaged extension.
- Uses no server, localhost process, external API, telemetry, or remote inference.

## Architecture

- A content script discovers visible, lazy-loaded, and dynamically inserted images and displays scores.
- The service worker retrieves the bytes of images already displayed on a page and passes them to an extension-owned offscreen document.
- The offscreen document inspects provenance metadata and runs ONNX Runtime Web locally with WebGPU, falling back to WebAssembly.
- Inference assets are bundled with the extension. Runtime network access is used only to retrieve the webpage images selected for local analysis.

The upstream detector reports 89.3% mean per-generator accuracy on 21 held-out generators. That is upstream evidence, not a claim about the private bounty benchmark. See [Community Forensics](https://arxiv.org/abs/2411.04125) and the pinned model details in [`models/README.md`](models/README.md).

## Build

Prerequisites: Node.js 20 or newer.

```sh
npm ci
python3.11 -m venv .venv311
.venv311/bin/pip install -r tools/model-requirements.txt
npm run model:export
npm run model:verify
npm run verify
npm run package
```

Load `dist/` using **Chrome → Extensions → Developer mode → Load unpacked**.

The packaged artifact is `release/blur-chrome.zip`. The source checkpoint and generated ONNX file are intentionally excluded from Git; their exact revisions and checksums are committed, and the export refuses a mismatched download.

## Privacy and offline verification

1. Build and load the extension in a clean Chrome profile.
2. Visit a test page once while online so its images are available.
3. Disable network access and block localhost.
4. Reload from browser cache or open a local fixture page.
5. Confirm scores are produced without any inference-related request.

## Verification performed

- TypeScript, ESLint, and unit tests pass.
- ONNX graph validation passes.
- PyTorch-to-ONNX logit parity passes with maximum absolute error `8.58306884765625e-06`.
- A clean-profile Chrome for Testing smoke test produces an on-page score using WebGPU.
- The same smoke test with GPU disabled produces an on-page score using WASM.
- Chrome accepts and packs the MV3 manifest.

## Limitations

AI-image detection is probabilistic. Recompression, screenshots, novel generators, illustrations, CGI, and partial AI edits can produce false positives or false negatives. Unsigned metadata only adjusts the visual score; its absence is never treated as evidence that an image is real.

See [`docs/COMPLIANCE.md`](docs/COMPLIANCE.md) and [`PRIVACY.md`](PRIVACY.md).

## License

[MIT](LICENSE). Required runtime and model attributions are recorded in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
