# Bounty compliance

| Requirement | Implementation |
| --- | --- |
| MIT licensed | Root `LICENSE` and source headers/third-party notices |
| Native Manifest V3 | `static/manifest.json` |
| Browser-only inference | ONNX Runtime Web in an extension offscreen document |
| WebGPU with fallback | WebGPU first; WASM/SIMD fallback verified with GPU disabled |
| Offline after model setup | Runtime and model assets bundled in the release ZIP |
| Automatic webpage analysis | Content script, intersection scheduling, load/src/srcset mutation handling |
| Score for every analyzed image | Fixed overlay badge with full internal score and rounded display score |
| Fixed 65% threshold | `score >= 0.65`; boundary covered by a unit test |
| No cloud/external inference | No inference endpoint or remote executable asset exists |
| No localhost dependency | No native messaging or local process interaction exists |
| Reproducible source build | Locked JS/Python dependencies, pinned checkpoint revision and hashes, deterministic exporter |

## Clean-profile acceptance procedure

1. Follow the complete build steps in the README.
2. Create a fresh Chrome profile and load `dist/` unpacked.
3. Open `e2e/fixture.html` through any static HTTP server.
4. Verify a score badge appears and its title reports `Local webgpu analysis`.
5. Relaunch Chrome with GPU unavailable or disabled and verify the title reports `Local wasm analysis`.
6. Disconnect the network and repeat with a local or cached page to demonstrate that no inference asset is downloaded.
7. Inspect network activity and confirm there are no developer, model, telemetry, API, or inference requests.

## Scoring contract

The extension emits a continuous fake-image score in `[0, 1]`. The only classification rule is `score >= 0.65`. The benchmark CLI in `scripts/evaluate.mjs` reports confusion counts, AI recall, real recall, and balanced accuracy using that same rule.

## Known unsupported cases

Page-owned `blob:` images are currently skipped because Chrome messaging does not reliably transfer raw `ArrayBuffer` values across all supported versions. HTTP(S) and `data:` images are supported. Missing or inaccessible metadata remains neutral.
