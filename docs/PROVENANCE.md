# Local provenance verification

Blur uses the official `@contentauth/c2pa-web` verifier rather than treating marker strings or a JUMBF container as proof. The build bundles both the verifier WebAssembly and its CSP-compatible worker; neither is loaded from a CDN.

The standalone integration API is:

```ts
import { verifyC2pa } from './provenance';

const provenance = await verifyC2pa(new Uint8Array(imageBytes), mimeType);
const finalScore = Math.max(pixelScore, provenance.scoreFloor);
const signals = [...metadataSignals, ...provenance.signals];
```

Only call it with original fetched bytes, before canvas or bitmap decoding discards metadata. Reuse the module-level SDK instance and do not initialize one verifier per image.

Interpretation is deliberately conservative:

- `Valid` means the claim signature and asset binding validated locally.
- `Trusted` additionally means the signer chained to locally available trust material.
- A validated active `c2pa.actions`/`c2pa.actions.v2` assertion with `trainedAlgorithmicMedia` supplies a `0.995` AI score floor.
- A validated `compositedWithTrainedAlgorithmicMedia` assertion supplies a `0.9` AI-modified score floor.
- An invalid manifest, an unverified JUMBF marker, or no manifest supplies no AI/real score change.
- Missing provenance is never evidence that an image is real.

The extension intentionally does not fetch a mutable trust list or external manifest during inference. That preserves the offline guarantee but means a cryptographically valid signer may be reported as not locally trusted. Vendor-private invisible watermark detectors are not implemented because no general redistributable offline detector is available.

Packaged runtime cost at `@contentauth/c2pa-web` 0.13.4 is approximately 7.8 MiB for `c2pa_bg.wasm`, 43 KiB for the worker, and about 43 KiB of additional bundled JavaScript before minification. The manifest CSP must retain `script-src 'self' 'wasm-unsafe-eval'` and `worker-src 'self'`.
