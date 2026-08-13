# Data provenance ledger

No third-party training or evaluation images are committed to this repository.

Before a source is admitted to a local manifest, record:

| Field | Required evidence |
|---|---|
| Source name and version | Generator release or real collection identifier |
| Origin URL | Authoritative dataset/model/provider page |
| Acquisition date | UTC date |
| License or terms | Exact license and any redistribution/use restrictions |
| Label rationale | Why the source is known generated or real |
| Lineage key | Stable ID joining originals and every derived image |
| Split assignment | Entire source assigned to train, calibration, or test |
| Transform history | Metadata stripping, JPEG/WebP, crop, resize, screenshot, etc. |

Generated images obtained from a public gallery or social platform are not automatically licensed for training or redistribution. Real-image datasets can also impose non-commercial, attribution, or no-redistribution terms. Keep restricted datasets outside the repository and document their terms in the private run ledger.

The base checkpoint is `OwensLab/commfor-model-384`, revision `6076002bf0d9dd37537f965ee2f06f826c333b61`, SHA-256 `b89f36275f3bf5e2b040eee36597a8f19db051bff9a473a9cf7b2466284fb387`, published under MIT. Its timm ViT backbone is Apache-2.0. See `THIRD_PARTY_NOTICES.md` for redistribution notices.

## Recent-generator experiment sources

The optional recent-generator experiment uses pinned, target-side-only extraction. Raw images remain local and are never committed.

| Family | Dataset revision | Accepted field | Declared license |
|---|---|---|---|
| FLUX.2 klein 9B | `stablellama/FLUX.2-klein-base-9B_samples@c07dd3cf504b2c4ca67251e21febf3e8b0a46c36` | `data/animal/dataset_0`, indices 1–80 | CC-BY-4.0 |
| GPT-4o Image | `Rapidata/OpenAI-4o_t2i_human_preference@9fafb39b4bb3bac6e2fbabd13503fa1199fde400` | `image2` only where `model2 == 4o-26-3-25` | CDLA-Permissive-2.0 |
| Ideogram V2 | `Rapidata/Ideogram-V2_t2i_human_preference@9d9bb0aa365e9fbc77e865731ec96655a10e0990` | `image2` only where `model2 == ideogram` | CDLA-Permissive-2.0 |

`tools/extract_recent_parquet.py` never writes the opposite preference-pair side. This matters because those comparison fields contain generator families reserved for protected evaluation. Image bytes are SHA-256 deduplicated, and Rapidata prompt siblings receive a shared `splitGroup`.

Hugging Face repository metadata declares the licenses above, but that metadata does not independently establish a complete per-image sublicense chain for outputs from proprietary generation services. This is recorded as an upstream provenance limitation. Blur distributes only derived model weights, not these dataset images.
