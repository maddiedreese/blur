# Detector model comparison

## GAPL feasibility audit

GAPL was evaluated as a candidate after the pinned Community Forensics model produced false negatives. This audit uses only the official sources:

- Code: `UltraCapture/GAPL`, commit `ea32aeb2d619b548daffa6db83296ee1aee70605`
- Checkpoint: `AbyssLumine/GAPL`, revision `ea0341c1ca59862508a1a621fb8072c274bc31dd`
- Checkpoint SHA-256: `ffbcb5eb526f0df0fd197d7266bdd0325b66813e95010f1285685acf2d267235`
- Paper: *Scaling Up AI-Generated Image Detection with Generator-Aware Prototypes*, CVPR 2026

The paper reports 90.4% mean accuracy and 94.9% mean average precision across 55 subsets in six benchmarks. It also reports comparatively low degradation under JPEG compression and Gaussian blur. Those are upstream paper results, not Blur measurements.

### Exact released architecture

The official checkpoint contains 305,736,320 float32 model parameters occupying 1,222,945,280 tensor bytes, plus a `64 × 128` float32 prototype matrix. The 1,223,255,255-byte checkpoint contains:

| Component | Parameters | Float32 bytes |
|---|---:|---:|
| CLIP vision tensors outside PEFT wrappers | 227,608,576 | 910,434,304 |
| PEFT-wrapped q/k/v base layers | 75,571,200 | 302,284,800 |
| LoRA adapters | 2,359,296 | 9,437,184 |
| 1024→128 projection | 131,072 | 524,288 |
| Four-head cross-attention | 66,048 | 264,192 |
| Binary head | 128 | 512 |

The backbone is OpenAI CLIP ViT-L/14. Inference center-crops to `224 × 224`, scales to `[0,1]`, and uses ImageNet mean/std. The model projects the CLIP pooler output from 1024 to 128 dimensions, normalizes it, attends over 64 learned 128-dimensional prototypes, and emits one fake-image logit. The official example applies sigmoid and a 0.5 threshold.

### Browser decision

GAPL is not an immediate ONNX Runtime Web candidate for Blur:

- A direct float32 ONNX would be approximately 1.23 GB before graph overhead—about fourteen times the current 87 MB model.
- Float16 still has a tensor-weight floor near 611 MB. An INT8 weight-only floor is about 306 MB before scales, graph data, runtime buffers, and a separate fallback representation.
- The model requires a CLIP ViT-L/14 transformer plus LoRA-wrapped attention, learned prototype cross-attention, and a large activation/runtime memory budget. Supporting WebGPU and WASM reliably would require distinct quantization work and browser profiling.
- The GitHub source repository has no `LICENSE` file at the audited commit. The Hugging Face model card declares MIT, but that does not provide an explicit source-code license text or copyright notice for copied implementation code.

The official checkpoint was downloaded into ignored `artifacts/gapl/checkpoint.pt` and verified, but no code was copied from the repository. ONNX export and held-out scoring were stopped because package/runtime feasibility and source-license clarity fail the immediate integration gate. Consequently, GAPL has no local accuracy result and must not be described as outperforming the production detector on Blur's held-out sets.

Reproduce the safe checkpoint inventory with:

```sh
.venv311/bin/python tools/gapl_inspect.py artifacts/gapl/checkpoint.pt
```

Reconsider GAPL only if all of the following become acceptable: an explicit upstream source license, a browser-loadable quantized artifact, WebGPU and WASM parity, tolerable per-image latency/memory, and improved balanced accuracy plus real recall on both full-resolution and thumbnail held-out manifests.
