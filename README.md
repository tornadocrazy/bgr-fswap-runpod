# bgr-fswap-runpod

RunPod serverless worker, **no ComfyUI** — plain Python handler. Ported from the Modal
worker (`bgr-fswap-modal`). Does **face swap (inswapper) + GFPGAN restore + BiRefNet
background removal** with the crop-swap-stitch + erode/feather edge-clean optimizations.

## Request
```json
{ "input": { "op": "both", "image": "<b64 generated>", "source_face": "<b64 selfie>",
             "feather": 0.8, "erode": 1 } }
```
`op`: `both` (default) | `faceswap` | `bgremove`. Returns `{ "image": "<b64 png>", "had_alpha": true }`.

## Pipeline
crop (upper40%×centre40%) → inswapper swap → GFPGAN restore → stitch → BiRefNet bg-removal
(erode+feather edge clean). Models baked into the image; no runtime downloads.

## Deploy
RunPod → Serverless → New Endpoint → GitHub source → this repo, Dockerfile `Dockerfile`.
Pick a cheap GPU (16 GB: A4000 / RTX 2000 Ada). Set workers min 0–1, max ~3.
No build secrets needed (all model sources public).

## Measured (on Modal T4, same code — RunPod A4500 ≈ similar)
warm: swap ~115ms + GFPGAN restore ~340ms + bg-removal ~360ms = ~1.3s compute.
Cold start (no ComfyUI boot): ~10–15s model load + GPU acquire.

## Lessons baked in (do not regress)
- torch **cu124** (onnxruntime-gpu 1.20 needs CUDA 12; torch default cu13 breaks it)
- `LD_LIBRARY_PATH` → torch's nvidia/*/lib (onnxruntime finds libcublasLt.so.12)
- basicsr patch on **.py only** + delete `.pyc` (sed on .pyc → "bad marshal data")
