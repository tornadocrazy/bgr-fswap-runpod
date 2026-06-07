"""
RunPod serverless worker (NO ComfyUI): face swap (InsightFace inswapper) + GFPGAN
restore + background removal (BiRefNet). Ported from the Modal worker.

Input  (job["input"]):
  op:          "both" (default) | "faceswap" | "bgremove"
  image:       base64 of the target/generated image
  source_face: base64 of the user's selfie (required for faceswap/both)
  feather:     float, gaussian edge feather px (default 0.8)
  erode:       int, erode px to cut dark fringe (default 1)
  crop:        bool (default True). True = swap+restore on the upper face crop then
               stitch back (fast). False = swap+restore the FULL image (no crop, no
               stitch; a bit slower).
Output:
  { "image": "<base64 png>", "format": "png", "had_alpha": bool }

Pipeline (matches Modal): [crop upper-40%×centre-40%] -> inswapper swap -> GFPGAN
restore -> [stitch back] -> BiRefNet bg-removal (erode+feather edge clean).
The bracketed crop/stitch steps are skipped when crop=False.
"""
import os, io, base64, time

os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")

import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import transforms
import insightface
from insightface.app import FaceAnalysis
from transformers import AutoModelForImageSegmentation
from gfpgan import GFPGANer
import runpod

MODELS_DIR = "/models"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
HALF = DEV == "cuda"
# HEURISTIC (not ORT's default EXHAUSTIVE) skips the ~30s of first-inference cuDNN
# conv-algorithm autotuning that dominated cold starts; warm requests are unaffected.
PROVIDERS = [
    ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC", "cudnn_conv_use_max_workspace": "0"}),
    "CPUExecutionProvider",
]

print(f"[init] device={DEV} loading models...", flush=True)
_t0 = time.time()

# BiRefNet (bg removal). Prefer the RunPod Model Cache copy (linked to
# /models/birefnet by link_models.sh); fall back to the HF id when the image
# bakes models in (so the same handler runs on cached + embedded builds).
_BIREFNET_SRC = "/models/birefnet" if os.path.isdir("/models/birefnet") else "ZhengPeng7/BiRefNet"
print(f"[init] BiRefNet source: {_BIREFNET_SRC}", flush=True)
_birefnet = (
    AutoModelForImageSegmentation
    .from_pretrained(_BIREFNET_SRC, trust_remote_code=True)
    .to(DEV).eval()
)
_birefnet = _birefnet.half() if HALF else _birefnet.float()
_tf = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# InsightFace: detection + recognition only (skip 2D/3D landmark + genderage)
_face = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"], providers=PROVIDERS)
_face.prepare(ctx_id=0 if HALF else -1, det_size=(640, 640))
_swapper = insightface.model_zoo.get_model(f"{MODELS_DIR}/inswapper_128.onnx", providers=PROVIDERS)

# GFPGAN restore (matches RunPod ReActor: GFPGANv1.4)
_restorer = GFPGANer(
    model_path=f"{MODELS_DIR}/GFPGANv1.4.pth",
    upscale=1, arch="clean", channel_multiplier=2, bg_upsampler=None, device=DEV,
)
print(f"[init] ready in {time.time()-_t0:.1f}s on {DEV} (fp16={HALF})", flush=True)


def _b64_to_pil(s):
    if isinstance(s, str) and "," in s and s.lstrip().startswith("data:"):
        s = s.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB")


def _pil_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _swap(target_pil, source_pil, crop=True):
    """Face-swap source onto target.
      crop=True  (default): swap+restore on the upper-40%×centre-40% face region for
                 speed, then stitch it back into the full image.
      crop=False: swap+restore on the FULL image — no crop, no stitch."""
    src = cv2.cvtColor(np.array(source_pil), cv2.COLOR_RGB2BGR)
    sfs = _face.get(src)
    if not sfs:
        raise ValueError("no face detected in source_face")
    sf = max(sfs, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    W, H = target_pil.size
    if crop:
        # upper-40% height x centre-40% width (face region) for fast swap+restore
        x0, y0, x1, y1 = int(0.30 * W), 0, int(0.70 * W), int(0.40 * H)
        region = cv2.cvtColor(np.array(target_pil.crop((x0, y0, x1, y1))), cv2.COLOR_RGB2BGR)
        tfs = _face.get(region)
        if not tfs:                          # face outside the crop → fall back to full
            region = cv2.cvtColor(np.array(target_pil), cv2.COLOR_RGB2BGR)
            x0, y0 = 0, 0
            tfs = _face.get(region)
    else:
        region = cv2.cvtColor(np.array(target_pil), cv2.COLOR_RGB2BGR)
        x0, y0 = 0, 0
        tfs = _face.get(region)
    if not tfs:
        raise ValueError("no face detected in target image")

    t = time.time()
    res = region
    for f in tfs:
        res = _swapper.get(res, f, sf, paste_back=True)
    t_swap = time.time() - t
    t = time.time()
    _, _, res = _restorer.enhance(res, has_aligned=False, only_center_face=False, paste_back=True)
    print(f"[time] swap={t_swap*1000:.0f}ms restore={(time.time()-t)*1000:.0f}ms crop={crop}", flush=True)

    full = cv2.cvtColor(np.array(target_pil), cv2.COLOR_RGB2BGR)
    rh, rw = res.shape[:2]
    full[y0:y0 + rh, x0:x0 + rw] = res
    return Image.fromarray(cv2.cvtColor(full, cv2.COLOR_BGR2RGB))


def _bgremove(pil, feather=0.8, erode=1):
    t = time.time()
    inp = _tf(pil).unsqueeze(0).to(DEV)
    if HALF:
        inp = inp.half()
    with torch.no_grad():
        pred = _birefnet(inp)[-1].sigmoid().float().cpu()[0].squeeze()
    m = np.array(transforms.ToPILImage()(pred).resize(pil.size))
    if erode and erode > 0:
        m = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=int(erode))
    if feather and feather > 0:
        m = cv2.GaussianBlur(m, (0, 0), float(feather))
    out = pil.copy()
    out.putalpha(Image.fromarray(m))
    print(f"[time] bgremove={(time.time()-t)*1000:.0f}ms", flush=True)
    return out


def _detect_face_box(pil_rgba):
    """insightface bbox of the (largest) face in the final image. Returns the box in
    THIS image's pixel coords so the backend can scale directly — no server-side ML."""
    rgb = pil_rgba.convert("RGB")
    bgr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
    faces = _face.get(bgr)
    if not faces:
        return None
    f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    x1, y1, x2, y2 = [int(v) for v in f.bbox]
    return {"left": x1, "top": y1, "width": x2 - x1, "height": y2 - y1}


def handler(job):
    try:
        inp = job.get("input", {}) or {}
        op = inp.get("op", "both")
        if "image" not in inp:
            return {"error": "missing 'image' (base64)"}
        img = _b64_to_pil(inp["image"])

        if op in ("faceswap", "both"):
            if "source_face" not in inp:
                return {"error": "op requires 'source_face' (base64)"}
            crop = bool(inp.get("crop", True))  # False → swap full image (no stitch seam)
            img = _swap(img, _b64_to_pil(inp["source_face"]), crop=crop)

        had_alpha = False
        if op in ("bgremove", "both"):
            img = _bgremove(img, float(inp.get("feather", 0.8)), int(inp.get("erode", 1)))
            had_alpha = True
            # trim transparent border on the GPU side so the returned cutout's top row
            # is the head top (backend anchors head edge at headTopY) — saves a server op.
            alpha = img.getchannel("A")
            bbox = alpha.getbbox()
            if bbox:
                img = img.crop(bbox)

        # insightface bbox of the final user face, in the RETURNED image's coords, so
        # the backend scales (refFaceHeight / face.height) without running getFaceBox.
        face = _detect_face_box(img)

        return {"image": _pil_to_b64(img), "format": "png", "had_alpha": had_alpha, "face": face}
    except Exception as e:
        import traceback
        return {"error": str(e), "trace": traceback.format_exc()[-800:]}


runpod.serverless.start({"handler": handler})
