"""Build-time model baking (runs in Dockerfile). No GPU on build host — only fetch
+ CPU-init to trigger downloads to runtime-correct paths."""
import os, zipfile, urllib.request

os.makedirs("/models", exist_ok=True)

def dl(url, dst):
    print(f"[dl] {dst}", flush=True)
    urllib.request.urlretrieve(url, dst)

# inswapper face-swap model
dl("https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
   "/models/inswapper_128.onnx")

# GFPGAN restore model
dl("https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
   "/models/GFPGANv1.4.pth")

# InsightFace buffalo_l (detect/recognize)
ins = os.path.expanduser("~/.insightface/models")
os.makedirs(ins, exist_ok=True)
zp = os.path.join(ins, "buffalo_l.zip")
dl("https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip", zp)
zipfile.ZipFile(zp).extractall(os.path.join(ins, "buffalo_l"))
os.remove(zp)

# BiRefNet — cache weights + remote code (CPU load is fine, just downloads)
print("[dl] BiRefNet", flush=True)
from transformers import AutoModelForImageSegmentation
AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True)

# GFPGAN: instantiate on CPU to trigger facexlib detection/parsing weight downloads
# to the runtime-correct path (gfpgan/weights). basicsr must already be patched.
print("[dl] GFPGAN facexlib weights", flush=True)
from gfpgan import GFPGANer
GFPGANer(model_path="/models/GFPGANv1.4.pth", upscale=1, arch="clean",
         channel_multiplier=2, bg_upsampler=None, device="cpu")

print("[dl] all models baked.", flush=True)
