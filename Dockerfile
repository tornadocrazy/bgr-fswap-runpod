# RunPod serverless worker (NO ComfyUI): inswapper swap + GFPGAN restore + BiRefNet
# bg-removal. Ports the Modal worker. Carries the hard-won fixes:
#  - torch cu128 + onnxruntime-gpu 1.22 (CUDA 12.8) → sm_75..sm_120, runs on every
#    RunPod GPU incl. Blackwell (cu124/ort1.20 capped at sm_90 → unhealthy on Blackwell)
#  - LD_LIBRARY_PATH -> torch's bundled nvidia libs so onnxruntime finds libcublasLt.so.12
#  - basicsr functional_tensor patch on .py ONLY (+ delete stale .pyc), asserted
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget libgl1 libglib2.0-0 build-essential && rm -rf /var/lib/apt/lists/*

# numpy + torch (CUDA 12.8 build) — cu128 ships kernels for sm_75..sm_120, so it
# runs on EVERY RunPod serverless GPU incl. Blackwell (RTX PRO 6000 / B200, sm_120).
# cu124 only had up to sm_90 → "no kernel image" + unhealthy worker on Blackwell.
RUN pip install --no-cache-dir "numpy<2" torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128

# insightface prebuilt wheel (no source compile) + GPU onnxruntime + the rest
RUN pip install --no-cache-dir \
    https://huggingface.co/iwr-redmond/linux-wheels/resolve/main/insightface-0.7.3-cp312-cp312-linux_x86_64.whl \
    onnxruntime-gpu==1.22.0 \
    "transformers>=4.49" timm einops kornia safetensors huggingface_hub \
    opencv-python-headless pillow gfpgan runpod

# basicsr imports torchvision.transforms.functional_tensor (removed) — patch .py
# files only (sed-ing the .pyc corrupts it: "bad marshal data"), drop stale .pyc,
# then ASSERT the string is gone (fail build loudly if not).
RUN P=$(ls -d /usr/local/lib/python*/site-packages/basicsr) && \
    grep -rl --include='*.py' functional_tensor "$P" | xargs -r \
      sed -i 's/torchvision\.transforms\.functional_tensor/torchvision.transforms.functional/' && \
    find "$P" -name '*.pyc' -delete && \
    ! grep -rq --include='*.py' functional_tensor "$P"

# onnxruntime-gpu needs torch's bundled CUDA 12 libs on the loader path
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:/usr/local/lib/python3.12/site-packages/nvidia/curand/lib:/usr/local/lib/python3.12/site-packages/nvidia/cufft/lib

# Models are NOT baked. They live in RunPod's HF Model Cache for
# `techwavelaps/bgr-fswap-models` and are symlinked into place at boot by
# link_models.sh (run from start.sh before the handler imports them). This
# keeps the image ~6.5 GB instead of ~8.8 GB and speeds cold-start pulls.
WORKDIR /

COPY link_models.sh /link_models.sh
COPY start.sh /start.sh
COPY handler.py /handler.py
RUN chmod +x /link_models.sh /start.sh

CMD ["/start.sh"]
