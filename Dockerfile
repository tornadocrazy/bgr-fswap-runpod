# RunPod serverless worker (NO ComfyUI): inswapper swap + GFPGAN restore + BiRefNet
# bg-removal. Ports the Modal worker. Carries the hard-won fixes:
#  - torch cu124 (onnxruntime-gpu 1.20 needs CUDA 12, not torch's default cu13)
#  - LD_LIBRARY_PATH -> torch's bundled nvidia libs so onnxruntime finds libcublasLt.so.12
#  - basicsr functional_tensor patch on .py ONLY (+ delete stale .pyc), asserted
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget libgl1 libglib2.0-0 build-essential && rm -rf /var/lib/apt/lists/*

# numpy + torch (CUDA 12.4 build) first
RUN pip install --no-cache-dir "numpy<2" torch torchvision \
    --index-url https://download.pytorch.org/whl/cu124

# insightface prebuilt wheel (no source compile) + GPU onnxruntime + the rest
RUN pip install --no-cache-dir \
    https://huggingface.co/iwr-redmond/linux-wheels/resolve/main/insightface-0.7.3-cp312-cp312-linux_x86_64.whl \
    onnxruntime-gpu==1.20.0 \
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

# bake models into the image (no runtime download)
COPY download_models.py /tmp/download_models.py
RUN python /tmp/download_models.py && rm /tmp/download_models.py

COPY handler.py /handler.py
CMD ["python", "-u", "/handler.py"]
