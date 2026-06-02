#!/usr/bin/env bash
# Symlink the worker's model files from RunPod's HF Model Cache into the paths
# the handler + gfpgan/facexlib/insightface expect. Requires the endpoint to
# have Model Caching enabled for HF model id `techwavelaps/bgr-fswap-models`.
#
# Why: baking the 5 model groups (~2.3 GB) into the image bloated it to ~8.8 GB
# and slowed cold-start image pulls. Moving them to RunPod's Model Cache keeps
# the image ~6.5 GB (torch-dominated) and workers overlay-mount the models
# host-locally instead of pulling them in the layer.
#
# Cache layout (RunPod docs):
#   /runpod-volume/huggingface-cache/hub/models--<org>--<name>/
#       refs/main               -> current snapshot hash
#       snapshots/<hash>/<file> -> the actual files (overlay-mounted)
# Snapshot hash changes when the HF repo updates, so resolve it from refs/main.
set -e

CACHE_ROOT=/runpod-volume/huggingface-cache/hub/models--techwavelaps--bgr-fswap-models
REFS_FILE="$CACHE_ROOT/refs/main"

echo "[link_models] waiting for HF model cache..."
WAIT=0; TIMEOUT=300
while [ ! -f "$REFS_FILE" ]; do
    if [ $WAIT -ge $TIMEOUT ]; then
        echo "[link_models] ERROR: cache not ready after ${TIMEOUT}s." >&2
        echo "[link_models] Is Model Caching enabled for techwavelaps/bgr-fswap-models?" >&2
        exit 1
    fi
    sleep 2; WAIT=$((WAIT + 2))
done
HASH=$(cat "$REFS_FILE")
SRC="$CACHE_ROOT/snapshots/$HASH"
echo "[link_models] cache ready in ${WAIT}s, snapshot $HASH"
[ -d "$SRC" ] || { echo "[link_models] ERROR: snapshot dir missing: $SRC" >&2; exit 1; }

mkdir -p /models /models/birefnet /root/.insightface/models/buffalo_l /gfpgan/weights

# face-swap + restore models
ln -sf "$SRC/inswapper_128.onnx" /models/inswapper_128.onnx
ln -sf "$SRC/GFPGANv1.4.pth"     /models/GFPGANv1.4.pth

# BiRefNet (transformers from_pretrained reads this dir; trust_remote_code)
for f in config.json birefnet.py BiRefNet_config.py model.safetensors; do
    ln -sf "$SRC/birefnet/$f" "/models/birefnet/$f"
done

# insightface buffalo_l (FaceAnalysis reads ~/.insightface/models/buffalo_l/)
for f in "$SRC"/buffalo_l/*.onnx; do
    ln -sf "$f" "/root/.insightface/models/buffalo_l/$(basename "$f")"
done

# facexlib weights (GFPGANer face-helper loads from ./gfpgan/weights; WORKDIR=/)
ln -sf "$SRC/facexlib/detection_Resnet50_Final.pth" /gfpgan/weights/detection_Resnet50_Final.pth
ln -sf "$SRC/facexlib/parsing_parsenet.pth"         /gfpgan/weights/parsing_parsenet.pth

# verify nothing dangles
MISSING=0
for f in \
    /models/inswapper_128.onnx \
    /models/GFPGANv1.4.pth \
    /models/birefnet/config.json \
    /models/birefnet/birefnet.py \
    /models/birefnet/BiRefNet_config.py \
    /models/birefnet/model.safetensors \
    /root/.insightface/models/buffalo_l/det_10g.onnx \
    /root/.insightface/models/buffalo_l/w600k_r50.onnx \
    /gfpgan/weights/detection_Resnet50_Final.pth \
    /gfpgan/weights/parsing_parsenet.pth ; do
    [ -e "$f" ] || { echo "[link_models] MISSING: $f" >&2; MISSING=1; }
done
[ $MISSING -eq 0 ] || { echo "[link_models] ERROR: cache incomplete." >&2; exit 1; }
echo "[link_models] all model files linked."
