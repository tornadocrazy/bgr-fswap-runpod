#!/usr/bin/env bash
# Boot order matters: handler.py loads every model at import time, so the cache
# symlinks must exist BEFORE python imports it.
set -e
echo "[start] linking models from RunPod HF Model Cache..."
/link_models.sh
echo "[start] models linked — starting handler"
exec python -u /handler.py
