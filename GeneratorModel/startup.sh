#!/bin/bash
set -euo pipefail

MODEL_DIR="/models/fused_model"
GCS_PATH="gs://automend-model2/fused_model/"

# Download from GCS only if model is not already cached
if [ -f "${MODEL_DIR}/config.json" ]; then
    echo "Cache hit: ${MODEL_DIR}/config.json exists, skipping download."
else
    echo "Downloading fused model from ${GCS_PATH} ..."
    gcloud storage cp -r "${GCS_PATH}" "${MODEL_DIR}"
    echo "Download complete."
fi

exec python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_DIR}" \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.90 \
    --dtype bf16 \
    --chat-template "${MODEL_DIR}/chat_template.jinja" \
    --host 0.0.0.0 \
    --port 8001
