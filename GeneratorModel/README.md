# AutoMend Backend -- GeneratorModel (Track B)

vLLM serving infrastructure for Model 2, the Generative Architect that converts MLOps incident descriptions into structured JSON remediation workflows.

---

## Overview

Model 2 (Track B) is a fine-tuned **Qwen2.5-1.5B-Instruct** served via vLLM. Given a natural language description of an MLOps incident, it outputs a structured JSON workflow using the 6-tool registry:

| Tool | Purpose |
|------|---------|
| `scale_deployment` | Scale a Kubernetes deployment to N replicas |
| `restart_rollout` | Trigger a rolling restart of a deployment |
| `undo_rollout` | Roll back a deployment to the previous revision |
| `send_notification` | Send an alert to Slack or PagerDuty |
| `request_approval` | Pause and request human approval before proceeding |
| `trigger_webhook` | Fire an arbitrary webhook |

vLLM exposes an OpenAI-compatible `/v1/chat/completions` endpoint on **port 8001**.

---

## Model Preparation

The adapter was trained with **MLX** (Apple Silicon), not PEFT/HuggingFace. Because MLX adapter format is not directly loadable by vLLM, the adapter was fused into the base model weights before deployment.

**Fuse command used:**
```bash
python3 -m mlx_lm fuse \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --adapter-path model_2_training/outputs/checkpoints/best_model \
    --save-path model_2_training/outputs/fused_model \
    --dequantize
```

The `--dequantize` flag converts quantized adapter weights back to full precision so vLLM can load them without MLX.

The fused model was then uploaded to GCS:
```bash
gcloud storage cp -r model_2_training/outputs/fused_model/ gs://automend-model2/fused_model/
```

At runtime, `startup.sh` pulls it back down to `/models/fused_model/` inside the container.

---

## Architecture

No custom FastAPI server. vLLM handles all serving with built-in optimizations: continuous batching, PagedAttention, and efficient KV cache management.

```
docker compose up
       |
       v
  startup.sh
       |
       +-- /models/fused_model/config.json exists?
       |          YES --> skip download
       |          NO  --> gcloud storage cp gs://automend-model2/fused_model/
       |
       v
  vllm.entrypoints.openai.api_server
  --model /models/fused_model
  --port 8001
  (OpenAI-compatible /v1/chat/completions)
```

---

## Prerequisites

- GCP VM with an NVIDIA GPU (A100 recommended for bf16 at this model size)
- Docker + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed
- GCP service account key with read access to `gs://automend-model2/`
- Service account: `automend-trainer@automend.iam.gserviceaccount.com`

---

## Setup

```bash
cp .env.example .env
# Edit .env and set GOOGLE_APPLICATION_CREDENTIALS to the absolute path
# of your GCP service account key JSON file on the host machine
```

---

## Running

```bash
docker compose up
```

**First startup:** downloads the fused model from GCS to `./models/` (a few minutes depending on network speed, model is approximately 3 GB).

**Subsequent startups:** `startup.sh` detects `config.json` in the cache and skips the download. vLLM is serving within seconds.

---

## vLLM Configuration

| Flag | Value | Reason |
|------|-------|--------|
| `--max-model-len` | `2048` | Matches training context length |
| `--gpu-memory-utilization` | `0.90` | Leaves headroom for CUDA overhead |
| `--dtype` | `bf16` | Fused model is in bf16; matches training precision |
| `--chat-template` | `/models/fused_model/chat_template.jinja` | Uses the template saved alongside the fused weights |
| `--port` | `8001` | Model 1 classifier occupies 8000 |

---

## Testing

With the vLLM server running:

```bash
python tests/smoke_test.py
```

Sends a sample scale request and verifies:
1. HTTP 200 response
2. `choices` array present
3. Assistant content is valid JSON
4. JSON contains `workflow.steps`

Prints the full response for manual inspection.

---

## File Structure

```
GeneratorModel/
├── docker-compose.yml        # vLLM container config with NVIDIA runtime
├── startup.sh                # GCS download + vLLM launch script
├── .env.example              # Required environment variables (copy to .env)
├── README.md                 # This file
├── Dockerfile                # Placeholder
├── models/                   # Local GCS model cache (gitignored)
│   └── .gitkeep
├── tests/
│   └── smoke_test.py         # Endpoint verification script
└── generatorModelAPI/        # Validation proxy -- Jennisha and Sriram's work
    └── .gitkeep
```

---

## Important Notes for Teammates

**Cannot run locally on Mac.** vLLM requires an NVIDIA GPU and has no MPS (Apple Silicon) support. For local Mac inference testing, use `mlx_lm.generate` directly against the fused model instead.

**The `model` field in API requests must match the `--model` flag.** Always use `/models/fused_model` as the model name in request payloads:
```json
{ "model": "/models/fused_model", ... }
```
Sending any other value returns a 404 from vLLM.

**Use `temperature: 0.0` for workflow generation.** The model is trained to produce deterministic JSON. Non-zero temperature increases the chance of malformed output.

**`generatorModelAPI/` is reserved** for Jennisha and Sriram's validation proxy work (Pydantic schema validation layer on top of the raw vLLM output). Do not add files there without coordinating with them.
