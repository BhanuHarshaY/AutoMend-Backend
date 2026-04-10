# AutoMend Backend

ML model serving layer for the AutoMend self-healing MLOps platform.

AutoMend automates the **detect -> diagnose -> remediate** loop for ML production incidents. This repo contains the two model serving components that power that loop:

- **ClassifierModel (Track A)** -- detects and classifies the anomaly type from raw infrastructure telemetry
- **GeneratorModel (Track B)** -- generates a structured JSON remediation workflow from the classified incident

The main AutoMend backend (Celery pipeline, PostgreSQL, audit trail, human-in-the-loop approvals) consumes both of these services over HTTP.

---

## How the Two Models Fit Together

```
Raw infrastructure telemetry
         |
         v
  ClassifierModel API          (port 8000)
  POST /predict_anomaly
  RoBERTa classifier
         |
         | {"class_id": 1, "confidence_score": 0.87, "label": "Resource_Exhaustion"}
         |
         v
  Confidence gate (main backend -- threshold 0.7)
  If score < 0.7: escalate to human, stop here
         |
         v
  GeneratorModel Proxy API     (port 8002)
  POST /generate_workflow
  Qwen2.5-1.5B + LoRA via vLLM (port 8001)
         |
         | {"success": true, "workflow": {"workflow": {"steps": [...]}}}
         |
         v
  Main backend executes workflow steps
  via tool adapters (Kubernetes, Slack, etc.)
```

---

## Repository Structure

```
AutoMend-Backend/
├── ClassifierModel/
│   ├── Dockerfile                         # Container image for Track A API
│   └── classifierModelAPI/
│       ├── app/
│       │   ├── main.py                    # FastAPI app, lifespan loading, /health, /predict_anomaly
│       │   ├── inference.py               # Vocab mapping, tokenization, RoBERTa forward pass
│       │   └── schemas/
│       │       └── anomaly.py             # AnomalyRequest, AnomalyResponse, LABEL_NAMES
│       ├── docs/
│       │   └── track_a_flow.png           # Request flow diagram
│       ├── models/                        # Drop trained checkpoint files here (gitignored)
│       │   └── temp.txt
│       ├── requirements.txt
│       └── README.md
│
├── GeneratorModel/
│   ├── docker-compose.yml                 # vLLM container with NVIDIA runtime
│   ├── startup.sh                         # GCS model download + vLLM launch
│   ├── .env.example                       # Required credentials (copy to .env)
│   ├── Dockerfile                         # Placeholder
│   ├── models/                            # Local cache for GCS model download (gitignored)
│   ├── tests/
│   │   ├── smoke_test.py                  # Direct vLLM endpoint verification
│   │   ├── mock_proxy.py                  # Standalone mock server for Mac dev (port 8002)
│   │   ├── requirements-test.txt
│   │   └── integration/
│   │       ├── conftest.py                # PROXY_URL config, shared helpers
│   │       ├── test_basic_workflows.py    # All 6 tools, single-step
│   │       ├── test_multi_step.py         # Chained multi-step workflows
│   │       ├── test_edge_cases.py         # Empty, long, ambiguous, injection inputs
│   │       └── test_system_prompts.py     # Hallucination and determinism checks
│   └── generatorModelAPI/
│       ├── app/
│       │   ├── main.py                    # FastAPI proxy: vLLM call + guardrails + schema validation
│       │   ├── guardrails.py              # JSON repair (markdown fences, trailing commas, brackets)
│       │   └── schemas/
│       │       └── workflow.py            # WorkflowStep, Workflow, WorkflowResponse, per-tool param models
│       ├── tests/
│       │   ├── conftest.py                # Shared fixtures and mock vLLM response builder
│       │   ├── test_schemas.py            # Pydantic schema unit tests
│       │   ├── test_guardrails.py         # JSON repair unit tests
│       │   └── test_proxy.py             # Proxy endpoint tests (mocked vLLM)
│       ├── requirements.txt
│       └── README.md
│
├── .gitignore
└── README.md                              # This file
```

---

## Model 1 -- ClassifierModel (Track A)

### What it does

Takes a list of integer token IDs representing a 5-minute infrastructure telemetry window and classifies it into one of 7 anomaly categories.

### Model

- **Architecture:** `AutoModelForSequenceClassification` wrapping `roberta-base` (125M params)
- **Training:** Full fine-tune with Focal Loss. No LoRA. NOT YET TRAINED -- dev mode loads base weights from HuggingFace Hub.
- **Input:** `sequence_ids: list[int]` -- integers encoded from CPU/memory metrics, pod status events, and LogHub template IDs
- **Output:** `class_id (0-6)`, `confidence_score (0.0-1.0)`, `label (str)`

### Anomaly Classes

| class_id | label |
|----------|-------|
| 0 | Normal |
| 1 | Resource_Exhaustion |
| 2 | System_Crash |
| 3 | Network_Failure |
| 4 | Data_Drift |
| 5 | Auth_Failure |
| 6 | Permission_Denied |

### Token Vocabulary

Infrastructure events are encoded as integers before reaching this API. The encoding must match the training vocabulary exactly:

| Integer Range | Meaning |
|---------------|---------|
| `0` | `[PAD_TOK]` |
| `100-109` | CPU utilization decile buckets (`[CPU_0]`...`[CPU_9]`) |
| `200-209` | Memory utilization decile buckets (`[MEM_0]`...`[MEM_9]`) |
| `300-304` | Pod status: Terminated, Failed, Waiting, Running, Unknown |
| `400-403` | Events: Add, Remove, Failure, Unknown |
| `1-999` (others) | LogHub event template IDs (`[TMPL_N]`) |

The encoding logic is in the companion data-pipeline repo at `model_1_training/src/data/tokenizer_setup.py`. The main backend's ingest step must encode raw telemetry before calling this API.

### API

**`GET /health`**
```json
{"status": "healthy", "model_loaded": true, "device": "mps"}
```

**`POST /predict_anomaly`**

Request:
```json
{"sequence_ids": [100, 205, 301, 402]}
```

Response:
```json
{"class_id": 1, "confidence_score": 0.87, "label": "Resource_Exhaustion"}
```

Error codes: `422` (empty/invalid input), `503` (model not loaded), `500` (inference error)

### Running

```bash
cd ClassifierModel/classifierModelAPI
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**With trained checkpoint:** place all `save_pretrained()` output files (including `model.safetensors`) into `models/`. The server detects `models/config.json` on startup and loads from there automatically.

**Dev fallback (no checkpoint):** loads `roberta-base` from HuggingFace Hub with a randomly initialized classifier head. Inference runs but predictions are meaningless until trained weights are placed in `models/`.

### Important

The confidence gate is enforced by the **main backend**, not this service. If `confidence_score < 0.7`, the main backend must not proceed to Track B and must escalate to human review.

---

## Model 2 -- GeneratorModel (Track B)

### What it does

Given a natural language MLOps incident description, generates a structured JSON remediation workflow using a 6-tool registry.

### Model

- **Architecture:** `Qwen2.5-1.5B-Instruct` with a QLoRA adapter (r=16, alpha=32)
- **Training:** Trained with MLX on Apple Silicon. Adapter fused into base weights using `mlx_lm fuse --dequantize` before deployment.
- **Status:** FULLY TRAINED. Fused weights at `gs://automend-model2/fused_model/`.
- **Serving:** vLLM with OpenAI-compatible `/v1/chat/completions` on port 8001.

### Tool Registry (6 tools)

| Tool | Required Params |
|------|----------------|
| `scale_deployment` | `namespace (str)`, `deployment_name (str)`, `replicas (int >= 1)` |
| `restart_rollout` | `namespace (str)`, `deployment_name (str)` |
| `undo_rollout` | `namespace (str)`, `deployment_name (str)` |
| `send_notification` | `channel (str)`, `message (str)`, `severity (info\|warning\|critical)` |
| `request_approval` | `channel (str)`, `prompt_message (str)` |
| `trigger_webhook` | `url (str, http/https)`, `method (GET\|POST\|PUT\|DELETE)`, `payload (dict)` |

### Two-Layer Architecture

Raw LLM output is never returned directly to callers. It passes through two layers:

```
vLLM server (port 8001)        -- raw text generation
      |
      v
generatorModelAPI proxy (port 8002)
      |-- guardrails.py        -- JSON repair (markdown strip, trailing commas, bracket close)
      |-- schemas/workflow.py  -- Pydantic validation against 6-tool registry
      v
Structured response to main backend
```

### Response Format

Success:
```json
{
  "success": true,
  "workflow": {
    "workflow": {
      "steps": [
        {"step_id": 1, "tool": "scale_deployment", "params": {"namespace": "prod", "deployment_name": "fraud-model", "replicas": 5}},
        {"step_id": 2, "tool": "send_notification", "params": {"channel": "#ops", "message": "Scaled fraud-model to 5", "severity": "info"}}
      ]
    }
  },
  "error": null,
  "details": null,
  "raw_output": null
}
```

Error (JSON parse failure, schema validation failure, or vLLM connection error):
```json
{
  "success": false,
  "workflow": null,
  "error": "Validation failed",
  "details": "steps -> 0 -> tool: Input should be 'scale_deployment' or ... Valid tools: scale_deployment, ...",
  "raw_output": "<original LLM text>"
}
```

Note: `response.workflow.workflow.steps` is double-nested by design. `WorkflowResponse` wraps the `Workflow` object to match the LLM output format `{"workflow": {...}}`, which is then stored under `GenerateResponse.workflow`.

### Running

**vLLM server (requires NVIDIA GPU -- not Mac):**
```bash
cd GeneratorModel
cp .env.example .env
# Set GOOGLE_APPLICATION_CREDENTIALS in .env
docker compose up
```

First startup downloads the fused model from `gs://automend-model2/fused_model/` (~3 GB). Subsequent starts use the local cache in `models/`.

**Proxy API:**
```bash
cd GeneratorModel/generatorModelAPI
pip install -r requirements.txt
VLLM_URL=http://localhost:8001 uvicorn app.main:app --host 0.0.0.0 --port 8002
```

**Mac local dev (no GPU needed):**
```bash
cd GeneratorModel
python tests/mock_proxy.py   # starts mock on port 8002
PROXY_URL=http://localhost:8002 pytest tests/integration/ -v
```

### Testing

```bash
# Smoke test against live vLLM (must be running)
python GeneratorModel/tests/smoke_test.py

# Unit tests for proxy, schemas, guardrails (no GPU needed)
cd GeneratorModel/generatorModelAPI
pip install pytest pytest-asyncio
pytest tests/ -v

# Integration tests against mock (Mac, no GPU)
cd GeneratorModel
pip install -r tests/requirements-test.txt
python tests/mock_proxy.py &
PROXY_URL=http://localhost:8002 pytest tests/integration/ -v
```

### Important

- `temperature: 0.0` must be used for workflow generation. Non-zero temperature produces malformed JSON.
- The `model` field in vLLM requests must be `"/models/fused_model"` exactly -- any other value returns 404.
- Cannot run vLLM locally on Mac. Use `mlx_lm.generate` for local inference testing against the fused model.

---

## Integration Contract for the Main Backend

### Full call sequence

```python
import httpx

# Step 1: classify the anomaly
track_a = httpx.post(
    "http://classifier:8000/predict_anomaly",
    json={"sequence_ids": incident.sequence_ids},
    timeout=10.0,
).json()

# Step 2: confidence gate (enforce in main backend)
if track_a["confidence_score"] < 0.7:
    # escalate to human, write audit event, stop
    ...

incident.anomaly_label = track_a["class_id"]
incident.anomaly_name  = track_a["label"]
incident.anomaly_prob  = track_a["confidence_score"]

# Step 3: generate remediation workflow
track_b = httpx.post(
    "http://generator-proxy:8002/generate_workflow",
    json={"user_message": f"{track_a['label']} detected: {incident.description}"},
    timeout=60.0,
).json()

if not track_b["success"]:
    # log track_b["error"] and track_b["raw_output"] to audit trail
    # escalate to human
    ...

steps = track_b["workflow"]["workflow"]["steps"]
# dispatch steps to tool adapters
```

### What this repo does NOT implement

These belong in the main AutoMend backend:

- Celery task chain (`ingest_telemetry`, `classify_anomaly`, `generate_workflow`, `execute_workflow`)
- Debounce logic (Redis SETNX on anomaly_class + resource_id + fingerprint)
- Confidence gate enforcement (threshold configurable per anomaly class)
- Budget ceiling checks (cost per LLM call + estimated action cost)
- Human-in-the-loop approval flow (PostgreSQL + WebSocket push)
- Tool adapter execution (Kubernetes, Docker, Slack, PagerDuty, MLflow)
- Audit trail (append-only PostgreSQL writes at every state transition)
- JWT auth and API key middleware

---

## GCP Resources

| Resource | Value |
|----------|-------|
| Project | `automend` |
| Region | `us-central1` |
| Model bucket | `gs://automend-model2` |
| Fused model path | `gs://automend-model2/fused_model/` |
| Container registry | `us-central1-docker.pkg.dev/automend/automend-images` |
| Service account | `automend-trainer@automend.iam.gserviceaccount.com` |

---

## Team

| Component | Owner |
|-----------|-------|
| ClassifierModel API (`app/main.py`, `inference.py`) | Bhanu Harsha |
| ClassifierModel schemas (`schemas/anomaly.py`) | Ahnaf |
| ClassifierModel Dockerfile | Ahnaf |
| GeneratorModel vLLM infra (`docker-compose.yml`, `startup.sh`) | Bhanu Harsha |
| GeneratorModel proxy (`generatorModelAPI/app/`) | Sriram |
| Integration tests (`tests/integration/`, `mock_proxy.py`) | Jennisha |
