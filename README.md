# AutoMend Backend — Track A Inference Service

FastAPI service that serves the **Track A Trigger Engine** — the RoBERTa-based 7-class anomaly classifier for the AutoMend self-healing MLOps platform.

This service is one component of the larger AutoMend backend. It handles the **classify_anomaly** step in the Celery pipeline: it takes a tokenized infrastructure telemetry sequence and returns a predicted anomaly class with a confidence score.

---

## Repository Structure

```
AutoMend-Backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app, lifespan model loading, endpoints
│   ├── inference.py         # Vocab mapping, tokenization, forward pass
│   └── schemas/
│       ├── __init__.py
│       └── anomaly.py       # AnomalyRequest, AnomalyResponse, LABEL_NAMES
├── models/
│   └── temp.txt             # Keeps directory tracked by git (weights are gitignored)
├── .gitignore
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## What Is Implemented

### 1. FastAPI Application (`app/main.py`)

- **App factory** with a `lifespan` context manager (modern FastAPI pattern, no deprecated `@app.on_event`).
- **Startup sequence:**
  1. Auto-detects device: CUDA > MPS (Apple Silicon) > CPU.
  2. Loads the RoBERTa classifier and tokenizer via `_load_model()`.
  3. Sets model to `eval()` mode and disables gradients globally via `torch.set_grad_enabled(False)`.
  4. Stores `model`, `tokenizer`, `device`, and `model_loaded` flag on `app.state`.
  5. Logs: device selected, load time in seconds, total parameter count.
- **Shutdown:** deletes model and tokenizer from `app.state`.

#### Model Loading Logic (`_load_model`)

| Condition | Behavior |
|-----------|----------|
| `models/config.json` exists | Loads model + tokenizer from `models/` directory (`from_pretrained("models/")`) |
| `models/config.json` absent | Falls back to `roberta-base` from HuggingFace Hub — **dev/testing only, classifier head is randomly initialized** |

The `models/config.json` sentinel is used because it is always written by `save_pretrained()` and is not gitignored. Place the trained checkpoint files in `models/` and the service will automatically use them on next startup.

---

### 2. Inference Pipeline (`app/inference.py`)

Three functions that replicate the exact pipeline from the Track A training code (`model_1_training/src/data/`):

#### `build_token_vocab() -> dict[int, str]`

Builds the integer-to-token-string mapping used during training:

| Integer Range | Token String |
|---------------|-------------|
| `0` | `[PAD_TOK]` |
| `100-109` | `[CPU_0]` ... `[CPU_9]` (CPU utilization decile buckets) |
| `200-209` | `[MEM_0]` ... `[MEM_9]` (Memory utilization decile buckets) |
| `300` | `[STS_TERMINATED]` |
| `301` | `[STS_FAILED]` |
| `302` | `[STS_WAITING]` |
| `303` | `[STS_RUNNING]` |
| `304` | `[STS_UNKNOWN]` |
| `400` | `[EVT_ADD]` |
| `401` | `[EVT_REMOVE]` |
| `402` | `[EVT_FAILURE]` |
| `403` | `[EVT_UNKNOWN]` |
| `1-999` (all others) | `[TMPL_{i}]` (LogHub event template E-codes) |

Any integer not in the vocab defaults to `[PAD_TOK]`.

#### `sequence_ids_to_string(sequence_ids, vocab) -> str`

Converts the integer list to a space-separated token string:
```
[100, 205, 301, 402] -> "[CPU_0] [MEM_5] [STS_FAILED] [EVT_FAILURE]"
```

#### `run_inference(model, tokenizer, sequence_ids, device) -> (int, float)`

Full forward pass pipeline:
1. Build vocab, convert `sequence_ids` to token string.
2. Tokenize with `max_length=512`, `padding="max_length"`, `truncation=True` — matches training exactly.
3. Move `input_ids` and `attention_mask` to device.
4. Forward pass under `torch.no_grad()`.
5. Softmax over logits, `argmax` for `class_id`, index back in for `confidence_score`.
6. Returns Python `(int, float)` — not tensors.

---

### 3. Pydantic Schemas (`app/schemas/anomaly.py`)

#### `AnomalyRequest`
| Field | Type | Validation |
|-------|------|-----------|
| `sequence_ids` | `list[int]` | Non-empty (raises 422 if empty) |

#### `AnomalyResponse`
| Field | Type | Constraints |
|-------|------|------------|
| `class_id` | `int` | `0 <= class_id <= 6` |
| `confidence_score` | `float` | `0.0 <= score <= 1.0` |
| `label` | `str` | Human-readable class name |

#### `LABEL_NAMES`
```python
{
    0: "Normal",
    1: "Resource_Exhaustion",
    2: "System_Crash",
    3: "Network_Failure",
    4: "Data_Drift",
    5: "Auth_Failure",
    6: "Permission_Denied",
}
```
This is the single source of truth for label mapping. `main.py` imports and uses it directly — no redefinition elsewhere.

---

## API Endpoints

### `GET /health`

Returns server and model status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "mps"
}
```

### `POST /predict_anomaly`

Classifies a telemetry sequence.

**Request body:**
```json
{
  "sequence_ids": [100, 205, 301, 402]
}
```

**Response:**
```json
{
  "class_id": 1,
  "confidence_score": 0.87,
  "label": "Resource_Exhaustion"
}
```

**Error responses:**
| Code | Condition |
|------|-----------|
| `422` | `sequence_ids` is empty or not a list of ints (Pydantic validation) |
| `503` | Model not loaded (startup failed or still in progress) |
| `500` | Inference threw an unexpected exception |

---

## Running Locally

### Prerequisites
```
Python 3.10+
pip install -r requirements.txt
```

### Start the server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### With a trained checkpoint

Place all files produced by `save_pretrained()` from the Track A training run into `models/`:
```
models/
├── config.json           # required -- triggers local load path
├── tokenizer.json
├── tokenizer_config.json
├── vocab.json
├── merges.txt
└── model.safetensors     # gitignored, add manually
```

The server detects `models/config.json` on startup and loads from `models/` automatically. No code change needed.

### Docker
```bash
docker build -t automend-track-a .
docker run -p 8000:8000 -v $(pwd)/models:/app/models automend-track-a
```

The Dockerfile installs CPU-only PyTorch for a smaller image. For GPU inference, replace the torch install line with the appropriate CUDA wheel URL.

---

## Integration Notes for the Main Backend

### How this service fits into the pipeline

This service handles the `classify_anomaly` Celery task step. The main backend calls it as an HTTP sidecar:

```
ingest_telemetry  -->  POST /predict_anomaly  -->  generate_workflow (Track B)
```

### Expected call from the Celery task

```python
import httpx

response = httpx.post(
    "http://track-a-service:8000/predict_anomaly",
    json={"sequence_ids": incident.sequence_ids},
    timeout=10.0,
)
result = response.json()
# result = {"class_id": 1, "confidence_score": 0.87, "label": "Resource_Exhaustion"}
```

### Confidence gate

The main backend must enforce the confidence gate **after** receiving the response:

```python
CONFIDENCE_THRESHOLD = 0.7

if result["confidence_score"] < CONFIDENCE_THRESHOLD:
    # Do NOT proceed to Track B workflow generation
    # Escalate to human, write audit event
    ...
```

This service returns the raw confidence score and does not apply the gate itself — the gate is the main backend's responsibility since the threshold is configurable per anomaly class.

### Track A output shape for the incident object

Map the response to the canonical incident fields:
```python
incident.anomaly_label = result["class_id"]
incident.anomaly_name  = result["label"]
incident.anomaly_prob  = result["confidence_score"]
```

### Tokenization contract

The `sequence_ids` passed to this service must use the same integer encoding that was used during training (see vocab table above). The encoding logic lives in the companion data-pipeline repo at `model_1_training/src/data/tokenizer_setup.py`. The main backend's ingest step is responsible for encoding raw telemetry into `sequence_ids` before calling this service.

---

## What Is NOT Implemented Here

These are handled by other parts of the main backend:

- Celery task wiring (`ingest_telemetry`, `classify_anomaly` task definitions)
- Track B (Qwen2.5 LLM workflow generation) — separate vLLM sidecar
- Debounce logic (Redis SETNX)
- Budget and confidence gate enforcement
- Human-in-the-loop approval flow
- Audit trail writes (PostgreSQL)
- JWT auth / API key middleware
- WebSocket push for approvals
