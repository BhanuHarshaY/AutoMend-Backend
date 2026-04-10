# GeneratorModel API -- Validation Proxy (Track B)

Lightweight FastAPI proxy that sits between the **Core Backend** and the **vLLM server** (Model 2). It forwards incident descriptions to vLLM, parses the raw LLM string output into valid JSON, validates it against the 6-tool workflow schema using Pydantic v2, and returns a clean response or a structured error. The Core Backend never sees broken JSON.

## Architecture

```
                         Port 8002                       Port 8001
 Core Backend  ──POST──>  Proxy   ──POST──>  vLLM (/v1/chat/completions)
               <──JSON──  (this)  <──raw str──  Qwen2.5-1.5B-Instruct
                          │
                  ┌───────┴────────┐
                  │  guardrails.py │  parse + repair raw LLM string
                  │  workflow.py   │  Pydantic schema gate
                  └────────────────┘
```

The proxy:
1. Receives an incident description from the Core Backend
2. Builds a ChatML payload with the tool-registry system prompt
3. Forwards it to vLLM on port 8001
4. Extracts the assistant message content
5. Runs a 5-stage JSON repair pipeline (guardrails)
6. Validates the parsed JSON against strict Pydantic schemas
7. Returns `{"success": true, "workflow": {...}}` or `{"success": false, "error": "...", "details": "...", "raw_output": "..."}`

## File Structure

```
generatorModelAPI/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application (port 8002)
│   ├── guardrails.py        # JSON parsing and repair utilities
│   └── schemas/
│       ├── __init__.py
│       └── workflow.py      # Pydantic v2 models for the 6-tool registry
├── tests/
│   ├── __init__.py
│   ├── conftest.py          # Shared pytest fixtures + mock vLLM builder
│   ├── test_guardrails.py   # 35 unit tests for JSON repair
│   ├── test_schemas.py      # 39 unit tests for schema validation
│   └── test_proxy.py        # 22 integration tests (mocked vLLM)
├── requirements.txt
└── README.md                # This file
```

## The 6-Tool Registry

Every workflow the LLM generates is composed of steps that call exactly these 6 tools. The proxy rejects anything else.

| Tool | Parameters | Purpose |
|------|-----------|---------|
| `scale_deployment` | `namespace` (str), `deployment_name` (str), `replicas` (int, >= 1) | Scale a Kubernetes deployment to N replicas |
| `restart_rollout` | `namespace` (str), `deployment_name` (str) | Trigger a rolling restart |
| `undo_rollout` | `namespace` (str), `deployment_name` (str) | Roll back to the previous revision |
| `send_notification` | `channel` (str), `message` (str), `severity` ("info" \| "warning" \| "critical") | Send a Slack/PagerDuty alert |
| `request_approval` | `channel` (str), `prompt_message` (str) | Pause and request human approval |
| `trigger_webhook` | `url` (str, must start with http/https), `method` ("GET" \| "POST" \| "PUT" \| "DELETE"), `payload` (dict) | Fire an arbitrary HTTP webhook |

## Prerequisites

- Python 3.10 or higher
- pip
- Access to a running vLLM instance (or use mocked tests without one)

## Setup

### Step 1: Navigate to the project directory

```bash
cd GeneratorModel/generatorModelAPI
```

### Step 2: Create and activate a virtual environment (recommended)

```bash
# Linux/macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `fastapi` -- web framework
- `uvicorn` -- ASGI server
- `pydantic` -- schema validation (v2)
- `httpx` -- async HTTP client for calling vLLM
- `pytest` / `pytest-asyncio` -- testing

## Running the Proxy Server

### Option A: With a running vLLM instance (GCP VM)

Set the `VLLM_URL` environment variable to point at the vLLM server:

```bash
# Linux/macOS
export VLLM_URL=http://<gcp-vm-ip>:8001

# Windows PowerShell
$env:VLLM_URL = "http://<gcp-vm-ip>:8001"
```

Start the server:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8002
```

### Option B: Docker network (alongside vLLM container)

When both containers are on the same Docker network, the default `VLLM_URL=http://vllm-generator:8001` works out of the box:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8002
```

### Option C: Local development (no vLLM)

You can start the server without vLLM -- it will boot fine, but `/generate_workflow` calls will return a structured connection error:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload
```

The `--reload` flag enables hot-reloading during development.

## API Endpoints

### GET /health

Health check. Returns the proxy status and which vLLM URL it is configured to use.

**Request:**
```bash
curl http://localhost:8002/health
```

**Response:**
```json
{
  "status": "healthy",
  "vllm_url": "http://vllm-generator:8001"
}
```

### POST /generate_workflow

Main endpoint. Accepts an incident description, calls vLLM, validates the response, and returns a structured result.

**Request:**
```bash
curl -X POST http://localhost:8002/generate_workflow \
  -H "Content-Type: application/json" \
  -d '{
    "user_message": "Memory leak detected in recommendation-pod, memory at 92% and climbing",
    "system_context": "Namespace: production, Current replicas: 3"
  }'
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `user_message` | string | Yes | Natural-language incident description |
| `system_context` | string | No | Additional context (e.g., RAG-retrieved docs, current system state) |

**Success Response (HTTP 200):**
```json
{
  "success": true,
  "workflow": {
    "workflow": {
      "steps": [
        {
          "step_id": 1,
          "tool": "send_notification",
          "params": {
            "channel": "#mlops-alerts",
            "message": "Memory leak detected in recommendation-pod. Initiating restart.",
            "severity": "warning"
          }
        },
        {
          "step_id": 2,
          "tool": "request_approval",
          "params": {
            "channel": "#mlops-approvals",
            "prompt_message": "Confirm rolling restart of recommendation-pod in production?"
          }
        },
        {
          "step_id": 3,
          "tool": "restart_rollout",
          "params": {
            "namespace": "production",
            "deployment_name": "recommendation-pod"
          }
        },
        {
          "step_id": 4,
          "tool": "send_notification",
          "params": {
            "channel": "#mlops-alerts",
            "message": "Rolling restart complete. Memory utilization stabilized.",
            "severity": "info"
          }
        }
      ]
    }
  },
  "error": null,
  "details": null,
  "raw_output": null
}
```

**Error Response -- Hallucinated tool (HTTP 200):**
```json
{
  "success": false,
  "workflow": null,
  "error": "Validation failed",
  "details": "workflow -> steps -> 0 -> tool: Input should be 'scale_deployment', 'restart_rollout', 'undo_rollout', 'send_notification', 'request_approval' or 'trigger_webhook'",
  "raw_output": "{\"workflow\": {\"steps\": [{\"step_id\": 1, \"tool\": \"kubectl_apply\", ...}]}}"
}
```

**Error Response -- Broken JSON (HTTP 200):**
```json
{
  "success": false,
  "workflow": null,
  "error": "JSON parsing failed",
  "details": "Could not parse LLM output as valid JSON after repair attempts (direct parse, markdown strip, trailing-comma fix, bracket closing).",
  "raw_output": "I'm sorry, I cannot generate a workflow for that."
}
```

**Error Response -- vLLM unreachable (HTTP 200):**
```json
{
  "success": false,
  "workflow": null,
  "error": "vLLM connection failed",
  "details": "Could not connect to vLLM at http://vllm-generator:8001. Is the server running?",
  "raw_output": null
}
```

Note: The proxy always returns HTTP 200 with a `success` boolean. This is by design -- the Core Backend checks `success` instead of HTTP status codes. The only exception is HTTP 422 for invalid request payloads (missing `user_message`).

## How the Guardrails Work

The LLM sometimes returns malformed output. `guardrails.py` implements a 5-stage repair pipeline that tries progressively harder fixes:

| Stage | What It Does | Example It Fixes |
|-------|-------------|-----------------|
| 1. Direct parse | `json.loads()` on the raw string | Clean JSON (most common case) |
| 2. Strip markdown fences | Removes `` ```json ... ``` `` wrappers | LLM wraps output in code blocks |
| 3. Extract JSON object | Finds the first `{ ... }` in surrounding prose | `"Here is the workflow: {...} Let me know"` |
| 4. Fix trailing commas | Removes `,` before `}` and `]` | `{"a": 1,}` becomes `{"a": 1}` |
| 5. Close unclosed brackets | Appends missing `}` / `]` at the end | Truncated output from `max_tokens` limit |

If all 5 stages fail, the proxy returns a structured `"JSON parsing failed"` error with the `raw_output` included for debugging.

## How Schema Validation Works

After JSON parsing succeeds, the parsed dict is validated against strict Pydantic v2 models:

1. **Top-level structure**: Must be `{"workflow": {"steps": [...]}}` -- nothing else
2. **Steps array**: Must be non-empty
3. **step_id ordering**: Must be sequential starting from 1 (1, 2, 3, ...) with no gaps or duplicates
4. **Tool name**: Must be one of the 6 valid tools -- anything else (e.g., `"replicas"`, `"kubectl_apply"`, `"scale"`) is rejected as a hallucination
5. **Tool parameters**: Each tool's params are validated against its specific Pydantic model:
   - All required fields must be present
   - Types must match (e.g., `replicas` must be an int, not `"five"`)
   - Enum values are enforced (e.g., `severity` must be `"info"`, `"warning"`, or `"critical"`)
   - String fields must be non-empty
   - URLs must start with `http://` or `https://`
6. **Extra fields**: Silently ignored (the LLM sometimes adds metadata -- this is harmless)

When validation fails, the error response includes a human-readable `details` string showing exactly which step and field failed.

## Running Tests

All 96 tests run with mocked vLLM responses. No GPU or running vLLM instance needed.

```bash
cd GeneratorModel/generatorModelAPI
python -m pytest tests/ -v
```

### Test breakdown

| File | Tests | What It Covers |
|------|-------|---------------|
| `test_guardrails.py` | 35 | Markdown fence stripping, JSON object extraction, trailing comma fixes, bracket closing, full repair pipeline, edge cases (None, empty string, arrays, prose) |
| `test_schemas.py` | 39 | All 6 tool param models individually, WorkflowStep with hallucinated tools/wrong types/missing params, Workflow with empty steps/bad step_ids, WorkflowResponse top-level |
| `test_proxy.py` | 22 | Happy path (single/multi-step, system_context), guardrail repair through the proxy, schema validation errors, JSON parse failures, vLLM errors (connection refused, timeout, 500, 404), request validation (missing/empty user_message), health endpoint |

### Run specific test files

```bash
# Only guardrails tests
python -m pytest tests/test_guardrails.py -v

# Only schema tests
python -m pytest tests/test_schemas.py -v

# Only proxy integration tests
python -m pytest tests/test_proxy.py -v
```

### Run a specific test by name

```bash
python -m pytest tests/test_schemas.py::TestWorkflowStep::test_hallucinated_tool_name -v
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_URL` | `http://vllm-generator:8001` | Base URL of the vLLM server. Override for local testing or pointing at a remote GCP instance. |

## How Teammates Integrate with This

### Core Backend (Raghav)

Call `POST /generate_workflow` on port 8002. Check the `success` field in the response:

```python
import requests

resp = requests.post("http://localhost:8002/generate_workflow", json={
    "user_message": "Fraud model latency spike, p99 at 2.3s",
    "system_context": "Namespace: production, Deployment: fraud-model-v2, Current replicas: 2"
})

data = resp.json()
if data["success"]:
    workflow = data["workflow"]["workflow"]
    for step in workflow["steps"]:
        print(f"Step {step['step_id']}: {step['tool']}({step['params']})")
else:
    print(f"Error: {data['error']} -- {data['details']}")
```

### Integration Tests (Jennisha)

Hit port 8002 (the proxy), not port 8001 (vLLM directly). The proxy handles all parsing and validation:

```python
# Your tests should call this proxy
resp = requests.post("http://localhost:8002/generate_workflow", json={
    "user_message": "Data drift detected in feature pipeline"
})

# Check the structured response
assert resp.json()["success"] is True
```

### vLLM Infrastructure (Bhanu)

This proxy calls vLLM at `VLLM_URL/v1/chat/completions`. The model field in requests is hardcoded to `/models/fused_model` and temperature is fixed at `0.0` for deterministic output. No changes needed on the vLLM side.

## Port Assignments

| Service | Port |
|---------|------|
| Model 1 -- Classifier API | 8000 |
| Model 2 -- vLLM (raw LLM) | 8001 |
| Model 2 -- Validation Proxy (this) | 8002 |

## Troubleshooting

**`vLLM connection failed` error:**
The proxy cannot reach vLLM. Check that `VLLM_URL` points to the right address and that the vLLM container is running.

**`JSON parsing failed` with `raw_output` showing plain English:**
The LLM generated a conversational response instead of JSON. This happens with ambiguous prompts. Make the `user_message` more specific about the incident.

**`Validation failed` with tool errors:**
The LLM hallucinated an invalid tool name or wrong parameter types. The `details` field shows exactly what went wrong. The Core Backend should handle this as a failed generation and optionally retry.

**`vLLM returned HTTP 404`:**
The `model` field in the request doesn't match the `--model` flag vLLM was started with. The proxy hardcodes `/models/fused_model` -- make sure vLLM is serving from that path.

**`vLLM request timed out`:**
The LLM took longer than 60 seconds to generate a response. This could indicate GPU memory pressure or an extremely long prompt. Check GPU utilization on the vLLM host.
