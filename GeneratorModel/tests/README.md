# Integration Test Suite -- End-to-End Pipeline Validation (Track B)

Black-box test suite that validates the entire Model 2 pipeline end-to-end. It sends natural-language incident descriptions to the **validation proxy** at `/generate_workflow` on port 8002 and verifies the full chain (proxy -> vLLM -> JSON parsing -> Pydantic validation) produces correct, structured results.

## Architecture

```
                         Port 8002                       Port 8001
  Test Suite  ──POST──>  Proxy   ──POST──>  vLLM (/v1/chat/completions)
  (238 tests) <──JSON──  (8002)  <──raw str──  Qwen2.5-1.5B-Instruct
                          │
                  ┌───────┴────────┐
                  │  mock_proxy.py │  local stand-in for the real proxy
                  └────────────────┘
```

The tests:

1. Send natural-language incident prompts to port 8002
2. Verify the response has `"success": true` and a valid workflow
3. Check tool names match the 6-tool registry
4. Validate parameter names, types, and values against the Pydantic schemas
5. Confirm edge cases (empty input, unicode, injection, ambiguous) never produce HTTP 500
6. Ensure deterministic output (same prompt produces the same workflow every time)

## File Structure

```
tests/
├── smoke_test.py                # Smoke test for vLLM directly (port 8001)
├── mock_proxy.py                # Mock server mimicking the validation proxy
├── requirements-test.txt        # Test dependencies
├── README.md
└── integration/
    ├── __init__.py
    ├── conftest.py              # Shared helpers and config
    ├── test_basic_workflows.py  # 79 tests -- single-tool requests
    ├── test_multi_step.py       # 43 tests -- multi-step workflow chains
    ├── test_edge_cases.py       # 78 tests -- error handling and edge cases
    └── test_system_prompts.py   # 38 tests -- tool registry and determinism
```

## The 6-Tool Registry

Every workflow step must use one of these 6 tools. The test suite flags any other tool name as a hallucination.

| Tool                | Parameters                                                                                                 | Purpose                                     |
| ------------------- | ---------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| `scale_deployment`  | `namespace` (str), `deployment_name` (str), `replicas` (int, >= 1)                                         | Scale a Kubernetes deployment to N replicas |
| `restart_rollout`   | `namespace` (str), `deployment_name` (str)                                                                 | Trigger a rolling restart                   |
| `undo_rollout`      | `namespace` (str), `deployment_name` (str)                                                                 | Roll back to the previous revision          |
| `send_notification` | `channel` (str), `message` (str), `severity` ("info" \| "warning" \| "critical")                           | Send a Slack/PagerDuty alert                |
| `request_approval`  | `channel` (str), `prompt_message` (str)                                                                    | Pause and request human approval            |
| `trigger_webhook`   | `url` (str, must start with http/https), `method` ("GET" \| "POST" \| "PUT" \| "DELETE"), `payload` (dict) | Fire an arbitrary HTTP webhook              |

## Prerequisites

- Python 3.10 or higher
- pip
- No PyTorch, no ML libraries, no GPU needed

## Setup

### Step 1: Navigate to the project root

```bash
cd AutoMend-Backend
```

### Step 2: Create and activate a virtual environment (recommended)

```bash
# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
pip install -r GeneratorModel/tests/requirements-test.txt
```

This installs:

- `pytest` -- test runner
- `requests` -- HTTP client for calling the proxy
- `httpx` -- alternative HTTP client
- `fastapi` -- needed to run the mock server
- `uvicorn` -- ASGI server for the mock
- `pydantic` -- needed by the mock's request validation

## Running the Mock Server

Start the mock server that mimics the real validation proxy:

```bash
python GeneratorModel/tests/mock_proxy.py
```

The mock starts on `http://localhost:8002`. Verify it is running:

```bash
curl http://localhost:8002/health
```

```json
{
  "status": "healthy",
  "vllm_url": "mock"
}
```

## Running Tests

### Option A: Against the mock server (local, no GPU)

Make sure the mock server is running, then:

```bash
PROXY_URL=http://localhost:8002 pytest GeneratorModel/tests/integration/ -v
```

### Option B: Against the real proxy on GCP

Once the proxy and vLLM are deployed, point at the remote instance:

```bash
PROXY_URL=http://<gcp-instance-ip>:8002 pytest GeneratorModel/tests/integration/ -v
```

### Run specific test files

```bash
# Only basic single-tool tests
PROXY_URL=http://localhost:8002 pytest GeneratorModel/tests/integration/test_basic_workflows.py -v

# Only multi-step tests
PROXY_URL=http://localhost:8002 pytest GeneratorModel/tests/integration/test_multi_step.py -v

# Only edge case tests
PROXY_URL=http://localhost:8002 pytest GeneratorModel/tests/integration/test_edge_cases.py -v

# Only system prompt and determinism tests
PROXY_URL=http://localhost:8002 pytest GeneratorModel/tests/integration/test_system_prompts.py -v
```

### Run a specific test by name

```bash
PROXY_URL=http://localhost:8002 pytest GeneratorModel/tests/integration/test_basic_workflows.py::TestScaleDeployment::test_params_present_and_correct -v
```

## Request and Response Format

The test suite sends requests to `POST /generate_workflow` and validates the response.

**Request:**

```bash
curl -X POST http://localhost:8002/generate_workflow \
  -H "Content-Type: application/json" \
  -d '{
    "user_message": "Scale my fraud-model deployment to 5 replicas in production",
    "system_context": "Current replicas: 2, CPU at 94%"
  }'
```

| Field            | Type   | Required           | Description                                                         |
| ---------------- | ------ | ------------------ | ------------------------------------------------------------------- |
| `user_message`   | string | Yes (min_length=1) | Natural-language incident description                               |
| `system_context` | string | No                 | Additional context (e.g., RAG-retrieved docs, current system state) |

**Success Response (HTTP 200):**

```json
{
  "success": true,
  "workflow": {
    "workflow": {
      "steps": [
        {
          "step_id": 1,
          "tool": "scale_deployment",
          "params": {
            "namespace": "production",
            "deployment_name": "fraud-model",
            "replicas": 5
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

**Error Response (HTTP 200):**

```json
{
  "success": false,
  "workflow": null,
  "error": "Validation failed",
  "details": "Step 1: unknown tool 'replicas'. Valid tools: scale_deployment, restart_rollout, ...",
  "raw_output": "<original LLM string>"
}
```

The proxy always returns HTTP 200 with a `success` boolean. The only exception is HTTP 422 for invalid request payloads (empty or missing `user_message`).

```python
def get_steps(data):
    return data["workflow"]["workflow"]["steps"]
```

## Test Breakdown

All 238 tests run against the proxy endpoint.

| File                      | Tests | What It Covers                                                                                                                                                                                                    |
| ------------------------- | ----- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test_basic_workflows.py` | 79    | Each of the 6 tools triggered individually, param value constraints (replicas >= 1, severity literals, URL format), required param completeness, step structure, variations across namespaces and replica counts  |
| `test_multi_step.py`      | 43    | Multi-tool chaining (2-step through 5-step workflows), step count tolerance, sequential step IDs, logical ordering (approval before destructive actions), per-step param correctness                              |
| `test_edge_cases.py`      | 78    | Empty/missing input (HTTP 422), oversized prompts, ambiguous requests, off-topic input, SQL/XSS/JSON injection, unicode and emoji, multi-line formatting, rapid repeated requests, system_context variations      |
| `test_system_prompts.py`  | 38    | Tool registry enforcement with hallucination blacklist, determinism (same prompt 3x = same result), param completeness per tool, system_context safety, response envelope consistency, cross-tool param isolation |

## How the Mock Server Works

`mock_proxy.py` is a lightweight FastAPI server that replicates the exact request validation, response structure, and error format of the real proxy. It pattern-matches on prompt keywords to return appropriate responses:

| Keyword                    | Response                                                            |
| -------------------------- | ------------------------------------------------------------------- |
| `"scale"`                  | scale_deployment step with regex-extracted replicas and namespace   |
| `"restart"`                | restart_rollout step                                                |
| `"rollback"` / `"undo"`    | undo_rollout step                                                   |
| `"notify"` / `"alert"`     | send_notification step                                              |
| `"approval"` / `"approve"` | request_approval step                                               |
| `"webhook"` / `"trigger"`  | trigger_webhook step                                                |
| `"memory leak"`            | 4-step workflow (notify -> approve -> restart -> notify)            |
| `"critical outage"`        | 5-step workflow (notify -> approve -> restart -> webhook -> notify) |
| `"poem"` / `"weather"`     | Structured error response                                           |
| Anything else              | Single send_notification step                                       |

## Environment Variables

| Variable    | Default                 | Description                                                                                |
| ----------- | ----------------------- | ------------------------------------------------------------------------------------------ |
| `PROXY_URL` | `http://localhost:8002` | Base URL of the proxy to test against. Override with a GCP instance IP for remote testing. |

## Port Assignments

| Service                     | Port |
| --------------------------- | ---- |
| Model 1 -- Classifier API   | 8000 |
| Model 2 -- vLLM (raw LLM)   | 8001 |
| Model 2 -- Validation Proxy | 8002 |

## Troubleshooting

**Tests fail with `ConnectionError`:**
The mock server or real proxy is not running on the expected port. Start the mock with `python GeneratorModel/tests/mock_proxy.py` or set `PROXY_URL` to the correct address.

**Tests fail with `AssertionError: Expected HTTP 200, got 422`:**
The prompt being sent is empty or missing `user_message`. Check the test's `PROMPT` value.

**Tests pass on mock but fail on real proxy:**
The LLM may produce different output than the mock's hardcoded responses. Assertions that check exact values (like `replicas == 5`) are sensitive to this. Fuzzy assertions (like `"fraud" in deployment_name`) are more resilient.

**`TestDeterminism` fails on real proxy:**
The vLLM server may be running with `temperature > 0`. The proxy hardcodes `temperature=0.0`, but verify the vLLM configuration has not been overridden.

**`TestNonMLOpsRequest` unexpectedly succeeds:**
The LLM may generate a valid workflow even for off-topic prompts. This is not a failure if the tool names are in the registry, the test only asserts no 500 error, not that the LLM refuses.

**Mock returns unexpected responses:**
The mock uses keyword matching with a priority order. If a prompt contains multiple keywords (e.g "scale" and "notify"), it may match a multi-step pattern instead of a single-tool pattern. Check the pattern priority in `mock_proxy.py`.
