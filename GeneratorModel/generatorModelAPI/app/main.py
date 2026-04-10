"""
FastAPI proxy that sits between the Core Backend and the vLLM server.

Flow:
  Core Backend  -->  POST /generate_workflow  -->  vLLM /v1/chat/completions
                <--  validated JSON or error  <--  raw LLM string
"""

from __future__ import annotations

import logging
import os

import httpx
from fastapi import FastAPI
from pydantic import BaseModel, Field, ValidationError

from app.guardrails import parse_llm_output
from app.schemas.workflow import VALID_TOOLS, WorkflowResponse

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

VLLM_URL = os.getenv("VLLM_URL", "http://vllm-generator:8001")
VLLM_CHAT_ENDPOINT = f"{VLLM_URL}/v1/chat/completions"

SYSTEM_PROMPT = (
    "You are AutoMend, an automated MLOps incident remediation system. "
    "You generate JSON workflow definitions to resolve infrastructure incidents.\n\n"
    "Available tools:\n"
    "1. scale_deployment(namespace, deployment_name, replicas) "
    "- Scale a Kubernetes deployment\n"
    "2. restart_rollout(namespace, deployment_name) "
    "- Rolling restart of a deployment\n"
    "3. undo_rollout(namespace, deployment_name) "
    "- Roll back to previous version\n"
    "4. send_notification(channel, message, severity) "
    "- Send a Slack alert (severity: info | warning | critical)\n"
    "5. request_approval(channel, prompt_message) "
    "- Request human approval before proceeding\n"
    "6. trigger_webhook(url, method, payload) "
    "- Fire an HTTP webhook (method: GET | POST | PUT | DELETE)\n\n"
    "Respond ONLY with a valid JSON object in this exact format:\n"
    '{"workflow": {"steps": ['
    '{"step_id": 1, "tool": "<tool_name>", "params": {<tool_params>}}, ...'
    "]}}\n\n"
    "Rules:\n"
    "- Use only the 6 tools listed above\n"
    "- step_id starts at 1 and increments sequentially\n"
    "- Include ALL required parameters for each tool\n"
    "- Do NOT include any text outside the JSON object\n"
    "- Do NOT wrap the JSON in markdown code fences"
)

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    """Payload the Core Backend sends to this proxy."""

    user_message: str = Field(
        ...,
        min_length=1,
        description="Natural-language description of the MLOps incident.",
    )
    system_context: str | None = Field(
        default=None,
        description="Optional additional context (e.g. RAG-retrieved docs).",
    )


class GenerateResponse(BaseModel):
    """Unified envelope — always returns 200 with ``success`` flag."""

    success: bool
    workflow: dict | None = None
    error: str | None = None
    details: str | None = None
    raw_output: str | None = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="AutoMend Generative API Proxy", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "healthy", "vllm_url": VLLM_URL}


@app.post("/generate_workflow", response_model=GenerateResponse)
async def generate_workflow(body: GenerateRequest) -> GenerateResponse:
    # ---- 1. Build ChatML payload ----------------------------------------
    system_content = SYSTEM_PROMPT
    if body.system_context:
        system_content += f"\n\nAdditional context:\n{body.system_context}"

    chat_payload = {
        "model": "/models/fused_model",
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": body.user_message},
        ],
        "temperature": 0.0,
        "max_tokens": 1024,
    }

    # ---- 2. Call vLLM ---------------------------------------------------
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            vllm_resp = await client.post(VLLM_CHAT_ENDPOINT, json=chat_payload)
    except httpx.ConnectError:
        logger.error("Cannot connect to vLLM at %s", VLLM_URL)
        return GenerateResponse(
            success=False,
            error="vLLM connection failed",
            details=f"Could not connect to vLLM at {VLLM_URL}. Is the server running?",
        )
    except httpx.TimeoutException:
        logger.error("vLLM request timed out (%s)", VLLM_URL)
        return GenerateResponse(
            success=False,
            error="vLLM request timed out",
            details="The vLLM server did not respond within 60 seconds.",
        )
    except httpx.HTTPError as exc:
        logger.error("HTTP error calling vLLM: %s", exc)
        return GenerateResponse(
            success=False,
            error="vLLM HTTP error",
            details=str(exc),
        )

    if vllm_resp.status_code != 200:
        logger.warning(
            "vLLM returned HTTP %s: %s",
            vllm_resp.status_code,
            vllm_resp.text[:300],
        )
        return GenerateResponse(
            success=False,
            error=f"vLLM returned HTTP {vllm_resp.status_code}",
            details=vllm_resp.text[:500],
        )

    # ---- 3. Extract assistant content -----------------------------------
    try:
        vllm_body = vllm_resp.json()
        raw_output = vllm_body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, ValueError) as exc:
        logger.error("Unexpected vLLM response structure: %s", exc)
        return GenerateResponse(
            success=False,
            error="Unexpected vLLM response format",
            details=str(exc),
            raw_output=vllm_resp.text[:1000],
        )

    # Log truncation warning — the LLM may have been cut off
    finish_reason = (
        vllm_body.get("choices", [{}])[0].get("finish_reason", "unknown")
    )
    if finish_reason == "length":
        logger.warning(
            "vLLM output was truncated (finish_reason=length). "
            "Guardrails will attempt bracket repair."
        )

    # ---- 4. Parse JSON (with repair) ------------------------------------
    parsed = parse_llm_output(raw_output)
    if parsed is None:
        logger.warning("All JSON parse attempts failed for vLLM output")
        return GenerateResponse(
            success=False,
            error="JSON parsing failed",
            details=(
                "Could not parse LLM output as valid JSON after repair "
                "attempts (direct parse, markdown strip, trailing-comma "
                "fix, bracket closing)."
            ),
            raw_output=raw_output,
        )

    # ---- 5. Validate against Pydantic schema ----------------------------
    try:
        validated = WorkflowResponse.model_validate(parsed)
    except ValidationError as exc:
        error_lines = _format_validation_errors(exc)
        logger.warning("Schema validation failed: %s", error_lines)
        return GenerateResponse(
            success=False,
            error="Validation failed",
            details=error_lines,
            raw_output=raw_output,
        )

    logger.info(
        "Workflow validated: %d step(s)", len(validated.workflow.steps)
    )
    # validated is a WorkflowResponse, so model_dump() produces {"workflow": {"steps": [...]}}.
    # This means the full response envelope is {"success": true, "workflow": {"workflow": {"steps": [...]}}}
    # The double-nesting (response.workflow.workflow.steps) is intentional -- WorkflowResponse
    # wraps the Workflow object to match the LLM output format {"workflow": {...}}.
    return GenerateResponse(
        success=True,
        workflow=validated.model_dump(),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_validation_errors(exc: ValidationError) -> str:
    """Turn Pydantic errors into a single human-readable string."""
    parts: list[str] = []
    valid_tools_str = ", ".join(VALID_TOOLS)
    for err in exc.errors():
        loc = " -> ".join(str(segment) for segment in err["loc"])
        msg = err["msg"]
        # Enrich 'tool' errors with the valid list
        if "tool" in loc and "literal" in msg.lower():
            msg += f". Valid tools: {valid_tools_str}"
        parts.append(f"{loc}: {msg}")
    return "; ".join(parts)