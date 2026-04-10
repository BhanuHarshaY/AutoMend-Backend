"""
Mock proxy server that mimics the /generate_workflow endpoint.

Run locally for developing and validating integration tests before
the real proxy and vLLM are deployed.

Usage:
    python tests/mock_proxy.py
    PROXY_URL=http://localhost:8002 pytest tests/integration/ -v
"""

import re
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="AutoMend Mock Proxy")

# the request model
class GenerateRequest(BaseModel):
    user_message: str = Field(..., min_length=1)
    system_context: str | None = None

# ---------------------------------------------------------------------------
# response builders
# ---------------------------------------------------------------------------

def _step(step_id: int, tool: str, params: dict) -> dict:
    return {"step_id": step_id, "tool": tool, "params": params}

def _success(steps: list[dict]) -> dict:
    return {
        "success": True,
        "workflow": {"workflow": {"steps": steps}},
        "error": None,
        "details": None,
        "raw_output": None,
    }


def _error(error: str, details: str = "", raw_output: str = "") -> dict:
    return {
        "success": False,
        "workflow": None,
        "error": error,
        "details": details,
        "raw_output": raw_output or None,
    }


# ---------------------------------------------------------------------------
# pattern matching for prompts
# ---------------------------------------------------------------------------

def _match_prompt(msg: str) -> dict:
    """Route a user message to the appropriate mock response."""
    low = msg.lower().strip()

    # non-MLOps request
    if any(kw in low for kw in ["poem", "weather", "recipe", "joke"]):
        return _error(
            error="JSON parsing failed",
            details="Could not parse LLM output as valid JSON after repair attempts",
            raw_output="I'm sorry, I cannot generate a workflow for that request.",
        )

    # multi-step patterns check
    if "critical outage" in low or "full incident" in low or (
        "outage" in low and "payment" in low
    ):
        wh_url = "https://health.company.com/check"
        url_match = re.search(r"https?://\S+", msg)
        if url_match:
            wh_url = url_match.group(0)
        ns = "production"
        if "staging" in low:
            ns = "staging"
        deploy = "payment-service"
        dm = re.search(r"(?:outage|restart|redeploy)\s+(?:on\s+)?(?:the\s+)?([\w-]+)", low)
        if dm and dm.group(1) not in ("on", "the", "in"):
            deploy = dm.group(1)
        return _success([
            _step(1, "send_notification", {
                "channel": "#incidents", "severity": "critical",
                "message": f"Critical outage detected on {deploy}",
            }),
            _step(2, "request_approval", {
                "channel": "#ops",
                "prompt_message": f"Approve emergency restart of {deploy} in {ns}?",
            }),
            _step(3, "restart_rollout", {
                "namespace": ns, "deployment_name": deploy,
            }),
            _step(4, "trigger_webhook", {
                "url": wh_url, "method": "POST",
                "payload": {"action": "health_check", "target": deploy},
            }),
            _step(5, "send_notification", {
                "channel": "#incidents", "severity": "info",
                "message": f"Restart completed for {deploy}, health check triggered",
            }),
        ])

    # memory leak
    if "memory leak" in low or ("notify" in low and "approval" in low and "restart" in low):
        return _success([
            _step(1, "send_notification", {
                "channel": "#incidents", "severity": "critical",
                "message": "Memory leak detected in recommendation-pod",
            }),
            _step(2, "request_approval", {
                "channel": "#ops",
                "prompt_message": "Confirm rolling restart of recommendation-pod in production?",
            }),
            _step(3, "restart_rollout", {
                "namespace": "production", "deployment_name": "recommendation-pod",
            }),
            _step(4, "send_notification", {
                "channel": "#incidents", "severity": "info",
                "message": "Restart completed for recommendation-pod",
            }),
        ])

    # rollback + notify + webhook
    if ("rollback" in low or "undo" in low) and "notify" in low and "webhook" in low:
        url = "https://tests.company.com/run"
        url_match = re.search(r"https?://\S+", msg)
        if url_match:
            url = url_match.group(0)
        return _success([
            _step(1, "undo_rollout", {
                "namespace": "production", "deployment_name": "api-server",
            }),
            _step(2, "send_notification", {
                "channel": "#incidents", "severity": "critical",
                "message": "Rolled back api-server in production",
            }),
            _step(3, "trigger_webhook", {
                "url": url, "method": "POST",
                "payload": {"action": "smoke_test"},
            }),
        ])

    # scaling and notifying
    if "scale" in low and "notify" in low:
        replicas = 10
        m = re.search(r"(\d+)\s*replicas?", low)
        if m:
            replicas = int(m.group(1))
        deployment = "fraud-model"
        dm = re.search(r"scale\s+([\w-]+)", low)
        if dm:
            deployment = dm.group(1)
        return _success([
            _step(1, "scale_deployment", {
                "namespace": "production", "deployment_name": deployment,
                "replicas": replicas,
            }),
            _step(2, "send_notification", {
                "channel": "#scaling", "severity": "info",
                "message": f"Scaled {deployment} to {replicas} replicas",
            }),
        ])

    # Notify + approval + restart (without "memory leak" keyword)
    if "notify" in low and ("approval" in low or "approve" in low) and "restart" in low:
        return _success([
            _step(1, "send_notification", {
                "channel": "#ops", "severity": "info",
                "message": "Requesting restart",
            }),
            _step(2, "request_approval", {
                "channel": "#ops",
                "prompt_message": "Approve restart?",
            }),
            _step(3, "restart_rollout", {
                "namespace": "staging", "deployment_name": "api-server",
            }),
        ])

    # single-tool patterns

    # request approval, checking before rollback
    if ("approval" in low or "approve" in low) and "ask" in low:
        channel = "#ops"
        cm = re.search(r"#([\w-]+)", msg)
        if cm:
            channel = f"#{cm.group(1)}"
        return _success([
            _step(1, "request_approval", {
                "channel": channel,
                "prompt_message": "Requesting approval before proceeding",
            }),
        ])

    # scale
    if "scale" in low:
        replicas = 5
        m = re.search(r"(\d+)\s*replicas?", low)
        if m:
            replicas = int(m.group(1))
        namespace = "production"
        if "staging" in low:
            namespace = "staging"
        elif "default" in low:
            namespace = "default"
        deployment = "fraud-model"
        dm = re.search(r"scale\s+(?:my\s+)?(?:the\s+)?([\w-]+)", low)
        if dm:
            deployment = dm.group(1)
        return _success([
            _step(1, "scale_deployment", {
                "namespace": namespace,
                "deployment_name": deployment,
                "replicas": replicas,
            }),
        ])

    # restart
    if "restart" in low:
        namespace = "default"
        if "production" in low:
            namespace = "production"
        elif "staging" in low:
            namespace = "staging"
        deployment = "api-server"
        dm = re.search(r"restart\s+(?:the\s+)?([\w-]+)", low)
        if dm:
            deployment = dm.group(1)
        return _success([
            _step(1, "restart_rollout", {
                "namespace": namespace, "deployment_name": deployment,
            }),
        ])

    # rollback
    if "rollback" in low or "undo" in low or "roll back" in low:
        namespace = "staging"
        if "production" in low:
            namespace = "production"
        elif "default" in low:
            namespace = "default"
        deployment = "recommendation-model"
        dm = re.search(r"(?:rollback|undo|roll back)\s+(?:the\s+)?([\w-]+)", low)
        if dm:
            deployment = dm.group(1)
        return _success([
            _step(1, "undo_rollout", {
                "namespace": namespace, "deployment_name": deployment,
            }),
        ])

    # send notification
    if "notify" in low or "alert" in low or "notification" in low or "send" in low:
        channel = "#incidents"
        cm = re.search(r"#([\w-]+)", msg)
        if cm:
            channel = f"#{cm.group(1)}"
        severity = "info"
        if "critical" in low:
            severity = "critical"
        elif "warning" in low:
            severity = "warning"
        return _success([
            _step(1, "send_notification", {
                "channel": channel, "severity": severity,
                "message": msg[:200],
            }),
        ])

    # request approval
    if "approval" in low or "approve" in low:
        channel = "#ops"
        cm = re.search(r"#([\w-]+)", msg)
        if cm:
            channel = f"#{cm.group(1)}"
        return _success([
            _step(1, "request_approval", {
                "channel": channel,
                "prompt_message": "Requesting approval",
            }),
        ])

    # trigger webhook
    if "webhook" in low or "trigger" in low:
        url = "https://example.com/webhook"
        url_match = re.search(r"https?://\S+", msg)
        if url_match:
            url = url_match.group(0)
        method = "POST"
        if "get" in low:
            method = "GET"
        return _success([
            _step(1, "trigger_webhook", {
                "url": url, "method": method,
                "payload": {},
            }),
        ])

    return _success([
        _step(1, "send_notification", {
            "channel": "#general", "severity": "info",
            "message": f"Automated response: {msg[:100]}",
        }),
    ])


# ---------------------------------------------------------------------------
# endpoint
# ---------------------------------------------------------------------------

@app.post("/generate_workflow")
async def generate_workflow(body: GenerateRequest):
    return _match_prompt(body.user_message)

@app.get("/health")
async def health():
    return {"status": "healthy", "vllm_url": "mock"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
