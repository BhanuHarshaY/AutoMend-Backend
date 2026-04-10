"""
Integration tests for the /generate_workflow proxy endpoint.

All tests mock the vLLM HTTP call — no GPU or running vLLM instance needed.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests.conftest import make_vllm_response

client = TestClient(app)

ENDPOINT = "/generate_workflow"


def _mock_vllm_post(content: str, status_code: int = 200, finish_reason: str = "stop"):
    """Return a mock that simulates httpx.AsyncClient.post → vLLM response."""
    vllm = make_vllm_response(content, status_code, finish_reason)
    mock_response = httpx.Response(
        status_code=vllm["status_code"],
        json=vllm["body"],
        request=httpx.Request("POST", "http://mock:8001/v1/chat/completions"),
    )
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client


# ===================================================================
# Happy path
# ===================================================================

class TestHappyPath:
    def test_valid_single_step(self):
        llm_output = json.dumps({
            "workflow": {
                "steps": [{
                    "step_id": 1,
                    "tool": "scale_deployment",
                    "params": {
                        "namespace": "production",
                        "deployment_name": "fraud-model",
                        "replicas": 5,
                    },
                }]
            }
        })
        with patch("app.main.httpx.AsyncClient", return_value=_mock_vllm_post(llm_output)):
            resp = client.post(ENDPOINT, json={"user_message": "Scale fraud model to 5"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["workflow"]["workflow"]["steps"][0]["tool"] == "scale_deployment"
        assert body["error"] is None

    def test_valid_multi_step(self):
        llm_output = json.dumps({
            "workflow": {
                "steps": [
                    {
                        "step_id": 1,
                        "tool": "send_notification",
                        "params": {"channel": "#ops", "message": "Starting restart", "severity": "warning"},
                    },
                    {
                        "step_id": 2,
                        "tool": "restart_rollout",
                        "params": {"namespace": "prod", "deployment_name": "api"},
                    },
                ]
            }
        })
        with patch("app.main.httpx.AsyncClient", return_value=_mock_vllm_post(llm_output)):
            resp = client.post(ENDPOINT, json={"user_message": "Restart API server"})

        body = resp.json()
        assert body["success"] is True
        assert len(body["workflow"]["workflow"]["steps"]) == 2

    def test_with_system_context(self):
        llm_output = json.dumps({
            "workflow": {
                "steps": [{
                    "step_id": 1,
                    "tool": "undo_rollout",
                    "params": {"namespace": "prod", "deployment_name": "ml-v2"},
                }]
            }
        })
        with patch("app.main.httpx.AsyncClient", return_value=_mock_vllm_post(llm_output)):
            resp = client.post(ENDPOINT, json={
                "user_message": "Rollback ml-v2",
                "system_context": "Model ml-v2 has 30% error rate since last deploy",
            })

        assert resp.json()["success"] is True


# ===================================================================
# Guardrails repair
# ===================================================================

class TestGuardrailsRepair:
    def test_markdown_wrapped_output(self):
        inner = json.dumps({
            "workflow": {
                "steps": [{
                    "step_id": 1,
                    "tool": "restart_rollout",
                    "params": {"namespace": "default", "deployment_name": "api-server"},
                }]
            }
        })
        llm_output = f"```json\n{inner}\n```"
        with patch("app.main.httpx.AsyncClient", return_value=_mock_vllm_post(llm_output)):
            resp = client.post(ENDPOINT, json={"user_message": "Restart api-server"})

        assert resp.json()["success"] is True

    def test_trailing_comma_repaired(self):
        llm_output = (
            '{"workflow": {"steps": [{"step_id": 1, "tool": "restart_rollout", '
            '"params": {"namespace": "prod", "deployment_name": "api"},}]}}'
        )
        with patch("app.main.httpx.AsyncClient", return_value=_mock_vllm_post(llm_output)):
            resp = client.post(ENDPOINT, json={"user_message": "Restart"})

        assert resp.json()["success"] is True

    def test_missing_closing_bracket_repaired(self):
        llm_output = (
            '{"workflow": {"steps": [{"step_id": 1, "tool": "scale_deployment", '
            '"params": {"namespace": "prod", "deployment_name": "fraud", "replicas": 5}}]}'
        )
        with patch("app.main.httpx.AsyncClient", return_value=_mock_vllm_post(llm_output)):
            resp = client.post(ENDPOINT, json={"user_message": "Scale fraud"})

        assert resp.json()["success"] is True


# ===================================================================
# Schema validation failures (LLM hallucinations)
# ===================================================================

class TestSchemaValidation:
    def test_hallucinated_tool_name(self):
        llm_output = json.dumps({
            "workflow": {
                "steps": [{
                    "step_id": 1,
                    "tool": "replicas",
                    "params": {"deployment": "my-dep", "replicas": 5, "pod": None},
                }]
            }
        })
        with patch("app.main.httpx.AsyncClient", return_value=_mock_vllm_post(llm_output)):
            resp = client.post(ENDPOINT, json={"user_message": "Do something"})

        body = resp.json()
        assert body["success"] is False
        assert body["error"] == "Validation failed"
        assert "tool" in body["details"]
        assert "scale_deployment" in body["details"]  # lists valid tools
        assert body["raw_output"] is not None

    def test_wrong_param_type(self):
        llm_output = json.dumps({
            "workflow": {
                "steps": [{
                    "step_id": 1,
                    "tool": "scale_deployment",
                    "params": {
                        "namespace": "prod",
                        "deployment_name": "fraud",
                        "replicas": "five",
                    },
                }]
            }
        })
        with patch("app.main.httpx.AsyncClient", return_value=_mock_vllm_post(llm_output)):
            resp = client.post(ENDPOINT, json={"user_message": "Scale to five"})

        body = resp.json()
        assert body["success"] is False
        assert body["error"] == "Validation failed"

    def test_empty_steps(self):
        llm_output = json.dumps({"workflow": {"steps": []}})
        with patch("app.main.httpx.AsyncClient", return_value=_mock_vllm_post(llm_output)):
            resp = client.post(ENDPOINT, json={"user_message": "Help"})

        body = resp.json()
        assert body["success"] is False
        assert body["error"] == "Validation failed"

    def test_non_sequential_step_ids(self):
        llm_output = json.dumps({
            "workflow": {
                "steps": [
                    {
                        "step_id": 1,
                        "tool": "restart_rollout",
                        "params": {"namespace": "p", "deployment_name": "d"},
                    },
                    {
                        "step_id": 5,
                        "tool": "restart_rollout",
                        "params": {"namespace": "p", "deployment_name": "d"},
                    },
                ]
            }
        })
        with patch("app.main.httpx.AsyncClient", return_value=_mock_vllm_post(llm_output)):
            resp = client.post(ENDPOINT, json={"user_message": "Restart twice"})

        body = resp.json()
        assert body["success"] is False
        assert "sequential" in body["details"]


# ===================================================================
# JSON parsing failures
# ===================================================================

class TestJsonFailures:
    def test_total_garbage(self):
        llm_output = "I'm sorry, I cannot generate a workflow for that request."
        with patch("app.main.httpx.AsyncClient", return_value=_mock_vllm_post(llm_output)):
            resp = client.post(ENDPOINT, json={"user_message": "Tell me a joke"})

        body = resp.json()
        assert body["success"] is False
        assert body["error"] == "JSON parsing failed"
        assert body["raw_output"] == llm_output

    def test_deeply_truncated(self):
        llm_output = '{"workflow": {"steps": [{"step_id": 1, "tool": "sca'
        with patch("app.main.httpx.AsyncClient", return_value=_mock_vllm_post(llm_output, finish_reason="length")):
            resp = client.post(ENDPOINT, json={"user_message": "Scale"})

        body = resp.json()
        # Either fails to parse or fails schema validation — either way, not success
        assert body["success"] is False

    def test_empty_llm_output(self):
        with patch("app.main.httpx.AsyncClient", return_value=_mock_vllm_post("")):
            resp = client.post(ENDPOINT, json={"user_message": "Empty"})

        body = resp.json()
        assert body["success"] is False
        assert body["error"] == "JSON parsing failed"


# ===================================================================
# vLLM connection failures
# ===================================================================

class TestVllmErrors:
    def test_connection_refused(self):
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("app.main.httpx.AsyncClient", return_value=mock_client):
            resp = client.post(ENDPOINT, json={"user_message": "Scale it"})

        body = resp.json()
        assert body["success"] is False
        assert body["error"] == "vLLM connection failed"

    def test_timeout(self):
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("app.main.httpx.AsyncClient", return_value=mock_client):
            resp = client.post(ENDPOINT, json={"user_message": "Scale it"})

        body = resp.json()
        assert body["success"] is False
        assert body["error"] == "vLLM request timed out"

    def test_vllm_500(self):
        mock_response = httpx.Response(
            status_code=500,
            text="Internal Server Error",
            request=httpx.Request("POST", "http://mock:8001/v1/chat/completions"),
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("app.main.httpx.AsyncClient", return_value=mock_client):
            resp = client.post(ENDPOINT, json={"user_message": "Scale it"})

        body = resp.json()
        assert body["success"] is False
        assert "500" in body["error"]

    def test_vllm_404_wrong_model(self):
        mock_response = httpx.Response(
            status_code=404,
            text="Model not found",
            request=httpx.Request("POST", "http://mock:8001/v1/chat/completions"),
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("app.main.httpx.AsyncClient", return_value=mock_client):
            resp = client.post(ENDPOINT, json={"user_message": "Scale it"})

        body = resp.json()
        assert body["success"] is False
        assert "404" in body["error"]


# ===================================================================
# Request validation
# ===================================================================

class TestRequestValidation:
    def test_missing_user_message(self):
        resp = client.post(ENDPOINT, json={})
        assert resp.status_code == 422

    def test_empty_user_message(self):
        resp = client.post(ENDPOINT, json={"user_message": ""})
        assert resp.status_code == 422

    def test_health_endpoint(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"