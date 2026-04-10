"""Shared fixtures for generatorModelAPI tests."""

from __future__ import annotations

import json
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Valid workflow fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def valid_scale_workflow() -> dict:
    return {
        "workflow": {
            "steps": [
                {
                    "step_id": 1,
                    "tool": "scale_deployment",
                    "params": {
                        "namespace": "production",
                        "deployment_name": "fraud-model",
                        "replicas": 5,
                    },
                }
            ]
        }
    }


@pytest.fixture()
def valid_multi_step_workflow() -> dict:
    """A realistic 4-step memory-leak remediation workflow."""
    return {
        "workflow": {
            "steps": [
                {
                    "step_id": 1,
                    "tool": "send_notification",
                    "params": {
                        "channel": "#mlops-alerts",
                        "message": "Memory leak detected in recommendation-pod. Initiating restart.",
                        "severity": "warning",
                    },
                },
                {
                    "step_id": 2,
                    "tool": "request_approval",
                    "params": {
                        "channel": "#mlops-approvals",
                        "prompt_message": "Confirm rolling restart of recommendation-pod in production?",
                    },
                },
                {
                    "step_id": 3,
                    "tool": "restart_rollout",
                    "params": {
                        "namespace": "production",
                        "deployment_name": "recommendation-pod",
                    },
                },
                {
                    "step_id": 4,
                    "tool": "send_notification",
                    "params": {
                        "channel": "#mlops-alerts",
                        "message": "Rolling restart complete. Memory utilization stabilized.",
                        "severity": "info",
                    },
                },
            ]
        }
    }


@pytest.fixture()
def valid_webhook_workflow() -> dict:
    return {
        "workflow": {
            "steps": [
                {
                    "step_id": 1,
                    "tool": "trigger_webhook",
                    "params": {
                        "url": "https://airflow.company.com/api/v1/dags/retrain/dagRuns",
                        "method": "POST",
                        "payload": {"conf": {"model": "fraud-v2", "priority": "high"}},
                    },
                }
            ]
        }
    }


# ---------------------------------------------------------------------------
# Mock vLLM response builder
# ---------------------------------------------------------------------------

def make_vllm_response(
    content: str,
    status_code: int = 200,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    """Build a dict matching the vLLM /v1/chat/completions response shape."""
    return {
        "status_code": status_code,
        "body": {
            "id": "cmpl-test",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "/models/fused_model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        },
    }