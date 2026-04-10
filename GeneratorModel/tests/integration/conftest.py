"""Shared fixtures and configuration for the integration test suite.

All tests hit the validation proxy at /generate_workflow (port 8002).
Set PROXY_URL env var to point at a remote instance for GCP testing.
"""

import os

import pytest
import requests

PROXY_URL = os.getenv("PROXY_URL", "http://localhost:8002")
GENERATE_ENDPOINT = f"{PROXY_URL}/generate_workflow"

# the 6 valid tools in the AutoMend registry
VALID_TOOLS = frozenset([
    "scale_deployment",
    "restart_rollout",
    "undo_rollout",
    "send_notification",
    "request_approval",
    "trigger_webhook",
])

def send_request(user_message: str, system_context: str | None = None, timeout: int = 120):
    """
    Sends a request to the proxy and returns (status_code, response_json).
    """
    payload = {"user_message": user_message}
    if system_context is not None:
        payload["system_context"] = system_context
    resp = requests.post(GENERATE_ENDPOINT, json=payload, timeout=timeout)
    return resp.status_code, resp.json()


def get_steps(data: dict) -> list:
    """
    Extracts the steps list from a successful response.
    """
    return data["workflow"]["workflow"]["steps"]

@pytest.fixture()
def proxy_url():
    """Expose the proxy URL to tests that need it."""
    return PROXY_URL

@pytest.fixture()
def generate_endpoint():
    """Expose the full generate endpoint URL."""
    return GENERATE_ENDPOINT
