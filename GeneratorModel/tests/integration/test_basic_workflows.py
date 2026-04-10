"""
Basic single-tool workflow tests.

Each test sends a clear, unambiguous prompt that should trigger exactly one tool
from the 6-tool registry and verifies the proxy returns a valid workflow step
with the correct tool name and expected parameters.

Response format (from Sriram's proxy):
    {"success": true, "workflow": {"workflow": {"steps": [...]}}}
"""

import pytest
from .conftest import VALID_TOOLS, get_steps, send_request

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assert_success_response(status_code: int, data: dict):
    """Common assertions for a successful proxy response."""
    assert status_code == 200, f"Expected HTTP 200, got {status_code}"
    assert data.get("success") is True, f"Expected success=true, got: {data}"
    assert "workflow" in data, f"Missing 'workflow' key in response: {data}"
    assert "workflow" in data["workflow"], f"Missing nested 'workflow' key: {data}"
    assert "steps" in data["workflow"]["workflow"], f"Missing 'steps': {data}"
    assert len(get_steps(data)) >= 1, "Workflow has no steps"

def get_first_step(data: dict) -> dict:
    """Return the first step from a successful workflow response."""
    return get_steps(data)[0]

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestScaleDeployment:
    """Test that a clear scale request produces a scale_deployment step."""

    PROMPT = "Scale my fraud-model deployment to 5 replicas in the production namespace"

    def test_response_is_successful(self):
        status, data = send_request(self.PROMPT)
        assert_success_response(status, data)

    def test_tool_is_scale_deployment(self):
        _, data = send_request(self.PROMPT)
        step = get_first_step(data)
        assert step["tool"] == "scale_deployment"

    def test_params_present_and_correct(self):
        _, data = send_request(self.PROMPT)
        step = get_first_step(data)
        params = step["params"]
        assert "namespace" in params
        assert "deployment_name" in params
        assert "replicas" in params
        assert params["namespace"] == "production"
        assert "fraud" in params["deployment_name"].lower()
        assert isinstance(params["replicas"], int)
        assert params["replicas"] == 5

class TestRestartRollout:
    """Test that a restart request produces a restart_rollout step."""

    PROMPT = "Restart the api-server deployment in the default namespace"

    def test_response_is_successful(self):
        status, data = send_request(self.PROMPT)
        assert_success_response(status, data)

    def test_tool_is_restart_rollout(self):
        _, data = send_request(self.PROMPT)
        step = get_first_step(data)
        assert step["tool"] == "restart_rollout"

    def test_params_present_and_correct(self):
        _, data = send_request(self.PROMPT)
        step = get_first_step(data)
        params = step["params"]
        assert "namespace" in params
        assert "deployment_name" in params
        assert params["namespace"] == "default"
        assert "api-server" in params["deployment_name"].lower()

class TestUndoRollout:
    """Test that a rollback request produces an undo_rollout step."""

    PROMPT = "Rollback the recommendation-model deployment in staging"

    def test_response_is_successful(self):
        status, data = send_request(self.PROMPT)
        assert_success_response(status, data)

    def test_tool_is_undo_rollout(self):
        _, data = send_request(self.PROMPT)
        step = get_first_step(data)
        assert step["tool"] == "undo_rollout"

    def test_params_present_and_correct(self):
        _, data = send_request(self.PROMPT)
        step = get_first_step(data)
        params = step["params"]
        assert "namespace" in params
        assert "deployment_name" in params
        assert params["namespace"] == "staging"
        assert "recommendation" in params["deployment_name"].lower()

class TestSendNotification:
    """Test that a notification request produces a send_notification step."""

    PROMPT = "Send a critical alert to #incidents channel: fraud model latency exceeded 500ms"

    def test_response_is_successful(self):
        status, data = send_request(self.PROMPT)
        assert_success_response(status, data)

    def test_tool_is_send_notification(self):
        _, data = send_request(self.PROMPT)
        step = get_first_step(data)
        assert step["tool"] == "send_notification"

    def test_params_present_and_correct(self):
        """Params: channel, message, severity (per Sriram's SendNotificationParams)."""
        _, data = send_request(self.PROMPT)
        step = get_first_step(data)
        params = step["params"]
        assert "channel" in params
        assert "message" in params
        assert "severity" in params
        assert "incidents" in params["channel"].lower()
        assert params["severity"] == "critical"

class TestRequestApproval:
    """Test that an approval request produces a request_approval step."""

    PROMPT = "Ask for approval in #ops channel before proceeding with the rollback"

    def test_response_is_successful(self):
        status, data = send_request(self.PROMPT)
        assert_success_response(status, data)

    def test_tool_is_request_approval(self):
        _, data = send_request(self.PROMPT)
        step = get_first_step(data)
        assert step["tool"] == "request_approval"

    def test_params_present_and_correct(self):
        """Params: channel, prompt_message (per Sriram's RequestApprovalParams)."""
        _, data = send_request(self.PROMPT)
        step = get_first_step(data)
        params = step["params"]
        assert "channel" in params
        assert "prompt_message" in params
        assert "ops" in params["channel"].lower()

class TestTriggerWebhook:
    """Test that a webhook request produces a trigger_webhook step."""

    PROMPT = (
        "Trigger a retraining job by hitting "
        "https://airflow.company.com/api/v1/dags/retrain/dagRuns with a POST request"
    )

    def test_response_is_successful(self):
        status, data = send_request(self.PROMPT)
        assert_success_response(status, data)

    def test_tool_is_trigger_webhook(self):
        _, data = send_request(self.PROMPT)
        step = get_first_step(data)
        assert step["tool"] == "trigger_webhook"

    def test_params_present_and_correct(self):
        """Params: url, method, payload (per Sriram's TriggerWebhookParams)."""
        _, data = send_request(self.PROMPT)
        step = get_first_step(data)
        params = step["params"]
        assert "url" in params
        assert "method" in params
        assert "payload" in params
        assert "airflow" in params["url"].lower()
        assert params["method"].upper() == "POST"
        assert isinstance(params["payload"], dict)


# ---------------------------------------------------------------------------
# param value constraints
# ---------------------------------------------------------------------------

# Required param keys per tool
REQUIRED_PARAMS = {
    "scale_deployment": {"namespace", "deployment_name", "replicas"},
    "restart_rollout": {"namespace", "deployment_name"},
    "undo_rollout": {"namespace", "deployment_name"},
    "send_notification": {"channel", "message", "severity"},
    "request_approval": {"channel", "prompt_message"},
    "trigger_webhook": {"url", "method", "payload"},
}

VALID_SEVERITIES = {"info", "warning", "critical"}
VALID_HTTP_METHODS = {"GET", "POST", "PUT", "DELETE"}

class TestParamValueConstraints:
    """
    Verify that param values meet the Pydantic constraints,
    not just that the keys exist.

    Constraints from workflow.py:
      - All string fields have min_length=1 (non-empty)
      - replicas: int, >= 1
      - severity: Literal["info", "warning", "critical"]
      - method: Literal["GET", "POST", "PUT", "DELETE"]
      - url: must start with http:// or https://
    """
    def test_scale_replicas_is_positive_integer(self):
        _, data = send_request(
            "Scale my fraud-model deployment to 5 replicas in the production namespace"
        )
        step = get_first_step(data)
        replicas = step["params"]["replicas"]
        assert isinstance(replicas, int), f"replicas should be int, got {type(replicas)}"
        assert replicas >= 1, f"replicas must be >= 1, got {replicas}"

    def test_notification_severity_is_valid_literal(self):
        _, data = send_request(
            "Send a critical alert to #incidents channel: fraud model latency exceeded 500ms"
        )
        step = get_first_step(data)
        severity = step["params"]["severity"]
        assert severity in VALID_SEVERITIES, (
            f"severity '{severity}' not in {VALID_SEVERITIES}"
        )

    def test_webhook_method_is_valid_literal(self):
        _, data = send_request(
            "Trigger a retraining job by hitting "
            "https://airflow.company.com/api/v1/dags/retrain/dagRuns with a POST request"
        )
        step = get_first_step(data)
        method = step["params"]["method"]
        assert method in VALID_HTTP_METHODS, (
            f"method '{method}' not in {VALID_HTTP_METHODS}"
        )

    def test_webhook_url_starts_with_http(self):
        _, data = send_request(
            "Trigger a retraining job by hitting "
            "https://airflow.company.com/api/v1/dags/retrain/dagRuns with a POST request"
        )
        step = get_first_step(data)
        url = step["params"]["url"]
        assert url.startswith(("http://", "https://")), (
            f"url must start with http:// or https://, got '{url}'"
        )

    def test_webhook_payload_is_dict(self):
        _, data = send_request(
            "Trigger a retraining job by hitting "
            "https://airflow.company.com/api/v1/dags/retrain/dagRuns with a POST request"
        )
        step = get_first_step(data)
        payload = step["params"]["payload"]
        assert isinstance(payload, dict), (
            f"payload should be dict, got {type(payload)}"
        )

    def test_all_string_params_are_non_empty(self):
        """Every string-valued param across all 6 tools must be non-empty."""
        prompts = {
            "scale_deployment": "Scale my fraud-model deployment to 5 replicas in production",
            "restart_rollout": "Restart the api-server deployment in the default namespace",
            "undo_rollout": "Rollback the recommendation-model deployment in staging",
            "send_notification": "Send a critical alert to #incidents channel: latency exceeded 500ms",
            "request_approval": "Ask for approval in #ops channel before proceeding with the rollback",
            "trigger_webhook": "Trigger a job by hitting https://airflow.company.com/run with a POST request",
        }
        for tool_name, prompt in prompts.items():
            _, data = send_request(prompt)
            step = get_first_step(data)
            for key, val in step["params"].items():
                if isinstance(val, str):
                    assert len(val) >= 1, (
                        f"Tool '{tool_name}', param '{key}' is empty string"
                    )


class TestRequiredParamKeys:
    """
    Verify each tool returns all the required params, no missing keys.
    """

    TOOL_PROMPTS = [
        ("scale_deployment", "Scale my fraud-model deployment to 5 replicas in production"),
        ("restart_rollout", "Restart the api-server deployment in the default namespace"),
        ("undo_rollout", "Rollback the recommendation-model deployment in staging"),
        ("send_notification", "Send a critical alert to #incidents channel: latency exceeded 500ms"),
        ("request_approval", "Ask for approval in #ops channel before proceeding with the rollback"),
        ("trigger_webhook", "Trigger a job by hitting https://airflow.company.com/run with a POST request"),
    ]

    @pytest.mark.parametrize("tool_name,prompt", TOOL_PROMPTS)
    def test_all_required_keys_present(self, tool_name, prompt):
        _, data = send_request(prompt)
        step = get_first_step(data)
        actual_keys = set(step["params"].keys())
        required = REQUIRED_PARAMS[tool_name]
        missing = required - actual_keys
        assert not missing, (
            f"Tool '{tool_name}' is missing required params: {missing}. "
            f"Got: {actual_keys}"
        )

class TestSingleStepStructure:
    """Verify structural properties specific to single-step workflows."""

    PROMPTS = [
        "Scale my fraud-model deployment to 5 replicas in production",
        "Restart the api-server deployment in the default namespace",
        "Rollback the recommendation-model deployment in staging",
        "Send a critical alert to #incidents channel: latency exceeded 500ms",
        "Ask for approval in #ops channel before proceeding with the rollback",
        "Trigger a job by hitting https://airflow.company.com/run with a POST request",
    ]

    @pytest.mark.parametrize("prompt", PROMPTS)
    def test_step_id_starts_at_one(self, prompt):
        _, data = send_request(prompt)
        step = get_first_step(data)
        assert step["step_id"] == 1, (
            f"Single-step workflow should have step_id=1, got {step['step_id']}"
        )

    @pytest.mark.parametrize("prompt", PROMPTS)
    def test_step_has_only_expected_keys(self, prompt):
        """Each step should only have step_id, tool, params — no extra keys."""
        _, data = send_request(prompt)
        step = get_first_step(data)
        allowed_keys = {"step_id", "tool", "params"}
        extra = set(step.keys()) - allowed_keys
        assert not extra, (
            f"Step has unexpected keys: {extra}. Allowed: {allowed_keys}"
        )

    @pytest.mark.parametrize("prompt", PROMPTS)
    def test_step_id_is_integer(self, prompt):
        _, data = send_request(prompt)
        step = get_first_step(data)
        assert isinstance(step["step_id"], int), (
            f"step_id should be int, got {type(step['step_id'])}"
        )

    @pytest.mark.parametrize("prompt", PROMPTS)
    def test_tool_is_string(self, prompt):
        _, data = send_request(prompt)
        step = get_first_step(data)
        assert isinstance(step["tool"], str), (
            f"tool should be str, got {type(step['tool'])}"
        )

    @pytest.mark.parametrize("prompt", PROMPTS)
    def test_params_is_dict(self, prompt):
        _, data = send_request(prompt)
        step = get_first_step(data)
        assert isinstance(step["params"], dict), (
            f"params should be dict, got {type(step['params'])}"
        )

class TestScaleDeploymentVariations:
    """Test scale_deployment with different replica counts and namespaces."""

    @pytest.mark.parametrize("prompt,expected_ns,expected_replicas", [
        ("Scale fraud-model to 3 replicas in staging", "staging", 3),
        ("Scale api-server to 1 replica in default namespace", "default", 1),
        ("Scale fraud-model to 5 replicas in the production namespace", "production", 5),
    ])
    def test_namespace_and_replicas(self, prompt, expected_ns, expected_replicas):
        _, data = send_request(prompt)
        step = get_first_step(data)
        assert step["tool"] == "scale_deployment"
        assert step["params"]["namespace"] == expected_ns
        assert step["params"]["replicas"] == expected_replicas

    def test_large_replica_count(self):
        """A large but valid replica count should still produce a valid response."""
        _, data = send_request(
            "Scale fraud-model to 100 replicas in production"
        )
        step = get_first_step(data)
        assert step["tool"] == "scale_deployment"
        assert step["params"]["replicas"] == 100
        assert isinstance(step["params"]["replicas"], int)

class TestNotificationSeverityVariations:
    """Test send_notification with all 3 severity levels."""

    @pytest.mark.parametrize("severity_word,expected", [
        ("critical", "critical"),
        ("warning", "warning"),
    ])
    def test_severity_levels(self, severity_word, expected):
        _, data = send_request(
            f"Send a {severity_word} alert to #incidents channel: something happened"
        )
        step = get_first_step(data)
        assert step["tool"] == "send_notification"
        assert step["params"]["severity"] == expected

class TestAllToolsUseValidNames:
    """Verify every tool name returned across the basic prompts is in the registry."""

    PROMPTS = [
        "Scale my fraud-model deployment to 5 replicas in the production namespace",
        "Restart the api-server deployment in the default namespace",
        "Rollback the recommendation-model deployment in staging",
        "Send a critical alert to #incidents channel: fraud model latency exceeded 500ms",
        "Ask for approval in #ops channel before proceeding with the rollback",
        "Trigger a retraining job by hitting https://airflow.company.com/api/v1/dags/retrain/dagRuns with a POST request",
    ]

    @pytest.mark.parametrize("prompt", PROMPTS)
    def test_tool_name_in_registry(self, prompt):
        _, data = send_request(prompt)
        for step in get_steps(data):
            assert step["tool"] in VALID_TOOLS, (
                f"Tool '{step['tool']}' is not in the valid registry: {VALID_TOOLS}"
            )
