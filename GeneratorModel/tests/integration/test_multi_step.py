"""
Multi-step workflow tests.

Verifies that the LLM can chain multiple tools together in a single workflow
when the prompt describes a sequence of actions.

Response format: {"success": true, "workflow": {"workflow": {"steps": [...]}}}
"""

import pytest
from .conftest import VALID_TOOLS, get_steps, send_request

def assert_success_response(status_code: int, data: dict):
    assert status_code == 200, f"Expected HTTP 200, got {status_code}"
    assert data.get("success") is True, f"Expected success=true, got: {data}"
    assert "workflow" in data
    assert "workflow" in data["workflow"]
    assert "steps" in data["workflow"]["workflow"]

def tool_names(steps: list) -> list:
    return [s["tool"] for s in steps]

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMemoryLeakIncident:
    """
    Prompt: Memory leak detected in recommendation-pod. Notify the team,
    ask for approval, restart the deployment, and send a confirmation after.

    Expected 4 steps: send_notification, request_approval, restart_rollout,
    send_notification.
    """

    PROMPT = (
        "Memory leak detected in recommendation-pod. Notify the team on #incidents, "
        "ask for approval in #ops, then restart the deployment in production, "
        "and send a confirmation notification to #incidents after."
    )
    EXPECTED_TOOLS = {"send_notification", "request_approval", "restart_rollout"}
    EXPECTED_COUNT = 4

    def test_response_is_successful(self):
        status, data = send_request(self.PROMPT)
        assert_success_response(status, data)

    def test_step_count(self):
        _, data = send_request(self.PROMPT)
        steps = get_steps(data)
        assert abs(len(steps) - self.EXPECTED_COUNT) <= 1, (
            f"Expected ~{self.EXPECTED_COUNT} steps, got {len(steps)}: {tool_names(steps)}"
        )

    def test_expected_tools_present(self):
        _, data = send_request(self.PROMPT)
        names = set(tool_names(get_steps(data)))
        for tool in self.EXPECTED_TOOLS:
            assert tool in names, (
                f"Expected tool '{tool}' not found in steps. Got: {names}"
            )

    def test_all_tools_in_registry(self):
        _, data = send_request(self.PROMPT)
        for step in get_steps(data):
            assert step["tool"] in VALID_TOOLS

    def test_step_ids_are_sequential(self):
        _, data = send_request(self.PROMPT)
        steps = get_steps(data)
        ids = [s["step_id"] for s in steps]
        assert ids == list(range(1, len(ids) + 1)), (
            f"Step IDs are not sequential starting from 1: {ids}"
        )

class TestScaleAndNotify:
    """
    Prompt: Scale fraud-model to 10 replicas and notify #scaling channel.

    Expected 2 steps: scale_deployment, send_notification.
    """

    PROMPT = "Scale fraud-model to 10 replicas in production and notify #scaling channel"
    EXPECTED_TOOLS = {"scale_deployment", "send_notification"}
    EXPECTED_COUNT = 2

    def test_response_is_successful(self):
        status, data = send_request(self.PROMPT)
        assert_success_response(status, data)

    def test_step_count(self):
        _, data = send_request(self.PROMPT)
        steps = get_steps(data)
        assert abs(len(steps) - self.EXPECTED_COUNT) <= 1, (
            f"Expected ~{self.EXPECTED_COUNT} steps, got {len(steps)}: {tool_names(steps)}"
        )

    def test_expected_tools_present(self):
        _, data = send_request(self.PROMPT)
        names = set(tool_names(get_steps(data)))
        for tool in self.EXPECTED_TOOLS:
            assert tool in names, (
                f"Expected tool '{tool}' not found in steps. Got: {names}"
            )

    def test_scale_params(self):
        _, data = send_request(self.PROMPT)
        scale_steps = [s for s in get_steps(data) if s["tool"] == "scale_deployment"]
        assert len(scale_steps) >= 1
        params = scale_steps[0]["params"]
        assert params.get("replicas") == 10


class TestRollbackNotifyWebhook:
    """
    Prompt: Rollback api-server in production, notify #incidents as critical,
    and trigger the smoke test webhook.

    Expected 3 steps: undo_rollout, send_notification, trigger_webhook.
    """

    PROMPT = (
        "Rollback api-server in production, notify #incidents as critical, "
        "and trigger the smoke test webhook at https://tests.company.com/run"
    )
    EXPECTED_TOOLS = {"undo_rollout", "send_notification", "trigger_webhook"}
    EXPECTED_COUNT = 3

    def test_response_is_successful(self):
        status, data = send_request(self.PROMPT)
        assert_success_response(status, data)

    def test_step_count(self):
        _, data = send_request(self.PROMPT)
        steps = get_steps(data)
        assert abs(len(steps) - self.EXPECTED_COUNT) <= 1, (
            f"Expected ~{self.EXPECTED_COUNT} steps, got {len(steps)}: {tool_names(steps)}"
        )

    def test_expected_tools_present(self):
        _, data = send_request(self.PROMPT)
        names = set(tool_names(get_steps(data)))
        for tool in self.EXPECTED_TOOLS:
            assert tool in names, (
                f"Expected tool '{tool}' not found in steps. Got: {names}"
            )

    def test_webhook_url(self):
        _, data = send_request(self.PROMPT)
        wh_steps = [s for s in get_steps(data) if s["tool"] == "trigger_webhook"]
        assert len(wh_steps) >= 1
        assert "tests.company.com" in wh_steps[0]["params"].get("url", "")

    def test_all_tools_in_registry(self):
        _, data = send_request(self.PROMPT)
        for step in get_steps(data):
            assert step["tool"] in VALID_TOOLS

# required param keys per tool
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


class TestComplexFiveStepWorkflow:
    """
    A complex incident requiring 5 steps:
    notify -> approve -> restart -> webhook -> notify.
    """

    PROMPT = (
        "Critical outage on payment-service in production. "
        "Notify #incidents as critical, request approval in #ops, "
        "restart the deployment, trigger a health check webhook at "
        "https://health.company.com/check, then send a confirmation to #incidents."
    )
    EXPECTED_TOOLS = {
        "send_notification", "request_approval", "restart_rollout", "trigger_webhook",
    }
    EXPECTED_COUNT = 5

    def test_response_is_successful(self):
        status, data = send_request(self.PROMPT)
        assert_success_response(status, data)

    def test_step_count(self):
        _, data = send_request(self.PROMPT)
        steps = get_steps(data)
        assert abs(len(steps) - self.EXPECTED_COUNT) <= 1, (
            f"Expected ~{self.EXPECTED_COUNT} steps, got {len(steps)}: {tool_names(steps)}"
        )

    def test_expected_tools_present(self):
        _, data = send_request(self.PROMPT)
        names = set(tool_names(get_steps(data)))
        for tool in self.EXPECTED_TOOLS:
            assert tool in names, (
                f"Expected tool '{tool}' not found. Got: {names}"
            )

    def test_step_ids_are_sequential(self):
        _, data = send_request(self.PROMPT)
        steps = get_steps(data)
        ids = [s["step_id"] for s in steps]
        assert ids == list(range(1, len(ids) + 1)), (
            f"Step IDs not sequential from 1: {ids}"
        )

    def test_all_tools_in_registry(self):
        _, data = send_request(self.PROMPT)
        for step in get_steps(data):
            assert step["tool"] in VALID_TOOLS

    def test_webhook_step_has_valid_url(self):
        _, data = send_request(self.PROMPT)
        wh_steps = [s for s in get_steps(data) if s["tool"] == "trigger_webhook"]
        assert len(wh_steps) >= 1, "No trigger_webhook step found"
        url = wh_steps[0]["params"].get("url", "")
        assert url.startswith(("http://", "https://")), (
            f"Webhook URL must start with http(s)://, got '{url}'"
        )

class TestMultiStepParamValidation:
    """
    Verify that every step in a multi-step workflow has all the required params
    for its tool type, with correct types.
    """

    PROMPTS = [
        (
            "Memory leak detected in recommendation-pod. Notify the team on #incidents, "
            "ask for approval in #ops, then restart the deployment in production, "
            "and send a confirmation notification to #incidents after."
        ),
        (
            "Scale fraud-model to 10 replicas in production and notify #scaling channel"
        ),
        (
            "Rollback api-server in production, notify #incidents as critical, "
            "and trigger the smoke test webhook at https://tests.company.com/run"
        ),
    ]

    @pytest.mark.parametrize("prompt", PROMPTS)
    def test_every_step_has_all_required_params(self, prompt):
        _, data = send_request(prompt)
        for step in get_steps(data):
            tool = step["tool"]
            actual_keys = set(step["params"].keys())
            required = REQUIRED_PARAMS[tool]
            missing = required - actual_keys
            assert not missing, (
                f"Step {step['step_id']} (tool='{tool}') missing params: {missing}. "
                f"Got: {actual_keys}"
            )

    @pytest.mark.parametrize("prompt", PROMPTS)
    def test_string_params_are_non_empty(self, prompt):
        _, data = send_request(prompt)
        for step in get_steps(data):
            for key, val in step["params"].items():
                if isinstance(val, str):
                    assert len(val) >= 1, (
                        f"Step {step['step_id']} param '{key}' is empty string"
                    )

    @pytest.mark.parametrize("prompt", PROMPTS)
    def test_notification_severity_is_valid(self, prompt):
        _, data = send_request(prompt)
        for step in get_steps(data):
            if step["tool"] == "send_notification":
                sev = step["params"].get("severity")
                assert sev in VALID_SEVERITIES, (
                    f"Step {step['step_id']} severity '{sev}' not in {VALID_SEVERITIES}"
                )

    @pytest.mark.parametrize("prompt", PROMPTS)
    def test_webhook_method_is_valid(self, prompt):
        _, data = send_request(prompt)
        for step in get_steps(data):
            if step["tool"] == "trigger_webhook":
                method = step["params"].get("method")
                assert method in VALID_HTTP_METHODS, (
                    f"Step {step['step_id']} method '{method}' not in {VALID_HTTP_METHODS}"
                )

    @pytest.mark.parametrize("prompt", PROMPTS)
    def test_scale_replicas_is_positive_int(self, prompt):
        _, data = send_request(prompt)
        for step in get_steps(data):
            if step["tool"] == "scale_deployment":
                r = step["params"].get("replicas")
                assert isinstance(r, int), f"replicas should be int, got {type(r)}"
                assert r >= 1, f"replicas must be >= 1, got {r}"


class TestStepOrdering:
    """
    Verifying logical ordering, approval should come before destructive actions
    like restart or rollback.
    """

    def test_approval_before_restart(self):
        _, data = send_request(
            "Memory leak detected in recommendation-pod. Notify the team on #incidents, "
            "ask for approval in #ops, then restart the deployment in production, "
            "and send a confirmation notification to #incidents after."
        )
        steps = get_steps(data)
        names = tool_names(steps)
        if "request_approval" in names and "restart_rollout" in names:
            approval_idx = names.index("request_approval")
            restart_idx = names.index("restart_rollout")
            assert approval_idx < restart_idx, (
                f"request_approval (step {approval_idx + 1}) should come before "
                f"restart_rollout (step {restart_idx + 1}). Order: {names}"
            )

    def test_notification_appears_first_in_incident(self):
        """In an incident response, the first step should typically be a notification."""
        _, data = send_request(
            "Critical outage on payment-service in production. "
            "Notify #incidents as critical, request approval in #ops, "
            "restart the deployment, trigger a health check webhook at "
            "https://health.company.com/check, then send a confirmation to #incidents."
        )
        steps = get_steps(data)
        assert steps[0]["tool"] == "send_notification", (
            f"First step should be send_notification, got '{steps[0]['tool']}'"
        )

    def test_no_duplicate_step_ids(self):
        """Step IDs must never be duplicated."""
        _, data = send_request(
            "Notify #ops, get approval, then restart api-server in staging"
        )
        steps = get_steps(data)
        ids = [s["step_id"] for s in steps]
        assert len(ids) == len(set(ids)), f"Duplicate step_ids found: {ids}"

class TestStepStructure:
    """Verify structural properties common to all the multi-step workflows."""

    PROMPTS = [
        "Notify #ops, get approval, then restart api-server in staging",
        "Scale fraud-model to 3 replicas in production and notify #scaling channel",
    ]

    @pytest.mark.parametrize("prompt", PROMPTS)
    def test_every_step_has_required_keys(self, prompt):
        _, data = send_request(prompt)
        for step in get_steps(data):
            assert "step_id" in step, f"Step missing step_id: {step}"
            assert "tool" in step, f"Step missing tool: {step}"
            assert "params" in step, f"Step missing params: {step}"

    @pytest.mark.parametrize("prompt", PROMPTS)
    def test_params_are_dicts(self, prompt):
        _, data = send_request(prompt)
        for step in get_steps(data):
            assert isinstance(step["params"], dict), (
                f"params should be a dict, got {type(step['params'])}"
            )
