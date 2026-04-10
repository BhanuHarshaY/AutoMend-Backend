"""
System prompt and determinism tests.

Verifies that the proxy correctly injects the tool registry system prompt
and that the LLM only produces tools from the allowed set. Also tests
that temperature=0 produces deterministic outputs.
"""

import pytest
from .conftest import VALID_TOOLS, get_steps, send_request

# tools that the LLM might hallucinate but are not in the registry
INVALID_TOOLS = {
    "kubectl",
    "docker",
    "restart_pod",
    "scale",
    "rollback",
    "deploy",
    "exec",
    "ssh",
    "run_command",
    "delete_pod",
    "create_deployment",
}

class TestToolRegistryEnforcement:
    """Every tool name returned by the model must be in the 6-tool registry."""

    PROMPTS = [
        "Scale my fraud-model deployment to 5 replicas in production",
        "Restart the api-server deployment in the default namespace",
        "Rollback the recommendation-model in staging",
        "Send a critical alert to #incidents channel",
        "Ask for approval in #ops channel before proceeding",
        "Trigger a webhook at https://airflow.company.com/api/v1/dags/retrain/dagRuns",
        "Notify the team, get approval, then restart api-server in production",
    ]

    @pytest.mark.parametrize("prompt", PROMPTS)
    def test_no_hallucinated_tools(self, prompt):
        _, data = send_request(prompt)
        if not data.get("success"):
            pytest.skip("Request did not succeed, skipping tool validation")

        steps = get_steps(data)
        for step in steps:
            tool = step.get("tool", "")
            assert tool in VALID_TOOLS, (
                f"Hallucinated tool '{tool}' is not in the valid registry.\n"
                f"Valid tools: {VALID_TOOLS}\n"
                f"Full step: {step}"
            )
            assert tool not in INVALID_TOOLS, (
                f"Model returned known-invalid tool '{tool}'"
            )

# NOTE: These determinism tests pass trivially against the mock server because
# mock_proxy.py uses regex pattern matching and always returns the same response
# for the same prompt. They are only meaningful when PROXY_URL points at a real
# vLLM instance (temperature=0 should produce identical outputs across runs).
class TestDeterminism:
    """
    With temperature=0, the same prompt should produce identical workflows.
    sending the same request 3 times and comparing outputs.
    """

    PROMPT = "Scale fraud-model to 5 replicas in production"
    NUM_REQUESTS = 3

    def test_identical_responses(self):
        responses = []
        for _ in range(self.NUM_REQUESTS):
            _, data = send_request(self.PROMPT)
            if not data.get("success"):
                pytest.skip("Request did not succeed, cannot test determinism")
            responses.append(data)

        # compares the workflow steps across all the responses
        first_steps = get_steps(responses[0])
        for i, resp in enumerate(responses[1:], start=2):
            other_steps = get_steps(resp)
            assert len(first_steps) == len(other_steps), (
                f"Response 1 has {len(first_steps)} steps but response {i} "
                f"has {len(other_steps)} steps"
            )
            for j, (s1, s2) in enumerate(zip(first_steps, other_steps)):
                assert s1["tool"] == s2["tool"], (
                    f"Step {j+1}: response 1 uses '{s1['tool']}' but "
                    f"response {i} uses '{s2['tool']}'"
                )

    def test_tool_order_is_consistent(self):
        tool_sequences = []
        for _ in range(self.NUM_REQUESTS):
            _, data = send_request(self.PROMPT)
            if not data.get("success"):
                pytest.skip("Request did not succeed")
            tools = [s["tool"] for s in get_steps(data)]
            tool_sequences.append(tools)

        for i, seq in enumerate(tool_sequences[1:], start=2):
            assert tool_sequences[0] == seq, (
                f"Tool order differs: response 1 = {tool_sequences[0]}, "
                f"response {i} = {seq}"
            )

# required param keys per tool
REQUIRED_PARAMS = {
    "scale_deployment": {"namespace", "deployment_name", "replicas"},
    "restart_rollout": {"namespace", "deployment_name"},
    "undo_rollout": {"namespace", "deployment_name"},
    "send_notification": {"channel", "message", "severity"},
    "request_approval": {"channel", "prompt_message"},
    "trigger_webhook": {"url", "method", "payload"},
}

# expected Python types per param from Pydantic models
PARAM_TYPES = {
    "namespace": str,
    "deployment_name": str,
    "replicas": int,
    "channel": str,
    "message": str,
    "severity": str,
    "prompt_message": str,
    "url": str,
    "method": str,
    "payload": dict,
}

class TestParamCompleteness:
    """
    For each tool, verify the response has all the required params
    with correct Python types. Catches cases where the LLM omits
    a field and validation should have rejected it.
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
    def test_has_all_required_params(self, tool_name, prompt):
        _, data = send_request(prompt)
        if not data.get("success"):
            pytest.skip("Request did not succeed")

        steps = get_steps(data)
        tool_steps = [s for s in steps if s["tool"] == tool_name]
        assert len(tool_steps) >= 1, f"No step with tool='{tool_name}'"

        actual_keys = set(tool_steps[0]["params"].keys())
        required = REQUIRED_PARAMS[tool_name]
        missing = required - actual_keys
        assert not missing, (
            f"Tool '{tool_name}' missing required params: {missing}. Got: {actual_keys}"
        )

    @pytest.mark.parametrize("tool_name,prompt", TOOL_PROMPTS)
    def test_param_types_are_correct(self, tool_name, prompt):
        _, data = send_request(prompt)
        if not data.get("success"):
            pytest.skip("Request did not succeed")

        steps = get_steps(data)
        tool_steps = [s for s in steps if s["tool"] == tool_name]
        assert len(tool_steps) >= 1

        for key, val in tool_steps[0]["params"].items():
            if key in PARAM_TYPES:
                expected_type = PARAM_TYPES[key]
                assert isinstance(val, expected_type), (
                    f"Tool '{tool_name}', param '{key}': expected {expected_type.__name__}, "
                    f"got {type(val).__name__} (value={val!r})"
                )


class TestSystemContextDoesNotBreakRegistry:
    """
    Providing system_context should not cause the LLM to hallucinate
    tools outside the 6-tool registry.
    """

    PROMPTS_WITH_CONTEXT = [
        (
            "Scale fraud-model to 5 replicas in production",
            "The fraud-model is overloaded due to Black Friday traffic spike.",
        ),
        (
            "Restart api-server in the default namespace",
            "The api-server has been leaking memory for 2 hours.",
        ),
        (
            "Rollback recommendation-model in staging",
            "The latest deployment introduced a regression in latency.",
        ),
    ]

    @pytest.mark.parametrize("prompt,context", PROMPTS_WITH_CONTEXT)
    def test_no_hallucinated_tools_with_context(self, prompt, context):
        _, data = send_request(prompt, system_context=context)
        if not data.get("success"):
            pytest.skip("Request did not succeed")

        for step in get_steps(data):
            assert step["tool"] in VALID_TOOLS, (
                f"Hallucinated tool '{step['tool']}' when system_context was provided"
            )

    @pytest.mark.parametrize("prompt,context", PROMPTS_WITH_CONTEXT)
    def test_response_is_successful_with_context(self, prompt, context):
        status, data = send_request(prompt, system_context=context)
        assert status == 200
        assert "success" in data


class TestResponseEnvelopeConsistency:
    """
    Verify the response always has the expected top-level keys
    from GenerateResponse, regardless of success or failure.
    """

    SUCCESS_PROMPTS = [
        "Scale fraud-model to 5 replicas in production",
        "Restart api-server in the default namespace",
    ]

    LIKELY_FAILURE_PROMPTS = [
        "Write me a poem about clouds",
    ]

    @pytest.mark.parametrize("prompt", SUCCESS_PROMPTS)
    def test_success_response_has_success_and_workflow(self, prompt):
        status, data = send_request(prompt)
        assert status == 200
        assert "success" in data
        if data["success"] is True:
            assert data.get("workflow") is not None, (
                "Successful response must have non-null workflow"
            )

    @pytest.mark.parametrize("prompt", LIKELY_FAILURE_PROMPTS)
    def test_failure_response_has_success_and_error(self, prompt):
        status, data = send_request(prompt)
        assert "success" in data
        if data["success"] is False:
            assert data.get("error") is not None, (
                "Failed response must have non-null error"
            )


class TestNoCrossToolParamContamination:
    """
    Ensure that params from one tool type don't leak into another.
    a restart_rollout step should not have a 'replicas' param.
    """

    def test_restart_has_no_replicas(self):
        _, data = send_request("Restart the api-server deployment in the default namespace")
        if not data.get("success"):
            pytest.skip("Request did not succeed")
        step = get_steps(data)[0]
        assert step["tool"] == "restart_rollout"
        assert "replicas" not in step["params"], (
            "restart_rollout should not have 'replicas' param"
        )

    def test_notification_has_no_namespace(self):
        _, data = send_request(
            "Send a critical alert to #incidents channel: latency exceeded 500ms"
        )
        if not data.get("success"):
            pytest.skip("Request did not succeed")
        step = get_steps(data)[0]
        assert step["tool"] == "send_notification"
        assert "namespace" not in step["params"], (
            "send_notification should not have 'namespace' param"
        )

    def test_approval_has_no_severity(self):
        _, data = send_request(
            "Ask for approval in #ops channel before proceeding with the rollback"
        )
        if not data.get("success"):
            pytest.skip("Request did not succeed")
        step = get_steps(data)[0]
        assert step["tool"] == "request_approval"
        assert "severity" not in step["params"], (
            "request_approval should not have 'severity' param"
        )

    def test_scale_has_no_message(self):
        _, data = send_request(
            "Scale my fraud-model deployment to 5 replicas in production"
        )
        if not data.get("success"):
            pytest.skip("Request did not succeed")
        step = get_steps(data)[0]
        assert step["tool"] == "scale_deployment"
        assert "message" not in step["params"], (
            "scale_deployment should not have 'message' param"
        )


class TestWorkflowStepSchema:
    """Verify that every step in a workflow adheres to the expected schema."""

    PROMPT = "Notify #ops, get approval, then restart api-server in staging"

    def test_steps_have_required_fields(self):
        _, data = send_request(self.PROMPT)
        if not data.get("success"):
            pytest.skip("Request did not succeed")

        for step in get_steps(data):
            assert "step_id" in step, f"Missing step_id: {step}"
            assert "tool" in step, f"Missing tool: {step}"
            assert "params" in step, f"Missing params: {step}"

    def test_step_id_is_integer(self):
        _, data = send_request(self.PROMPT)
        if not data.get("success"):
            pytest.skip("Request did not succeed")

        for step in get_steps(data):
            assert isinstance(step["step_id"], int), (
                f"step_id should be int, got {type(step['step_id'])}: {step}"
            )

    def test_tool_is_string(self):
        _, data = send_request(self.PROMPT)
        if not data.get("success"):
            pytest.skip("Request did not succeed")

        for step in get_steps(data):
            assert isinstance(step["tool"], str), (
                f"tool should be str, got {type(step['tool'])}: {step}"
            )

    def test_params_is_dict(self):
        _, data = send_request(self.PROMPT)
        if not data.get("success"):
            pytest.skip("Request did not succeed")

        for step in get_steps(data):
            assert isinstance(step["params"], dict), (
                f"params should be dict, got {type(step['params'])}: {step}"
            )
