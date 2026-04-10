"""Unit tests for app.schemas.workflow — Pydantic validation of the 6-tool registry."""

import pytest
from pydantic import ValidationError

from app.schemas.workflow import (
    VALID_TOOLS,
    RequestApprovalParams,
    RestartRolloutParams,
    ScaleDeploymentParams,
    SendNotificationParams,
    TriggerWebhookParams,
    UndoRolloutParams,
    Workflow,
    WorkflowResponse,
    WorkflowStep,
)


# ===================================================================
# Per-tool param models
# ===================================================================

class TestScaleDeploymentParams:
    def test_valid(self):
        p = ScaleDeploymentParams(namespace="prod", deployment_name="api", replicas=3)
        assert p.replicas == 3

    def test_replicas_zero_rejected(self):
        with pytest.raises(ValidationError):
            ScaleDeploymentParams(namespace="prod", deployment_name="api", replicas=0)

    def test_replicas_negative_rejected(self):
        with pytest.raises(ValidationError):
            ScaleDeploymentParams(namespace="prod", deployment_name="api", replicas=-1)

    def test_replicas_string_rejected(self):
        with pytest.raises(ValidationError):
            ScaleDeploymentParams(namespace="prod", deployment_name="api", replicas="five")

    def test_missing_namespace(self):
        with pytest.raises(ValidationError):
            ScaleDeploymentParams(deployment_name="api", replicas=3)

    def test_empty_namespace_rejected(self):
        with pytest.raises(ValidationError):
            ScaleDeploymentParams(namespace="", deployment_name="api", replicas=3)


class TestRestartRolloutParams:
    def test_valid(self):
        p = RestartRolloutParams(namespace="prod", deployment_name="api")
        assert p.namespace == "prod"

    def test_missing_deployment_name(self):
        with pytest.raises(ValidationError):
            RestartRolloutParams(namespace="prod")


class TestUndoRolloutParams:
    def test_valid(self):
        p = UndoRolloutParams(namespace="staging", deployment_name="ml-model-v2")
        assert p.deployment_name == "ml-model-v2"


class TestSendNotificationParams:
    def test_valid(self):
        p = SendNotificationParams(channel="#alerts", message="All clear", severity="info")
        assert p.severity == "info"

    def test_invalid_severity(self):
        with pytest.raises(ValidationError):
            SendNotificationParams(channel="#a", message="x", severity="urgent")

    def test_empty_message_rejected(self):
        with pytest.raises(ValidationError):
            SendNotificationParams(channel="#a", message="", severity="info")

    def test_all_severities(self):
        for sev in ("info", "warning", "critical"):
            p = SendNotificationParams(channel="#a", message="m", severity=sev)
            assert p.severity == sev


class TestRequestApprovalParams:
    def test_valid(self):
        p = RequestApprovalParams(channel="#approvals", prompt_message="Approve restart?")
        assert p.prompt_message == "Approve restart?"

    def test_missing_prompt_message(self):
        with pytest.raises(ValidationError):
            RequestApprovalParams(channel="#approvals")


class TestTriggerWebhookParams:
    def test_valid_post(self):
        p = TriggerWebhookParams(
            url="https://airflow.co/api/run",
            method="POST",
            payload={"dag": "retrain"},
        )
        assert p.method == "POST"

    def test_valid_get_empty_payload(self):
        p = TriggerWebhookParams(
            url="http://localhost:9090/health",
            method="GET",
            payload={},
        )
        assert p.payload == {}

    def test_invalid_method(self):
        with pytest.raises(ValidationError):
            TriggerWebhookParams(url="https://x.com", method="PATCH", payload={})

    def test_invalid_url_no_scheme(self):
        with pytest.raises(ValidationError):
            TriggerWebhookParams(url="airflow.co/api", method="POST", payload={})

    def test_missing_payload(self):
        with pytest.raises(ValidationError):
            TriggerWebhookParams(url="https://x.com", method="GET")

    def test_all_methods(self):
        for method in ("GET", "POST", "PUT", "DELETE"):
            p = TriggerWebhookParams(url="https://x.com", method=method, payload={})
            assert p.method == method


# ===================================================================
# WorkflowStep
# ===================================================================

class TestWorkflowStep:
    def test_valid_step(self):
        step = WorkflowStep(
            step_id=1,
            tool="scale_deployment",
            params={"namespace": "prod", "deployment_name": "api", "replicas": 5},
        )
        assert step.tool == "scale_deployment"

    def test_hallucinated_tool_name(self):
        with pytest.raises(ValidationError, match="literal"):
            WorkflowStep(
                step_id=1,
                tool="replicas",
                params={"deployment": "my-dep", "replicas": 5},
            )

    def test_hallucinated_tool_kubectl(self):
        with pytest.raises(ValidationError):
            WorkflowStep(step_id=1, tool="kubectl_apply", params={})

    def test_wrong_params_for_tool(self):
        """scale_deployment requires 'replicas' as int, not str."""
        with pytest.raises(ValidationError, match="Invalid params"):
            WorkflowStep(
                step_id=1,
                tool="scale_deployment",
                params={"namespace": "prod", "deployment_name": "api", "replicas": "five"},
            )

    def test_missing_required_param(self):
        """restart_rollout requires namespace AND deployment_name."""
        with pytest.raises(ValidationError, match="Invalid params"):
            WorkflowStep(
                step_id=1,
                tool="restart_rollout",
                params={"namespace": "prod"},
            )

    def test_step_id_zero_rejected(self):
        with pytest.raises(ValidationError):
            WorkflowStep(
                step_id=0,
                tool="restart_rollout",
                params={"namespace": "prod", "deployment_name": "api"},
            )

    def test_step_id_negative_rejected(self):
        with pytest.raises(ValidationError):
            WorkflowStep(
                step_id=-1,
                tool="restart_rollout",
                params={"namespace": "prod", "deployment_name": "api"},
            )

    def test_extra_params_are_tolerated(self):
        """LLM may add extra fields — they should not cause a crash."""
        step = WorkflowStep(
            step_id=1,
            tool="restart_rollout",
            params={
                "namespace": "prod",
                "deployment_name": "api",
                "extra_field": "should be ignored by param model",
            },
        )
        assert step.tool == "restart_rollout"


# ===================================================================
# Workflow (step ordering)
# ===================================================================

class TestWorkflow:
    def test_valid_single_step(self):
        w = Workflow(
            steps=[
                WorkflowStep(
                    step_id=1,
                    tool="undo_rollout",
                    params={"namespace": "prod", "deployment_name": "api"},
                )
            ]
        )
        assert len(w.steps) == 1

    def test_valid_multi_step(self):
        w = Workflow(
            steps=[
                WorkflowStep(
                    step_id=1,
                    tool="send_notification",
                    params={"channel": "#ops", "message": "Starting", "severity": "info"},
                ),
                WorkflowStep(
                    step_id=2,
                    tool="restart_rollout",
                    params={"namespace": "prod", "deployment_name": "api"},
                ),
            ]
        )
        assert len(w.steps) == 2

    def test_empty_steps_rejected(self):
        with pytest.raises(ValidationError):
            Workflow(steps=[])

    def test_non_sequential_ids_rejected(self):
        with pytest.raises(ValidationError, match="sequential"):
            Workflow(
                steps=[
                    WorkflowStep(
                        step_id=1,
                        tool="restart_rollout",
                        params={"namespace": "p", "deployment_name": "d"},
                    ),
                    WorkflowStep(
                        step_id=3,  # gap — should be 2
                        tool="restart_rollout",
                        params={"namespace": "p", "deployment_name": "d"},
                    ),
                ]
            )

    def test_duplicate_step_ids_rejected(self):
        with pytest.raises(ValidationError, match="sequential"):
            Workflow(
                steps=[
                    WorkflowStep(
                        step_id=1,
                        tool="restart_rollout",
                        params={"namespace": "p", "deployment_name": "d"},
                    ),
                    WorkflowStep(
                        step_id=1,  # duplicate
                        tool="restart_rollout",
                        params={"namespace": "p", "deployment_name": "d"},
                    ),
                ]
            )

    def test_ids_starting_at_zero_rejected(self):
        with pytest.raises(ValidationError):
            Workflow(
                steps=[
                    WorkflowStep(
                        step_id=0,
                        tool="restart_rollout",
                        params={"namespace": "p", "deployment_name": "d"},
                    ),
                ]
            )


# ===================================================================
# WorkflowResponse (top-level)
# ===================================================================

class TestWorkflowResponse:
    def test_valid_full_response(self, valid_scale_workflow):
        resp = WorkflowResponse.model_validate(valid_scale_workflow)
        assert resp.workflow.steps[0].tool == "scale_deployment"

    def test_valid_multi_step_response(self, valid_multi_step_workflow):
        resp = WorkflowResponse.model_validate(valid_multi_step_workflow)
        assert len(resp.workflow.steps) == 4
        assert resp.workflow.steps[0].tool == "send_notification"
        assert resp.workflow.steps[1].tool == "request_approval"
        assert resp.workflow.steps[2].tool == "restart_rollout"
        assert resp.workflow.steps[3].tool == "send_notification"

    def test_valid_webhook_response(self, valid_webhook_workflow):
        resp = WorkflowResponse.model_validate(valid_webhook_workflow)
        assert resp.workflow.steps[0].params["method"] == "POST"

    def test_missing_workflow_key(self):
        with pytest.raises(ValidationError):
            WorkflowResponse.model_validate({"steps": [{"step_id": 1}]})

    def test_completely_wrong_shape(self):
        with pytest.raises(ValidationError):
            WorkflowResponse.model_validate({"answer": "scale it up"})

    def test_null_workflow_rejected(self):
        with pytest.raises(ValidationError):
            WorkflowResponse.model_validate({"workflow": None})