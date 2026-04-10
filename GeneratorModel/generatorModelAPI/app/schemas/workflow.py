"""
Pydantic v2 schemas for validating Model 2 workflow output.

The 6-tool registry enforced by these models:
  scale_deployment   – Scale a Kubernetes deployment to N replicas
  restart_rollout    – Trigger a rolling restart
  undo_rollout       – Roll back to the previous revision
  send_notification  – Send a Slack / PagerDuty alert
  request_approval   – Pause for human approval
  trigger_webhook    – Fire an arbitrary HTTP webhook
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Canonical list of tools -- used for Literal type AND runtime checks
# ---------------------------------------------------------------------------
VALID_TOOLS: tuple[str, ...] = (
    "scale_deployment",
    "restart_rollout",
    "undo_rollout",
    "send_notification",
    "request_approval",
    "trigger_webhook",
)

# ---------------------------------------------------------------------------
# Per-tool parameter models
# ---------------------------------------------------------------------------


class ScaleDeploymentParams(BaseModel):
    namespace: str = Field(..., min_length=1)
    deployment_name: str = Field(..., min_length=1)
    replicas: int = Field(..., ge=1)


class RestartRolloutParams(BaseModel):
    namespace: str = Field(..., min_length=1)
    deployment_name: str = Field(..., min_length=1)


class UndoRolloutParams(BaseModel):
    namespace: str = Field(..., min_length=1)
    deployment_name: str = Field(..., min_length=1)


class SendNotificationParams(BaseModel):
    channel: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    severity: Literal["info", "warning", "critical"]


class RequestApprovalParams(BaseModel):
    channel: str = Field(..., min_length=1)
    prompt_message: str = Field(..., min_length=1)


class TriggerWebhookParams(BaseModel):
    url: str = Field(..., min_length=1)
    method: Literal["GET", "POST", "PUT", "DELETE"]
    payload: dict[str, Any]

    @field_validator("url")
    @classmethod
    def url_must_be_http(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("url must start with http:// or https://")
        return v


# Maps tool name -> param model for runtime dispatch
TOOL_PARAM_MODELS: dict[str, type[BaseModel]] = {
    "scale_deployment": ScaleDeploymentParams,
    "restart_rollout": RestartRolloutParams,
    "undo_rollout": UndoRolloutParams,
    "send_notification": SendNotificationParams,
    "request_approval": RequestApprovalParams,
    "trigger_webhook": TriggerWebhookParams,
}

# ---------------------------------------------------------------------------
# Workflow models
# ---------------------------------------------------------------------------


class WorkflowStep(BaseModel):
    """A single step in a remediation workflow."""

    step_id: int = Field(..., ge=1)
    tool: Literal[
        "scale_deployment",
        "restart_rollout",
        "undo_rollout",
        "send_notification",
        "request_approval",
        "trigger_webhook",
    ]
    params: dict[str, Any]

    @model_validator(mode="after")
    def validate_params_for_tool(self) -> WorkflowStep:
        """Dispatch params validation to the tool-specific Pydantic model."""
        param_model = TOOL_PARAM_MODELS[self.tool]
        try:
            param_model.model_validate(self.params)
        except Exception as exc:
            raise ValueError(
                f"Invalid params for tool '{self.tool}': {exc}"
            ) from exc
        return self


class Workflow(BaseModel):
    """Ordered list of workflow steps."""

    steps: list[WorkflowStep] = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_step_ids_sequential(self) -> Workflow:
        """step_id values must be 1, 2, 3, … with no gaps or duplicates."""
        expected = list(range(1, len(self.steps) + 1))
        actual = [s.step_id for s in self.steps]
        if actual != expected:
            raise ValueError(
                f"step_id values must be sequential starting from 1. "
                f"Expected {expected}, got {actual}."
            )
        return self


class WorkflowResponse(BaseModel):
    """Top-level wrapper matching the LLM output format: {"workflow": {...}}"""

    workflow: Workflow