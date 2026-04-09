"""
Pydantic schemas for the Model 1 anomaly classification endpoint.

Request:  A list of integer token IDs representing a 5-minute
          infrastructure telemetry window (CPU buckets, memory buckets,
          status codes, event codes, or log template IDs).

Response: The predicted anomaly class (0-6) with its confidence score
          and a human-readable label.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

# Maps class index to the label used during training (track_a.yaml)
LABEL_NAMES: dict[int, str] = {
    0: "Normal",
    1: "Resource_Exhaustion",
    2: "System_Crash",
    3: "Network_Failure",
    4: "Data_Drift",
    5: "Auth_Failure",
    6: "Permission_Denied",
}


class AnomalyRequest(BaseModel):
    """Incoming prediction request — a single 5-minute token window."""

    sequence_ids: list[int] = Field(
        ...,
        description="List of integer token IDs from the infrastructure telemetry window.",
        examples=[[100, 102, 200, 205, 300, 55, 72, 0, 0, 0]],
    )

    @field_validator("sequence_ids")
    @classmethod
    def must_not_be_empty(cls, v: list[int]) -> list[int]:
        if len(v) == 0:
            raise ValueError("sequence_ids must contain at least one token")
        return v


class AnomalyResponse(BaseModel):
    """Prediction result returned by the /predict_anomaly endpoint."""

    class_id: int = Field(
        ...,
        ge=0,
        le=6,
        description="Predicted anomaly class (0 = Normal, 1-6 = anomaly types).",
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Softmax probability of the predicted class.",
    )
    label: str = Field(
        ...,
        description="Human-readable name of the predicted class.",
        examples=["Normal", "Resource_Exhaustion"],
    )
