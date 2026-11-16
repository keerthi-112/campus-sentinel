"""Pydantic request / response shapes."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LoginIn(BaseModel):
    username: str
    password: str


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    username: str
    role: str


class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=6, max_length=128)
    role: Literal["admin", "operator"] = "operator"


class CameraIn(BaseModel):
    camera_name: str = Field(min_length=1, max_length=80)
    zone_name: str = Field(min_length=1, max_length=120)
    source_type: Literal["file", "webcam", "rtsp"] = "file"
    source: str = "0"
    restricted_region: list[list[int]] = Field(default_factory=list)
    tier: int = Field(default=2, ge=1, le=3)
    dwell_threshold: float = Field(default=4.0, gt=0, le=600)
    confidence_threshold: float = Field(default=0.40, gt=0, lt=1)
    crowd_count: int = Field(default=5, ge=2, le=500)
    loop_video: bool = True

    @field_validator("restricted_region")
    @classmethod
    def _poly(cls, v):
        if v and len(v) < 3:
            raise ValueError("a restricted region needs at least 3 points")
        for pt in v:
            if len(pt) != 2:
                raise ValueError("each point must be [x, y]")
        return v


class CameraOut(CameraIn):
    model_config = ConfigDict(from_attributes=True)
    id: int
    active_state: bool
    runtime: dict | None = None


class IncidentIn(BaseModel):
    """docs/schemas/incident.json (+ optional features) — what vision/ posts."""

    event_type: str
    camera_name: str
    zone_name: str
    timestamp: str
    confidence: float = Field(ge=0.0, le=1.0)
    duration_seconds: float = Field(ge=0.0)
    severity: Literal["low", "medium", "high"] = "medium"
    status: str = "open"
    incident_ref: str | None = None
    features: dict | None = None  # optional event_feature_vector.json record


class IncidentPatch(BaseModel):
    status: Literal["open", "acknowledged", "resolved", "false_alarm"]


class ReportOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    incident_ref: str
    report_text: str
    suggested_action: str
    source_section: str
    created_time: str


class IncidentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    incident_ref: str
    event_type: str
    camera_id: int | None
    camera_name: str
    zone_name: str
    timestamp: str
    confidence: float
    duration_seconds: float
    severity: str
    status: str
    classifier_decision: str
    classifier_score: float | None
    model_version: str | None
    decision_reason: str | None
    features: dict | None
    source: str
    report_job_id: str | None
    report_status: str | None
    report_error: str | None
    created_at: datetime
    report: ReportOut | None = None


class ChatIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class TrainIn(BaseModel):
    engine: Literal["simulation", "flower"] = "simulation"
    strategy: Literal["fedavg", "fedprox", "both"] = "both"
    rounds: int = Field(default=20, ge=1, le=200)
    clients: int = Field(default=5, ge=3, le=5)
    local_epochs: int = Field(default=3, ge=1, le=50)
    mu: float = Field(default=0.05, ge=0.0, le=10.0)
