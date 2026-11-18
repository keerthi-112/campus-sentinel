"""
models.py — The application's data model.

Raw video is never stored here: only camera/zone configuration, event
metadata, generated reports, federated round logs and users.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(256))
    role: Mapped[str] = mapped_column(String(16), default="operator")  # admin | operator
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Camera(Base):
    """Camera & zone record: camera_name, zone_name, source_type, restricted_region, active_state."""

    __tablename__ = "cameras"
    id: Mapped[int] = mapped_column(primary_key=True)
    camera_name: Mapped[str] = mapped_column(String(80), unique=True)
    zone_name: Mapped[str] = mapped_column(String(120))
    source_type: Mapped[str] = mapped_column(String(16), default="file")  # file | webcam | rtsp
    source: Mapped[str] = mapped_column(String(500), default="0")
    restricted_region: Mapped[list] = mapped_column(JSON, default=list)  # [[x, y], ...] in frame pixels
    tier: Mapped[int] = mapped_column(Integer, default=2)
    dwell_threshold: Mapped[float] = mapped_column(Float, default=4.0)
    confidence_threshold: Mapped[float] = mapped_column(Float, default=0.40)
    crowd_count: Mapped[int] = mapped_column(Integer, default=5)
    loop_video: Mapped[bool] = mapped_column(Boolean, default=True)
    active_state: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Incident(Base):
    """Incident record: event_type, camera, zone, timestamp, confidence, duration, severity, status."""

    __tablename__ = "incidents"
    id: Mapped[int] = mapped_column(primary_key=True)
    incident_ref: Mapped[str] = mapped_column(String(200), unique=True, index=True)
    event_type: Mapped[str] = mapped_column(String(40), index=True)
    camera_id: Mapped[int | None] = mapped_column(ForeignKey("cameras.id", ondelete="SET NULL"), nullable=True)
    camera_name: Mapped[str] = mapped_column(String(80))
    zone_name: Mapped[str] = mapped_column(String(120))
    timestamp: Mapped[str] = mapped_column(String(40), index=True)
    confidence: Mapped[float] = mapped_column(Float)
    duration_seconds: Mapped[float] = mapped_column(Float)
    severity: Mapped[str] = mapped_column(String(16))
    # open | acknowledged | resolved | false_alarm | suppressed
    status: Mapped[str] = mapped_column(String(20), default="open", index=True)

    # Event-classifier decision (federated model)
    classifier_decision: Mapped[str] = mapped_column(String(16), default="escalated")  # escalated | suppressed
    classifier_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(60), nullable=True)
    decision_reason: Mapped[str | None] = mapped_column(String(300), nullable=True)
    features: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    source: Mapped[str] = mapped_column(String(20), default="vision")  # vision | external | manual

    # RAG report job
    report_job_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    report_status: Mapped[str | None] = mapped_column(String(16), nullable=True)  # pending|running|ready|failed
    report_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class Report(Base):
    """Report record: incident_ref, report_text, suggested_action, source_section, created_time."""

    __tablename__ = "reports"
    id: Mapped[int] = mapped_column(primary_key=True)
    incident_ref: Mapped[str] = mapped_column(String(200), unique=True, index=True)
    report_text: Mapped[str] = mapped_column(Text)
    suggested_action: Mapped[str] = mapped_column(Text)
    source_section: Mapped[str] = mapped_column(String(300))
    created_time: Mapped[str] = mapped_column(String(40))
    job_id: Mapped[str | None] = mapped_column(String(64), nullable=True)


class FederatedRound(Base):
    """Federated round record: round_number, strategy, participating_clients, metrics."""

    __tablename__ = "federated_rounds"
    __table_args__ = (UniqueConstraint("run_id", "strategy", "round_number"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[str] = mapped_column(String(32), index=True)
    engine: Mapped[str] = mapped_column(String(16), default="simulation")
    round_number: Mapped[int] = mapped_column(Integer)
    strategy: Mapped[str] = mapped_column(String(16))
    participating_clients: Mapped[list] = mapped_column(JSON, default=list)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    comm_bytes: Mapped[int] = mapped_column(Integer, default=0)
    cumulative_comm_bytes: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
