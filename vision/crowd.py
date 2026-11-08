"""
crowd.py — Crowd formation detection from YOLO detections.

Computes crowd formation events based on person count, density, and persistence
without requiring an extra neural network.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from dataclasses import dataclass

from .detect import Detection


@dataclass
class CrowdConfig:
    """Configuration for crowd formation thresholds."""

    threshold_count: int = 5
    threshold_density: float = 0.30
    dwell_threshold: float = 3.0  # seconds crowd condition must persist


@dataclass
class CrowdIncidentCandidate:
    """Emitted when crowd condition persists past the dwell threshold."""

    camera_name: str
    person_count: int
    density: float
    dwell_seconds: float
    timestamp: float

    def to_incident_dict(self) -> dict:
        """
        Convert to a schema-compliant dictionary matching `docs/schemas/incident.json`.
        """
        if self.person_count >= 10 or self.density >= 0.50:
            severity = "high"
        elif self.person_count >= 6:
            severity = "medium"
        else:
            severity = "low"

        ts_iso = datetime.fromtimestamp(self.timestamp, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        return {
            "event_type": "crowd_formation",
            "camera_name": self.camera_name,
            "zone_name": "General Area",
            "timestamp": ts_iso,
            "confidence": 0.90,
            "duration_seconds": round(self.dwell_seconds, 1),
            "severity": severity,
            "status": "open",
        }


class CrowdMonitor:
    """
    Monitors crowd persistence across frames.
    """

    def __init__(self, config: CrowdConfig | None = None) -> None:
        self.config = config or CrowdConfig()
        self._crowd_start_time: float | None = None
        self._fired: bool = False

    def update(
        self,
        person_count: int,
        density: float,
        camera_name: str,
    ) -> list[CrowdIncidentCandidate]:
        """
        Evaluate current count and density. Returns candidate when persistent.
        """
        now = time.time()
        is_crowd = (
            person_count >= self.config.threshold_count
            or density >= self.config.threshold_density
        )

        candidates: list[CrowdIncidentCandidate] = []

        if is_crowd:
            if self._crowd_start_time is None:
                self._crowd_start_time = now

            elapsed = now - self._crowd_start_time
            if elapsed >= self.config.dwell_threshold and not self._fired:
                candidates.append(
                    CrowdIncidentCandidate(
                        camera_name=camera_name,
                        person_count=person_count,
                        density=density,
                        dwell_seconds=elapsed,
                        timestamp=now,
                    )
                )
                self._fired = True
        else:
            self._crowd_start_time = None
            self._fired = False

        return candidates
