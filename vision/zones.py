"""
zones.py — Restricted-zone configuration, point-in-polygon test, and dwell
timer logic.

A zone is a polygon drawn over a camera frame.  When a tracked person's
centroid enters the polygon and *stays* there longer than ``dwell_threshold``
seconds, the zone emits an incident candidate.  Single-frame noise never
reaches downstream consumers (classifier, RAG, backend).
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
import cv2
import numpy as np
from dataclasses import dataclass, field

from .tracker import TrackedPerson


@dataclass
class ZoneConfig:
    """
    One restricted zone on one camera.

    Parameters
    ----------
    zone_name : str
        Human-readable label (e.g. "Lab 2 restricted area").
    polygon : list[tuple[int, int]]
        Ordered vertices in pixel coordinates.
    dwell_threshold : float
        Seconds a person must stay inside before it counts (default 4.0).
    """

    zone_name: str
    polygon: list[tuple[int, int]]
    dwell_threshold: float = 4.0

    def __post_init__(self) -> None:
        # Pre-compute the numpy contour for cv2.pointPolygonTest
        self._contour = np.array(self.polygon, dtype=np.int32)


@dataclass
class _DwellEntry:
    """Internal: tracks when a person first entered and their peak confidence."""
    enter_time: float
    max_confidence: float = 0.0


@dataclass
class ZoneIncidentCandidate:
    """
    Emitted when a person has stayed inside a zone longer than the dwell
    threshold.

    Includes a helper method ``to_incident_dict()`` that produces records
    strictly compliant with ``docs/schemas/incident.json`` for RAG and backend.
    """

    person_id: int
    zone_name: str
    camera_name: str
    enter_time: float
    dwell_seconds: float
    max_confidence: float
    timestamp: float  # epoch seconds when threshold was crossed

    def to_incident_dict(self) -> dict:
        """
        Convert this candidate to a schema-compliant dictionary matching
        `docs/schemas/incident.json`.
        """
        # Calculate severity based on dwell duration and detection confidence
        if self.dwell_seconds >= 8.0 or self.max_confidence >= 0.92:
            severity = "high"
        elif self.dwell_seconds >= 4.0:
            severity = "medium"
        else:
            severity = "low"

        # UTC ISO 8601 string: YYYY-MM-DDTHH:MM:SSZ
        ts_iso = datetime.fromtimestamp(self.timestamp, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        return {
            "event_type": "restricted_area_entry",
            "camera_name": self.camera_name,
            "zone_name": self.zone_name,
            "timestamp": ts_iso,
            "confidence": round(self.max_confidence, 2),
            "duration_seconds": round(self.dwell_seconds, 1),
            "severity": severity,
            "status": "open",
        }


class ZoneMonitor:
    """
    Watches one or more zones on a single camera.

    Call ``update()`` every frame with the current tracked persons and the
    camera name. It returns a list of new incident candidates that crossed
    the dwell threshold.
    """

    def __init__(self, zones: list[ZoneConfig]) -> None:
        self.zones = zones
        # {zone_name -> {person_id -> _DwellEntry}}
        self._dwells: dict[str, dict[int, _DwellEntry]] = {
            z.zone_name: {} for z in zones
        }
        # Remember which (person, zone) pairs already fired so we don't
        # re-emit duplicate candidates each frame
        self._fired: set[tuple[int, str]] = set()

    def update(
        self,
        tracks: dict[int, TrackedPerson],
        camera_name: str,
    ) -> list[ZoneIncidentCandidate]:
        """
        Check every tracked person against every zone.

        Returns new incident candidates that crossed the dwell threshold
        *this frame* (never duplicates).
        """
        now = time.time()
        candidates: list[ZoneIncidentCandidate] = []

        for zone in self.zones:
            active_ids: set[int] = set()

            for pid, person in tracks.items():
                inside = self.is_point_in_zone(person.center, zone)

                if inside:
                    active_ids.add(pid)
                    dwells = self._dwells[zone.zone_name]

                    if pid not in dwells:
                        # Person just entered
                        dwells[pid] = _DwellEntry(
                            enter_time=now,
                            max_confidence=person.confidence,
                        )
                    else:
                        dwells[pid].max_confidence = max(
                            dwells[pid].max_confidence,
                            person.confidence,
                        )

                    entry = dwells[pid]
                    elapsed = now - entry.enter_time

                    if (elapsed >= zone.dwell_threshold
                            and (pid, zone.zone_name) not in self._fired):
                        candidates.append(ZoneIncidentCandidate(
                            person_id=pid,
                            zone_name=zone.zone_name,
                            camera_name=camera_name,
                            enter_time=entry.enter_time,
                            dwell_seconds=elapsed,
                            max_confidence=entry.max_confidence,
                            timestamp=now,
                        ))
                        self._fired.add((pid, zone.zone_name))

            # Purge people who left the zone
            gone = set(self._dwells[zone.zone_name]) - active_ids
            for pid in gone:
                del self._dwells[zone.zone_name][pid]
                self._fired.discard((pid, zone.zone_name))

        return candidates

    def get_zone_dwell_info(
        self,
        zone: ZoneConfig,
        tracks: dict[int, TrackedPerson],
    ) -> tuple[float, float]:
        """
        Public query for feature extraction: returns (max_dwell_seconds, max_confidence)
        for any tracked individual currently inside the given zone.
        """
        now = time.time()
        dwells = self._dwells.get(zone.zone_name, {})
        max_dwell = 0.0
        max_conf = 0.0

        for pid, person in tracks.items():
            if self.is_point_in_zone(person.center, zone):
                entry = dwells.get(pid)
                if entry:
                    elapsed = now - entry.enter_time
                    if elapsed > max_dwell:
                        max_dwell = elapsed
                    if entry.max_confidence > max_conf:
                        max_conf = entry.max_confidence

        return max_dwell, max_conf

    def is_point_in_zone(
        self,
        point: tuple[int, int],
        zone: ZoneConfig,
    ) -> bool:
        """True if the point lies inside or on the boundary of the polygon."""
        pt = (float(point[0]), float(point[1]))
        result = cv2.pointPolygonTest(zone._contour, pt, measureDist=False)
        return result >= 0

    # ------------------------------------------------------------------
    # Drawing helpers (for dashboard / operator view)
    # ------------------------------------------------------------------

    def draw_zones(self, frame: np.ndarray) -> np.ndarray:
        """Draw all zone polygons onto the frame (in-place + returned)."""
        for zone in self.zones:
            pts = zone._contour.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True,
                          color=(0, 0, 255), thickness=2)
            # Label
            tx, ty = zone.polygon[0]
            cv2.putText(frame, zone.zone_name, (tx, max(ty - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return frame
