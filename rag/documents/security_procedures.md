# Security Procedures

## Section 1 - Standard Response

On receiving a confirmed incident, the assigned security operator follows
this sequence: acknowledge the alert, review the camera feed and confidence
score, cross-check the zone's authorization list if available, and either
dismiss the incident as authorized or dispatch a response.

## Section 2 - Confidence Thresholds

Incidents with a detection confidence below 0.6 should be reviewed manually
before any action is taken, since low-confidence detections are more likely
to be false positives. Incidents at or above 0.6 confidence may trigger
automated notification, but still require operator sign-off before
escalation.

## Section 3 - Response Time Targets

- Tier 1 zones: acknowledge within 15 minutes.
- Tier 2 zones: acknowledge within 5 minutes.
- Tier 3 zones or after-hours incidents: acknowledge within 2 minutes.

## Section 4 - Documentation

Every incident, whether dismissed or escalated, must be logged with a
timestamp, the operator's decision, and a short justification. This log is
what the periodic safety review process audits.
