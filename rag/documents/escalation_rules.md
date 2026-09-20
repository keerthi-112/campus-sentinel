# Escalation Rules

## Section 1 - Severity Levels

Incidents carry one of three severities: **low** (logged only), **medium**
(operator notified, response-time target applies), **high** (emergency
procedures apply).

## Section 2 - Severity Assignment

Severity is derived from zone tier, time of day, and detection confidence:
a Tier 3 zone or after-hours entry starts at medium severity by default and
is raised to high if confidence is at or above 0.85 or if the entry
persists past 10 seconds. Tier 1 entries default to low severity unless
the operator manually raises them.

## Section 3 - Repeat Incidents

Three or more unauthorized-entry incidents at the same zone within a
24-hour window automatically raises the severity of the next incident at
that zone by one level, on the assumption that a persistent access-control
gap exists.
