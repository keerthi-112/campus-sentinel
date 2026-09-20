# Restricted Area Rules

## Section 1 - Zone Classification

Restricted areas are zones marked by campus security as off-limits to
unauthorized personnel. Each restricted zone is assigned a risk tier:

- **Tier 1 (Low)**: Staff-only areas such as storage rooms and back offices.
- **Tier 2 (Medium)**: Lab spaces, server rooms, and equipment bays.
- **Tier 3 (High)**: Areas involving hazardous materials, high-value
  equipment, or after-hours building access points.

## Section 2 - Unauthorized Entry

When a person is detected inside a restricted zone without a corresponding
authorization record, the event is classified as unauthorized entry. The
minimum dwell time before an entry is treated as confirmed (rather than a
person briefly passing an edge of the zone) is 3 seconds.

For Tier 2 and Tier 3 zones, unauthorized entry requires the assigned
security operator to be notified immediately and to verify the individual's
authorization before any further action is taken. For Tier 1 zones, a
logged notification without immediate operator interrupt is sufficient
during business hours.

## Section 3 - Authorized Access

Personnel with standing authorization for a zone (assigned staff, approved
contractors, escorted visitors) should be excluded from triggering
unauthorized-entry incidents wherever an access record is available to the
system. Where no access-control integration exists, every entry is treated
as unauthorized and left for the operator to confirm or dismiss.

## Section 4 - After-Hours Rules

Outside of posted operating hours, any detected entry into a restricted
zone is treated as Tier 3 regardless of its normal classification, and
triggers immediate operator notification.
