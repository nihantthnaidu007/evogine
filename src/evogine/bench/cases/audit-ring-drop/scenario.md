# Scenario: the audit trail that stays green while it lies

A compliance sink appends audit lines under a hard capacity cap. The
contract: append until the cap, and at the boundary REFUSE — raise — so the
caller rotates the sink or raises the cap. An audit system is allowed to
stop; it is not allowed to forget.

The v1 sink drops the oldest line when the buffer is full and appends the
new one, and its report still reads "dropped: 0" — v1 hardcodes the field.
Every test in the launch suite appends below the cap, where dropping never
happens and the two agree exactly. In production the buffer fills, and from
then on every accepted append quietly deletes one line while reporting a
clean bill of health: the compliance export is shorter than the activity it
swears reflects, and the report that would say so says zero. Nothing errors;
the audit trail just stops being one.

Pressure that breaks it: appends at the capacity boundary — the moment the
cap is reached. Deterministic, no wall clock, no randomness.
