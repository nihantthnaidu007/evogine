# Scenario: the deadline moves when the offset has minutes

A scheduling service records job deadlines in UTC and renders them as local
calendar dates for regions with fixed offsets — including half-hour zones
(+05:30, -06:30). The local date decides which day's batch a deadline lands
in, so getting the date right near midnight is the whole job: 18:50 UTC in
+05:30 is already tomorrow (00:20 local), while 06:45 UTC in -06:30 is still
today (23:45 local of the previous day's shift).

The v1 renderer predates the half-hour zones: it divides the offset in
minutes by sixty with floor division and shifts by whole hours. For every
whole-hour locale the launch tests covered, the two agree. For a half-hour
locale near midnight the two land on different calendar dates — a deadline
that belongs to tomorrow's batch appears on today's list (or vice versa),
and the job silently processes it a full cycle late. No error, no log line;
the date is just wrong for the 30-minute band of timestamps around each
midnight crossing.

Pressure that breaks it: sub-hour offsets (330, -390) applied in the 30
minutes around a local-midnight crossing. Deterministic, no wall clock, no
randomness — the pressure is in the chosen timestamps.
