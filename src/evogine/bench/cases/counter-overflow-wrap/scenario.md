# Scenario: the ids start coming back

Every event a pipeline ingests gets an id: an epoch tag shifted above a
20-bit per-epoch counter — 1,048,576 distinct ids per epoch, and the epoch
is bumped by the operator when an epoch fills. When the counter space is
exhausted, the source must saturate: next_id returns None and the caller
rolls the epoch. Every id ever emitted stays unique forever.

The v1 source wraps the counter modulo the space instead of saturating.
For the first 1,048,576 calls — every load test, every staging run — the
two are indistinguishable. Sometime after, the counter silently wraps to
zero and the source starts reissuing ids it already emitted: two different
events carry the same id, a downstream deduplicator drops the newer event
as a "duplicate", and a low-rate feed loses data at exactly the cadence of
the wrap. Nothing crashes; ids just stop meaning one event.

Pressure that breaks it: consuming the last of the counter space and
stepping past it — the overflow boundary itself. Deterministic: integer
arithmetic only, no wall clock, no randomness.
