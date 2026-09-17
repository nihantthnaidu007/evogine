# Scenario: the queue that admits everything and reports capacity

A scheduler fronts a bounded worker pool. Offers carry a weight; the queue
admits an offer only when the running load stays within capacity, sheds the
rest (offer returns False — backpressure), and load() reports the committed
load. Under a spike, shedding is the feature: callers see the rejection and
route elsewhere or retry later.

The v1 queue admits every offer — "capacity is advisory" — and its load()
gauge is min(actual, capacity), so the dashboard pins at exactly 100% and
never overflows. In staging the load never crossed the cap, and the two
agree: same admissions, same gauge, same drain. Under a spike the v1 queue
takes everything: the gauge reads a full-but-orderly 100% while the real
committed load grows without bound, drain returns work the pool never had
capacity for, and upstream callers that saw "True" never retry. The spike
the shedder existed for is the spike it cannot report.

Pressure that breaks it: offered weight crossing the capacity boundary —
the shedding decision and the honesty of the gauge. Deterministic: integer
weights, no wall clock, no randomness.
