# Scenario: the replica that replays arrivals, not history

Two services write to the same account. Every write arrives at the ledger
replica tagged with its sequence number — the order the WRITES happened, not
the order the NETWORK delivered them. Reconciliation folds the pending
writes by that contract: with concurrent writes to one field, the highest
sequence number wins. A second reconciliation must return the same value
(replay is idempotent) and the pending set must stay intact for audit.

The v1 replica folds writes in ARRIVAL order — the order record() happened
to be called — because in all launch tests, writes were recorded in
sequence order and the two are indistinguishable. Under pressure, packets
reorder: record(1), record(2) arrives as record(2), record(1), and the v1
reconciliation answers with the value that lost — the balance disagrees
with the primary by exactly the amount of the reorder. Two reconciliations
in a row agree with each other, confidently, on the wrong number. Nothing
errors; the replica just disagrees with history.

Pressure that breaks it: arrival order differing from sequence order —
concurrent writes under network reordering. Deterministic, no wall clock,
no randomness.
