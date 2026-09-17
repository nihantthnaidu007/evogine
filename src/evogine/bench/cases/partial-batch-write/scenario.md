# Scenario: the batch that half-happened

A ledger accepts batches of entries and posts them atomically: every entry
is validated first, and one malformed entry aborts the whole batch — status
"aborted", the offending index reported, nothing applied. Downstream
consumers rely on that contract: on "aborted" they re-submit the entire
batch.

The v1 ledger validates and applies in the same loop. On a malformed entry
it returns the same "aborted" status and the same failed index — but every
valid entry before the bad one is already applied. When the caller sees
"aborted" and resubmits the full batch, the surviving prefix is applied
twice: double-posted amounts, silently. The visible tests only abort on the
FIRST entry (nothing applied yet) or submit all-valid batches, so both
implementations look identical.

Pressure that breaks it: a malformed entry after valid ones — partial
application under a contract that promises atomicity. Deterministic, no
wall clock, no randomness.
