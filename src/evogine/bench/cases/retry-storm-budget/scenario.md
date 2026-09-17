# Scenario: the retry budget that never forgives

A client guards a flaky dependency with a retry budget: at most N attempts
inside a sliding window (millisecond logical clock), recorded as they are
attempted. After a storm — N rapid retries — the budget is spent. The point
of the window is recovery: once the storm passes and the window slides past
those attempts, the budget must refill and the client must retry again.

The v1 client counts every attempt it has ever seen and compares that count
to the cap. During the storm it behaves identically to the reference. After
the storm it never recovers: the count only grows, so every later attempt is
denied forever — the service that had one bad minute is black-holed for the
lifetime of the process. Staging tests storm once and assert the denial;
they never assert what happens a window later.

Pressure that breaks it: sustained or repeated retry storms with a sliding
window — the state the window is supposed to forget. Deterministic: a
logical millisecond clock, no wall clock, no randomness.
