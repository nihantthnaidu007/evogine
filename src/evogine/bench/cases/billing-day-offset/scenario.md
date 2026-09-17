# Scenario: month-end billing drifts or dies

Subscription renewals are scheduled on the customer's signup day of month.
A customer who signed up on the 31st renews on the last day of every month:
Jan 31 + 1 month is Feb 28 (Feb 29 in a leap year), never Mar 3. The same
clamping applies when a charge is pushed forward several months across a
year boundary.

The v1 billing job stores due dates as ISO strings and advances them with
day-of-month arithmetic that does not know month lengths. On mid-month
customers — the ones the launch tests covered — it is exact. On month-end
customers the date constructor hits an out-of-range day when the renewal
batch runs at month end: the job raises, that customer's renewal is skipped,
and every later month-end customer in the batch is dropped with it. The
visible tests never signed anybody up after the 28th, so the batch looks
green in staging all quarter.

Pressure that breaks it: month lengths (28/29/30/31), leap-year Februaries,
and year-crossing month arithmetic. Deterministic, no wall clock, no
randomness — the pressure is in the chosen dates.
