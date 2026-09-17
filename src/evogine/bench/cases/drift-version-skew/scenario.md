# Scenario: two teams, two schemas, one wire

A client and a server communicate through a tagged payload: a schema
version, the item list, and a checksum. The v2 wire format — version 2,
checksum = sum of items mod 2^32 — is the contract both sides must speak.
The pipeline facade encodes on the client and decodes on the server; the
decoder must refuse payloads it cannot verify, including wrong versions and
checksum mismatches, so skew fails loudly at the boundary instead of
corrupting data downstream.

The port happened the way ports usually happen: client and server were
updated in different sittings. The shipped pair speaks v1 on both sides —
version tag 1, no checksum — and,each side is internally
consistent: client encodes, server decodes, the facade round-trips, and
every single-payload test in the launch suite passes. The decay is in the
SEAM: v2-tagged payloads from anything that already moved (a second
producer, a replayed archive) are rejected by the v1 server, and v1 payloads
the v2 contract forbids are accepted without verification. Single-side tests
cannot see a seam; only the shared schema can.

Pressure that breaks it: payloads that cross the version boundary in either
direction — v2 in, v1 accepted-when-forbidden. Deterministic, no wall clock,
no randomness.
