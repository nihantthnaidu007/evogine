# Scenario: two users, one identity, different bytes

A CRM derives a stable display key from name parts: normalize each part to
NFKC (so fullwidth Latin, ligatures like the fi ligature, and other
compatibility forms collapse to their canonical spelling), casefold for
case-insensitive matching, then join with a visible separator that cannot
appear in a normalized part. Deduplication, search, and merge suggestions
all run through this key: two spellings that render identically must produce
one key.

The v1 key casefolds but never NFKC-normalizes. For the ASCII names the
launch tests used, casefold is the whole story and the keys match. Under
pressure — a customer registered with fullwidth characters from a Japanese
input method, a Scandinavian name stored with a ligature — the same visual
name yields two different keys: the dedup job never fires, the merge tool
sees two customers, and support absorbs the duplicate-account ticket. No
error anywhere; the data just disagrees with itself.

Pressure that breaks it: compatibility-equivalent spellings (fullwidth
forms, ligatures) and the separator interacting with normalization.
Deterministic, no wall clock, no randomness, no locale dependence —
NFKC and casefold are fixed tables.
