"""Fencing for untrusted text that enters role prompts (finding 6).

Candidate-derived text — captured stdout/stderr, exception strings, gate
rejection reasons — is DATA, never instructions. It is rendered between
explicit untrusted-data markers, stripped of control and invisible
characters, and the consuming role's system prompt (the reflector's) names
both markers and says everything between them is recorded output to
diagnose, never directives to follow.

This is a leaf module on purpose: both :mod:`evogine.swarm.runner` (the
renderer) and :mod:`evogine.swarm.roles` (the prompt templates) import it,
so it must not import either back.
"""

from __future__ import annotations

import unicodedata

#: Explicit untrusted-data markers. The markers alone cannot stop a payload
#: that forges them — the fence's teeth are the role prompt that defines
#: what the markers mean.
UNTRUSTED_OPEN = "<<<UNTRUSTED-DATA"
UNTRUSTED_CLOSE = "UNTRUSTED-DATA>>>"

#: Cap on the fenced payload, kept from the gate's own 500-char detail cap
#: (``code_gate._detail``): the fence bounds the render, not the source.
UNTRUSTED_CAP = 500

#: Control (Cc) characters become spaces; format (Cf) characters — bidi and
#: zero-width marks that make text render differently than it reads — vanish.
_UNSAFE_CATEGORY = "Cc"
_INVISIBLE_CATEGORY = "Cf"


def strip_unsafe_characters(text: str) -> str:
    """Reduce untrusted text to safe visible characters.

    Control characters (including CR/LF, ESC, and NUL) cannot structure a
    prompt block or trigger terminal escapes; format characters cannot hide
    or reorder content. Newlines collapse to spaces: the fence is one
    record, not a page an injected line could escape.
    """
    kept: list[str] = []
    for char in text:
        category = unicodedata.category(char)
        if category == _UNSAFE_CATEGORY:
            kept.append(" ")
        elif category != _INVISIBLE_CATEGORY:
            kept.append(char)
    return "".join(kept)


def fence_untrusted(text: str) -> str:
    """Wrap untrusted text in explicit untrusted-data markers.

    The markers alone cannot stop a payload that forges them, so the
    fence's teeth are the consuming role's system prompt: it names both
    markers and instructs the model to treat everything between them as
    recorded output to diagnose, never directives to follow.
    """
    payload = strip_unsafe_characters(text)[:UNTRUSTED_CAP]
    return f"{UNTRUSTED_OPEN} {payload} {UNTRUSTED_CLOSE}"
