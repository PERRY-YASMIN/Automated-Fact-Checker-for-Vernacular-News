from __future__ import annotations

from typing import List

from .fluff_filter import clean_text, split_sentences

VERB_HINTS = [
    " is ",
    " are ",
    " was ",
    " were ",
    " has ",
    " have ",
    " था ",
    " थी ",
    " थे ",
    " है ",
    " हैं ",
    " होगा ",
    " रही ",
    " रहे ",
]


def _is_potential_claim(sentence: str) -> bool:
    s = f" {sentence.strip()} "
    if len(s) < 20:
        return False
    if any(ch.isdigit() for ch in s):
        return True
    return any(hint in s for hint in VERB_HINTS)


def extract_claims(text: str) -> List[str]:
    """Extract sentence-level claim candidates from raw post text."""
    normalized = clean_text(text)
    sentences = split_sentences(normalized)

    claims: List[str] = []
    for sentence in sentences:
        if _is_potential_claim(sentence):
            claims.append(sentence)

    if not claims and normalized:
        claims.append(normalized)
    return claims
