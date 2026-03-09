from __future__ import annotations

from dataclasses import dataclass
from typing import List

import re

from .text_cleaning import normalize_for_embedding, simple_sentence_split


@dataclass
class Claim:
    text: str
    source_post_id: str
    sentence_index: int


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


def is_potential_claim(sentence: str) -> bool:
    s = f" {sentence.strip()} "
    if len(s) < 20:
        return False
    # contains digit?
    if any(ch.isdigit() for ch in s):
        return True
    # simple verb hints
    return any(h in s for h in VERB_HINTS)


def extract_claims_from_post(post_id: str, text: str) -> List[Claim]:
    normalized = normalize_for_embedding(text)
    sentences = simple_sentence_split(normalized)
    claims: List[Claim] = []
    for idx, sent in enumerate(sentences):
        if is_potential_claim(sent):
            claims.append(Claim(text=sent, source_post_id=post_id, sentence_index=idx))
    # Fallback: if we got nothing but there is text, take full normalized text
    if not claims and normalized:
        claims.append(Claim(text=normalized, source_post_id=post_id, sentence_index=0))
    return claims


if __name__ == "__main__":
    example_post_id = "post1"
    example_text = "Good morning! आज पेट्रोल की कीमत 120 रुपये प्रति लीटर हो गई है. कृपया सबको बताएं."
    claims = extract_claims_from_post(example_post_id, example_text)
    for c in claims:
        print(c)