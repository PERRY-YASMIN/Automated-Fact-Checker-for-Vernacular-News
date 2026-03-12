from __future__ import annotations

import re
from typing import List

import regex as reg
from unidecode import unidecode

# Common fluff phrases (add more over time)
FLUFF_PATTERNS = [
    r"\bplease\s+share\b",
    r"\bshare\s+this\b",
    r"\bforward\s+this\b",
    r"\bज्यादा से ज्यादा लोगों तक\b",
    r"\bसबको बताएं\b",
    r"\blike\s+and\s+share\b",
    r"\bsubscribe\s+to\s+my\b",
]

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")
WHITESPACE_RE = re.compile(r"\s+")
EMOJI_RE = reg.compile(r"\p{Emoji_Presentation}|\p{Extended_Pictographic}")
PUNCT_RE = re.compile(r"[!?]{2,}")


def basic_normalize(text: str) -> str:
    text = text.strip()
    # Remove URLs, mentions, hashtags
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(" ", text)
    # Remove emojis
    text = EMOJI_RE.sub(" ", text)
    # Collapse repeated punctuation
    text = PUNCT_RE.sub("!", text)
    # Normalize whitespace
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def remove_fluff(text: str) -> str:
    # Match fluff patterns case-insensitively while preserving original casing
    # of the remaining text, so downstream NLI models are not confused by
    # unexpected all-lowercase input.
    out = text
    for pat in FLUFF_PATTERNS:
        out = re.sub(pat, " ", out, flags=re.IGNORECASE)
    out = WHITESPACE_RE.sub(" ", out)
    return out.strip()


def normalize_for_embedding(text: str) -> str:
    """
    Full normalization: URL/emoji removal, fluff removal, Unicode normalization.
    """
    t = basic_normalize(text)
    t = remove_fluff(t)
    # Optional: ASCII fallback to help some models; keep original for Indic scripts
    # Only use unidecode on primarily Latin text
    if all(ord(ch) < 128 for ch in t):
        t = unidecode(t)
    return t.strip()


def simple_sentence_split(text: str) -> List[str]:
    """
    Simple sentence splitter based on punctuation.
    Good enough for Milestone 1.
    """
    parts = re.split(r"[.!?।]+", text)
    sentences = [p.strip() for p in parts if p.strip()]
    return sentences


if __name__ == "__main__":
    example = "Good morning! आज पेट्रोल की कीमत 120 रुपये प्रति लीटर हो गई है!!! कृपया सबको बताएं. https://example.com"
    from pprint import pprint

    print("Original:", example)
    norm = normalize_for_embedding(example)
    print("Normalized:", norm)
    print("Sentences:")
    pprint(simple_sentence_split(norm))