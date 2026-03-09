from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import py3langid


@dataclass
class LanguageInfo:
    lang: str
    score: float
    script: Optional[str]


def detect_script(text: str) -> Optional[str]:
    """Very rough script detection based on Unicode ranges."""
    for ch in text:
        code = ord(ch)
        if 0x0900 <= code <= 0x097F:
            return "Devanagari"
        if 0x0980 <= code <= 0x09FF:
            return "Bengali"
        if 0x0A00 <= code <= 0x0A7F:
            return "Gurmukhi"
        if 0x0B80 <= code <= 0x0BFF:
            return "Tamil"
        if 0x0C00 <= code <= 0x0C7F:
            return "Telugu"
        if 0x0C80 <= code <= 0x0CFF:
            return "Kannada"
        if 0x0D00 <= code <= 0x0D7F:
            return "Malayalam"
        if 0x0041 <= code <= 0x007A:
            return "Latin"
        if 0x0600 <= code <= 0x06FF:
            return "Arabic"
        if 0x0400 <= code <= 0x04FF:
            return "Cyrillic"
    return None


def detect_language(text: str) -> LanguageInfo:
    # Using the direct classify function from py3langid
    lang, score = py3langid.classify(text)
    script = detect_script(text)
    return LanguageInfo(lang=lang, score=score, script=script)


if __name__ == "__main__":
    examples = [
        "Good morning! Petrol price is very high today.",
        "आज पेट्रोल बहुत महंगा हो गया है।",
        "বাড়িতে সবাই কেমন আছেন?",
    ]
    for t in examples:
        info = detect_language(t)
        print(f"Text: {t[:30]}...")
        print(f"  -> lang={info.lang}, score={info.score:.3f}, script={info.script}\n")