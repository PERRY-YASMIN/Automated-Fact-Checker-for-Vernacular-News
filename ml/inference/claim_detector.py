from __future__ import annotations

from functools import lru_cache
from typing import List

import joblib

from ml import config
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


@lru_cache(maxsize=1)
def _load_trained_detector():
    model_path = config.CLAIM_DETECTOR_MODEL_DIR / "model.joblib"
    vec_path = config.CLAIM_DETECTOR_MODEL_DIR / "vectorizer.joblib"
    if not model_path.exists() or not vec_path.exists():
        return None

    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vec_path)
        return model, vectorizer
    except Exception:
        return None


def extract_claims(text: str) -> List[str]:
    """Extract sentence-level claim candidates from raw post text."""
    normalized = clean_text(text)
    sentences = split_sentences(normalized)

    loaded = _load_trained_detector()
    if loaded and sentences:
        model, vectorizer = loaded
        try:
            x = vectorizer.transform(sentences)
            preds = model.predict(x)
            claims = [s for s, pred in zip(sentences, preds) if int(pred) == 1 and s.strip()]
            if claims:
                return claims
        except Exception:
            # Fall back to heuristic extraction if the trained pipeline fails.
            pass

    claims: List[str] = []
    for sentence in sentences:
        if _is_potential_claim(sentence):
            claims.append(sentence)

    if not claims and normalized:
        claims.append(normalized)
    return claims
