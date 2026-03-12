from __future__ import annotations

from dataclasses import asdict, is_dataclass
from functools import lru_cache
from typing import Any


class MLServiceError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _load_ml_functions():
    try:
        from ml.inference.fluff_filter import clean_text
        from ml.inference.retrieval_pipeline import retrieve_facts
        from ml.inference.verifier import verify_claim
    except Exception as exc:
        raise MLServiceError(f"Unable to import ML modules: {exc}") from exc

    return clean_text, retrieve_facts, verify_claim


def _source_item(item: Any) -> dict[str, Any]:
    # Supports dataclass-based RetrievedFact and dict-like fallback.
    if is_dataclass(item):
        obj = asdict(item)
        fact = obj.get("fact", {})
        return {
            "id": fact.get("id"),
            "claim": fact.get("claim"),
            "language": fact.get("language"),
            "score": float(obj.get("score", 0.0)),
        }

    fact = getattr(item, "fact", None)
    if fact is not None:
        return {
            "id": getattr(fact, "id", None),
            "claim": getattr(fact, "claim", None),
            "language": getattr(fact, "language", None),
            "score": float(getattr(item, "score", 0.0)),
        }

    if isinstance(item, dict):
        return {
            "id": item.get("id") or item.get("fact_id"),
            "claim": item.get("claim") or item.get("text"),
            "language": item.get("language"),
            "score": float(item.get("score", 0.0)),
        }

    return {
        "id": None,
        "claim": str(item),
        "language": None,
        "score": 0.0,
    }


def verify_claim_logic(claim_text: str) -> dict[str, Any]:
    if not claim_text or not claim_text.strip():
        raise MLServiceError("Claim text is empty")

    clean_text, retrieve_facts, verify_claim = _load_ml_functions()

    try:
        normalized_claim = clean_text(claim_text)
        retrieved = retrieve_facts(normalized_claim, k=5)
        verification = verify_claim(normalized_claim, retrieved)
    except Exception as exc:
        raise MLServiceError(f"ML verification failed: {exc}") from exc

    return {
        "claim": normalized_claim,
        "verdict": verification.get("verdict", "NotEnoughEvidence"),
        "confidence": float(verification.get("confidence", 0.0)),
        "sources": [_source_item(item) for item in retrieved],
    }