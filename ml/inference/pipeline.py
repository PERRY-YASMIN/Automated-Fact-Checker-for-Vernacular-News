from __future__ import annotations

from typing import Any

from .claim_detector import extract_claims
from .embedder import embed_text
from .fluff_filter import clean_text
from .retrieval_pipeline import retrieve_facts
from .verifier import verify_claim


def fact_check_text(text: str) -> dict[str, Any]:
    """Run end-to-end fact checking for all extracted claims in text."""
    normalized_text = clean_text(text)
    claims = extract_claims(normalized_text)

    results = []
    for claim in claims:
        _ = embed_text(claim)
        retrieved = retrieve_facts(claim, k=5)
        verification = verify_claim(claim, retrieved)
        evidence = [
            {
                "id": r.fact.id,
                "claim": r.fact.claim,
                "language": r.fact.language,
                "score": float(r.score),
            }
            for r in retrieved
        ]

        results.append(
            {
                "claim": claim,
                "verdict": verification["verdict"],
                "confidence": float(verification["confidence"]),
                "evidence": evidence,
            }
        )

    return {"claims": results}
