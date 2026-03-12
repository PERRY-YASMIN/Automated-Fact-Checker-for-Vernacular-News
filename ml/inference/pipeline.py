from __future__ import annotations

from typing import Any

from .claim_detector import extract_claims
from .fluff_filter import clean_text
from .retrieval_pipeline import retrieve_facts
from .verifier import verify_claim as verify_against_facts


def _normalize_verdict(verdict: str) -> str:
    label = verdict.strip().lower()
    if label in {"supported", "true", "entailment"}:
        return "true"
    if label in {"refuted", "false", "contradiction"}:
        return "false"
    return "not enough evidence"


def verify_claim(text: str) -> dict[str, Any]:
    """Verify a single claim text and return a frontend-friendly response.

    Uses LaBSE multilingual embeddings for retrieval, so Hindi and English
    inputs retrieve cross-lingual matching facts transparently.
    """
    normalized_claim = clean_text(text)
    # retrieve_facts embeds the claim internally via LaBSE; the cross-lingual
    # index means a Hindi query surfaces English facts and vice-versa.
    retrieved = retrieve_facts(normalized_claim, k=5)
    verification = verify_against_facts(normalized_claim, retrieved)

    return {
        "claim": normalized_claim,
        "verdict": _normalize_verdict(str(verification.get("verdict", "NotEnoughEvidence"))),
        "confidence": float(verification.get("confidence", 0.0)),
        "sources": [r.fact.claim for r in retrieved],
    }


def fact_check_text(text: str) -> dict[str, Any]:
    """Run end-to-end fact checking for all extracted claims in text."""
    normalized_text = clean_text(text)
    claims = extract_claims(normalized_text)

    results = []
    for claim in claims:
        _ = embed_text(claim)
        retrieved = retrieve_facts(claim, k=5)
        verification = verify_against_facts(claim, retrieved)
        evidence = [
            {
                "id": r.fact.id,
                "claim": r.fact.claim,
                "language": r.fact.language,
                "score": float(r.score),
            }
            for r in retrieved
        ]
        sources = [item["claim"] for item in evidence]

        results.append(
            {
                "claim": claim,
                "verdict": verification["verdict"],
                "confidence": float(verification["confidence"]),
                "evidence": evidence,
                "sources": sources,
                "normalized_verdict": _normalize_verdict(str(verification["verdict"])),
            }
        )

    return {"claims": results}
