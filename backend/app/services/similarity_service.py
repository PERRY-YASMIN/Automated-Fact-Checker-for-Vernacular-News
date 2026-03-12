from __future__ import annotations

from functools import lru_cache

import numpy as np


class SimilarityServiceError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _embed_fn():
    try:
        from ml.inference.embedder import embed_text
    except Exception as exc:
        raise SimilarityServiceError(f"Unable to import embedding module: {exc}") from exc
    return embed_text


def generate_embedding(text: str) -> np.ndarray:
    if not text or not text.strip():
        raise SimilarityServiceError("Text is empty")

    embed_text = _embed_fn()
    try:
        vector = embed_text(text)
    except Exception as exc:
        raise SimilarityServiceError(f"Embedding generation failed: {exc}") from exc

    return np.asarray(vector, dtype=np.float32)


def calculate_similarity(claim_text: str, evidence_text: str) -> float:
    claim_vec = generate_embedding(claim_text)
    evidence_vec = generate_embedding(evidence_text)

    denominator = float(np.linalg.norm(claim_vec) * np.linalg.norm(evidence_vec))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(claim_vec, evidence_vec) / denominator)