from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ml import config
from .embedder import embed_texts


@dataclass
class Fact:
    id: str
    claim: str
    language: str


@dataclass
class RetrievedFact:
    fact: Fact
    score: float


def load_facts(path: Path | None = None) -> List[Fact]:
    if path is None:
        path = config.VERIFIED_FACTS_PATH
    facts: List[Fact] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        obj = json.loads(line)
        facts.append(Fact(id=obj["id"], claim=obj["claim"], language=obj.get("language", "unk")))
    return facts


def build_fact_index(facts: List[Fact]) -> Tuple[np.ndarray, List[Fact]]:
    texts = [f.claim for f in facts]
    embeddings = embed_texts(texts)
    return embeddings, facts


@lru_cache(maxsize=1)
def _cached_index() -> Tuple[np.ndarray, List[Fact]]:
    facts = load_facts()
    return build_fact_index(facts)


def retrieve_for_claim(
    claim_text: str,
    fact_embeddings: np.ndarray,
    facts: List[Fact],
    top_k: int | None = None,
) -> List[RetrievedFact]:
    if top_k is None:
        top_k = config.TOP_K_FACTS
    claim_vec = embed_texts([claim_text])
    sims = cosine_similarity(claim_vec, fact_embeddings)[0]
    idx_sorted = np.argsort(-sims)[:top_k]

    results: List[RetrievedFact] = []
    for idx in idx_sorted:
        score = float(sims[idx])
        if score < config.MIN_SIMILARITY:
            continue
        results.append(RetrievedFact(fact=facts[idx], score=score))
    return results


def retrieve_facts(claim: str, k: int = 5) -> List[RetrievedFact]:
    """Retrieve top-k fact candidates for a claim."""
    fact_embeddings, facts = _cached_index()
    return retrieve_for_claim(claim, fact_embeddings, facts, top_k=k)
