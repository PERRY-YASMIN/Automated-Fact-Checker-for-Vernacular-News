from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ml import config
from .embeddings import embed_texts


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


def retrieve_for_claim(
    claim_text: str,
    fact_embeddings: np.ndarray,
    facts: List[Fact],
    top_k: int = None,
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


if __name__ == "__main__":
    facts = load_facts()
    fact_embs, facts = build_fact_index(facts)
    claim = "आज पेट्रोल की कीमत 120 रुपये प्रति लीटर हो गई है।"
    print("Claim:", claim)
    matches = retrieve_for_claim(claim, fact_embs, facts)
    for m in matches:
        print(f"-> {m.fact.id}: {m.fact.claim} (score={m.score:.3f})")