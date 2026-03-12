from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ml import config
from ml.pipeline.language_id import detect_language
from .embedder import embed_texts


_ACTIVE_FACTS_FINGERPRINT: str | None = None


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


def _facts_fingerprint(path: Path) -> str:
    # Include the embedding model name so that cached vectors are invalidated
    # automatically whenever the model is changed (e.g., mpnet -> LaBSE).
    model_tag = config.EMBEDDING_MODEL_NAME.encode("utf-8")
    return hashlib.sha256(path.read_bytes() + model_tag).hexdigest()


def _cache_paths() -> tuple[Path, Path, Path]:
    cache_dir = config.RETRIEVAL_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    return (
        cache_dir / "facts_fingerprint.txt",
        cache_dir / "fact_embeddings.npy",
        cache_dir / "facts.json",
    )


def _load_cached_index(path: Path) -> Tuple[np.ndarray, List[Fact]] | None:
    fp_file, emb_file, facts_file = _cache_paths()
    if not (fp_file.exists() and emb_file.exists() and facts_file.exists()):
        return None

    expected = _facts_fingerprint(path)
    stored = fp_file.read_text(encoding="utf-8").strip()
    if expected != stored:
        return None

    embeddings = np.load(emb_file)
    facts_json = json.loads(facts_file.read_text(encoding="utf-8"))
    facts = [Fact(**item) for item in facts_json]
    return embeddings.astype("float32"), facts


def _save_cached_index(path: Path, embeddings: np.ndarray, facts: List[Fact]) -> None:
    fp_file, emb_file, facts_file = _cache_paths()
    np.save(emb_file, embeddings.astype("float32"))
    facts_file.write_text(
        json.dumps([{"id": f.id, "claim": f.claim, "language": f.language} for f in facts], ensure_ascii=False),
        encoding="utf-8",
    )
    fp_file.write_text(_facts_fingerprint(path), encoding="utf-8")


def build_fact_index(facts: List[Fact]) -> Tuple[np.ndarray, List[Fact]]:
    texts = [f.claim for f in facts]
    embeddings = embed_texts(texts)
    return embeddings, facts


@lru_cache(maxsize=1)
def _cached_index() -> Tuple[np.ndarray, List[Fact]]:
    global _ACTIVE_FACTS_FINGERPRINT
    facts_path = config.VERIFIED_FACTS_PATH
    current_fp = _facts_fingerprint(facts_path)

    cached = _load_cached_index(facts_path)
    if cached is not None:
        _ACTIVE_FACTS_FINGERPRINT = current_fp
        return cached

    facts = load_facts(facts_path)
    embeddings, facts = build_fact_index(facts)
    _save_cached_index(facts_path, embeddings, facts)
    _ACTIVE_FACTS_FINGERPRINT = current_fp
    return embeddings, facts


def rebuild_fact_index() -> Tuple[np.ndarray, List[Fact]]:
    """Force a rebuild of retrieval embeddings after fact KB updates."""
    global _ACTIVE_FACTS_FINGERPRINT
    _cached_index.cache_clear()
    facts = load_facts(config.VERIFIED_FACTS_PATH)
    embeddings, facts = build_fact_index(facts)
    _save_cached_index(config.VERIFIED_FACTS_PATH, embeddings, facts)
    _ACTIVE_FACTS_FINGERPRINT = _facts_fingerprint(config.VERIFIED_FACTS_PATH)
    return embeddings, facts


def retrieve_for_claim(
    claim_text: str,
    fact_embeddings: np.ndarray,
    facts: List[Fact],
    top_k: int | None = None,
) -> List[RetrievedFact]:
    if top_k is None:
        top_k = config.TOP_K_FACTS
    lang = detect_language(claim_text).lang
    min_similarity = config.MIN_SIMILARITY_HI if lang == "hi" else config.MIN_SIMILARITY

    claim_vec = embed_texts([claim_text])
    sims = cosine_similarity(claim_vec, fact_embeddings)[0]
    idx_sorted = np.argsort(-sims)[:top_k]

    results: List[RetrievedFact] = []
    for idx in idx_sorted:
        score = float(sims[idx])
        if score < min_similarity:
            continue
        results.append(RetrievedFact(fact=facts[idx], score=score))

    # Multilingual fallback: keep strongest candidates for low-resource/mismatch cases.
    if not results:
        for idx in idx_sorted:
            score = float(sims[idx])
            if score < config.MIN_SIMILARITY_FALLBACK:
                continue
            results.append(RetrievedFact(fact=facts[idx], score=score))
        if not results and len(idx_sorted) > 0:
            best_idx = int(idx_sorted[0])
            results.append(RetrievedFact(fact=facts[best_idx], score=float(sims[best_idx])))

    return results


def retrieve_facts(claim: str, k: int = 5) -> List[RetrievedFact]:
    """Retrieve top-k fact candidates for a claim."""
    global _ACTIVE_FACTS_FINGERPRINT
    current_fp = _facts_fingerprint(config.VERIFIED_FACTS_PATH)
    if _ACTIVE_FACTS_FINGERPRINT is not None and current_fp != _ACTIVE_FACTS_FINGERPRINT:
        _cached_index.cache_clear()
        _ACTIVE_FACTS_FINGERPRINT = None

    fact_embeddings, facts = _cached_index()
    return retrieve_for_claim(claim, fact_embeddings, facts, top_k=k)
