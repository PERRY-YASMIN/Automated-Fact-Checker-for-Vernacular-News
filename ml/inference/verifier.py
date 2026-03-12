from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ml import config
from .retrieval_pipeline import Fact, RetrievedFact


@dataclass
class VerificationResult:
    label: str
    confidence: float
    best_fact_id: Optional[str]
    best_fact_score: Optional[float]
    nli_probs: Optional[dict[str, float]]


@lru_cache(maxsize=1)
def _load_nli():
    device = config.get_device()

    model_source: str | Path
    if config.VERIFIER_MODEL_DIR.exists():
        model_source = config.VERIFIER_MODEL_DIR
    else:
        model_source = config.VERIFIER_MODEL_NAME

    tokenizer = AutoTokenizer.from_pretrained(model_source)
    model = AutoModelForSequenceClassification.from_pretrained(model_source)
    if device == "cuda":
        model = model.to(torch.device("cuda"))
        model = model.half()
    model.eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()} if model.config.id2label else None
    return tokenizer, model, id2label


def _softmax(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=-1)


def _label_map(id2label: Optional[dict[int, str]], probs: torch.Tensor) -> dict[str, float]:
    if id2label:
        out: dict[str, float] = {}
        for i, p in enumerate(probs.tolist()):
            out[str(id2label.get(i, i)).lower()] = float(p)
        return out

    if probs.numel() == 3:
        con, neu, ent = probs.tolist()
        return {"contradiction": float(con), "neutral": float(neu), "entailment": float(ent)}
    return {str(i): float(p) for i, p in enumerate(probs.tolist())}


def _pairs(claim: str, retrieved: Iterable[RetrievedFact]) -> List[tuple[str, str, RetrievedFact]]:
    pairs: List[tuple[str, str, RetrievedFact]] = []
    for rf in retrieved:
        pairs.append((rf.fact.claim, claim, rf))
    return pairs


def _coerce_retrieved(facts: list) -> List[RetrievedFact]:
    coerced: List[RetrievedFact] = []
    for item in facts:
        if isinstance(item, RetrievedFact):
            coerced.append(item)
            continue
        if isinstance(item, dict):
            fact_id = str(item.get("id") or item.get("fact_id") or "unknown")
            fact_text = str(item.get("claim") or item.get("text") or "")
            language = str(item.get("language") or "unk")
            score = float(item.get("score") or 0.0)
            coerced.append(RetrievedFact(fact=Fact(id=fact_id, claim=fact_text, language=language), score=score))
    return coerced


def _label_from_probs(
    entailment: float,
    contradiction: float,
    neutral: float,
    retrieval_score: float,
) -> tuple[str, float]:
    # Strong NLI + sufficient retrieval support.
    if entailment >= config.NLI_DECISION_THRESHOLD and retrieval_score >= config.RETRIEVAL_SUPPORT_THRESHOLD:
        return "Supported", entailment
    if contradiction >= config.NLI_DECISION_THRESHOLD and retrieval_score >= config.RETRIEVAL_SUPPORT_THRESHOLD:
        return "Refuted", contradiction

    # Weak-but-consistent signal with very strong retrieval.
    if entailment >= config.NLI_WEAK_SIGNAL_THRESHOLD and retrieval_score >= config.RETRIEVAL_STRONG_THRESHOLD:
        return "Supported", entailment
    if contradiction >= config.NLI_WEAK_SIGNAL_THRESHOLD and retrieval_score >= config.RETRIEVAL_STRONG_THRESHOLD:
        return "Refuted", contradiction

    return "NotEnoughEvidence", neutral


@torch.inference_mode()
def verify_claim_against_retrieved_facts(claim_text: str, retrieved: List[RetrievedFact]) -> VerificationResult:
    if not retrieved:
        return VerificationResult(
            label="NotEnoughEvidence",
            confidence=0.0,
            best_fact_id=None,
            best_fact_score=None,
            nli_probs=None,
        )

    tokenizer, model, id2label = _load_nli()
    device = config.get_device()

    triples = _pairs(claim_text, retrieved)
    premises = [t[0] for t in triples]
    hypotheses = [t[1] for t in triples]

    enc = tokenizer(
        [p.lower() for p in premises],
        [h.lower() for h in hypotheses],
        truncation=True, padding=True, return_tensors="pt"
    )
    if device == "cuda":
        enc = {k: v.to(torch.device("cuda")) for k, v in enc.items()}

    logits = model(**enc).logits
    probs = _softmax(logits).float().cpu()

    best_idx = 0
    best_strength = -1.0
    best_probs_map: Optional[dict[str, float]] = None
    best_label = "NotEnoughEvidence"
    best_conf = 0.0

    for i in range(probs.shape[0]):
        p_map = _label_map(id2label, probs[i])
        ent = p_map.get("entailment", 0.0)
        con = p_map.get("contradiction", 0.0)
        neu = p_map.get("neutral", 0.0)

        retrieval_score = float(triples[i][2].score)
        label, conf = _label_from_probs(ent, con, neu, retrieval_score)
        strength = conf

        if strength > best_strength:
            best_strength = strength
            best_idx = i
            best_probs_map = p_map
            best_label = label
            best_conf = conf

    rf = triples[best_idx][2]
    return VerificationResult(
        label=best_label,
        confidence=float(best_conf),
        best_fact_id=rf.fact.id,
        best_fact_score=float(rf.score),
        nli_probs=best_probs_map,
    )


def verify_claim(claim: str, facts: list) -> dict:
    """Standard verifier interface for backend-facing inference calls."""
    retrieved = _coerce_retrieved(facts)
    result = verify_claim_against_retrieved_facts(claim, retrieved)
    return {
        "verdict": result.label,
        "confidence": result.confidence,
        "best_fact_id": result.best_fact_id,
        "best_fact_score": result.best_fact_score,
        "nli_probs": result.nli_probs,
    }
