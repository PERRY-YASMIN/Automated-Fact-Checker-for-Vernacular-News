from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ml import config
from .retrieval import RetrievedFact


@dataclass
class VerificationResult:
    label: str  # Supported | Refuted | NotEnoughEvidence
    confidence: float
    best_fact_id: Optional[str]
    best_fact_score: Optional[float]
    nli_probs: Optional[dict[str, float]]


@lru_cache(maxsize=1)
def _load_nli():
    device = config.get_device()
    tokenizer = AutoTokenizer.from_pretrained(config.VERIFIER_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(config.VERIFIER_MODEL_NAME)
    if device == "cuda":
        model = model.to(torch.device("cuda"))
        # safe default on consumer GPUs
        model = model.half()
    model.eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()} if model.config.id2label else None
    return tokenizer, model, id2label


def _softmax(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=-1)


def _label_map(id2label: Optional[dict[int, str]], probs: torch.Tensor) -> dict[str, float]:
    # Try to use model-provided id2label; fallback to common XNLI ordering if absent.
    if id2label:
        out: dict[str, float] = {}
        for i, p in enumerate(probs.tolist()):
            out[str(id2label.get(i, i)).lower()] = float(p)
        return out

    # Common XNLI order: contradiction, neutral, entailment
    if probs.numel() == 3:
        c, n, e = probs.tolist()
        return {"contradiction": float(c), "neutral": float(n), "entailment": float(e)}
    return {str(i): float(p) for i, p in enumerate(probs.tolist())}


def _pairs(claim: str, retrieved: Iterable[RetrievedFact]) -> List[tuple[str, str, RetrievedFact]]:
    pairs: List[tuple[str, str, RetrievedFact]] = []
    for rf in retrieved:
        # Premise=fact, Hypothesis=claim
        pairs.append((rf.fact.claim, claim, rf))
    return pairs


@torch.inference_mode()
def verify_claim_against_retrieved_facts(
    claim_text: str,
    retrieved: List[RetrievedFact],
) -> VerificationResult:
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

    # Batch NLI over retrieved facts
    triples = _pairs(claim_text, retrieved)
    premises = [t[0] for t in triples]
    hyps = [t[1] for t in triples]

    enc = tokenizer(
        premises,
        hyps,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    if device == "cuda":
        enc = {k: v.to(torch.device("cuda")) for k, v in enc.items()}

    logits = model(**enc).logits  # [N, C]
    probs = _softmax(logits).float().cpu()  # keep on CPU for postprocessing

    # Pick the single best fact based on strongest entailment/contradiction signal
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

        if ent >= con and ent >= neu:
            label = "Supported"
            conf = ent
            strength = ent
        elif con >= ent and con >= neu:
            label = "Refuted"
            conf = con
            strength = con
        else:
            label = "NotEnoughEvidence"
            conf = neu
            strength = neu

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

