from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from ml import config

# This module wraps the multilingual sentence embedding model configured in
# ml/config.py.  The default model is LaBSE (Language-agnostic BERT Sentence
# Embeddings), which produces 768-dim L2-normalised vectors that are
# *language-agnostic* — i.e., semantically equivalent sentences in Hindi and
# English are mapped to nearly the same point in vector space.  This enables
# cross-lingual retrieval: a Hindi query surfaces English KB facts and vice
# versa with no translation step.


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Load embedding model from config (cached after first call)."""
    return SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=config.get_device())


def embed_texts(texts: List[str]) -> np.ndarray:
    """Encode a list of texts to L2-normalised float32 vectors.

    Works for both Hindi and English (and any of LaBSE's 109 supported
    languages) — cross-lingual cosine similarity between the resulting
    vectors is meaningful out of the box.
    """
    if not texts:
        return np.zeros((0, 768), dtype="float32")
    model = get_embedding_model()
    batch_size = 32 if config.get_device() == "cuda" else 16
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype("float32")


if __name__ == "__main__":
    # Quick cross-lingual similarity demo
    pairs = [
        ("The Earth is the third planet from the Sun.", "पृथ्वी सूर्य से तीसरा ग्रह है।"),
        ("500 and 1000 rupee notes were demonetized in 2016.",
         "500 और 1000 के नोटों को 2016 में बंद किया गया था।"),
    ]
    all_texts = [t for pair in pairs for t in pair]
    vecs = embed_texts(all_texts)
    for i, (en, hi) in enumerate(pairs):
        sim = float(np.dot(vecs[i * 2], vecs[i * 2 + 1]))
        print(f"Pair {i+1} EN<->HI similarity: {sim:.4f}")
        print(f"  EN: {en}")
        print(f"  HI: {hi}")
    print("\nEmbedding shape:", vecs.shape)