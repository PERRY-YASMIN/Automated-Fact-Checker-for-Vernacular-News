from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from ml import config


@lru_cache(maxsize=1)
def _get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=config.get_device())


def embed_texts(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 768), dtype="float32")
    model = _get_embedding_model()
    batch_size = 32 if config.get_device() == "cuda" else 16
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype("float32")


def embed_text(text: str) -> np.ndarray:
    """Return a single embedding vector for one text."""
    vectors = embed_texts([text])
    return vectors[0]
