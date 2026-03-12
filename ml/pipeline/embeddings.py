from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from ml import config


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    # Downloads on first call, then cached
    # Explicitly set device so GPU is used when available.
    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=config.get_device())
    return model


def embed_texts(texts: List[str]) -> np.ndarray:
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
    texts = [
        "Petrol price is 120 rupees per litre.",
        "Government reduced LPG prices.",
    ]
    emb = embed_texts(texts)
    print("Shape:", emb.shape)
    print("First vector (first 5 dims):", emb[0][:5])