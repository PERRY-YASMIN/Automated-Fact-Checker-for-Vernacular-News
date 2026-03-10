from __future__ import annotations

from ml.inference.pipeline import fact_check_text


def process_post_text(post_text: str) -> dict:
    """Example backend worker hook into ML inference pipeline."""
    result = fact_check_text(post_text)
    return result
