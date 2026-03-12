from __future__ import annotations

from typing import Any


def verify_claim(text: str) -> dict[str, Any]:
	from ml.inference.pipeline import verify_claim as _verify_claim

	return _verify_claim(text)


def fact_check_text(text: str) -> dict[str, Any]:
	from ml.inference.pipeline import fact_check_text as _fact_check_text

	return _fact_check_text(text)


__all__ = ["verify_claim", "fact_check_text"]
