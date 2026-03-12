from __future__ import annotations

from functools import lru_cache
from typing import List

from sqlmodel import Session

from app.models.claim import Claim
from app.models.post import Post


class IngestServiceError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _ml_ingest_fns():
    try:
        from ml.inference.claim_detector import extract_claims
        from ml.inference.fluff_filter import clean_text
    except Exception as exc:
        raise IngestServiceError(f"Unable to import ML ingest modules: {exc}") from exc

    return clean_text, extract_claims


def extract_and_store_claims(post: Post, session: Session) -> List[Claim]:
    if not post or not post.text:
        raise IngestServiceError("Post text is required for claim extraction")

    clean_text, extract_claims = _ml_ingest_fns()

    try:
        normalized = clean_text(post.text)
        extracted = extract_claims(normalized)
    except Exception as exc:
        raise IngestServiceError(f"Claim extraction failed: {exc}") from exc

    created_claims: List[Claim] = []
    for claim_text in extracted:
        if not claim_text or not claim_text.strip():
            continue

        claim = Claim(
            post_id=post.id,
            claim_text=claim_text.strip(),
            language=post.language,
        )
        session.add(claim)
        session.commit()
        session.refresh(claim)
        created_claims.append(claim)

    return created_claims
