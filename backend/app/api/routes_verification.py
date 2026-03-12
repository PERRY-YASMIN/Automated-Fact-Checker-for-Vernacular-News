from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlmodel import Session, select

from app.db.session import get_session
from app.models.claim import Claim
from app.models.verdict import Verdict
from app.services.verification_service import MLServiceError, verify_claim_logic

router = APIRouter(prefix="/verify", tags=["verification"])


class VerifyRequest(BaseModel):
    text: str


@router.post("")
def verify_text(payload: VerifyRequest):
    """Verify a raw claim text via ML pipeline without DB writes."""
    try:
        return verify_claim_logic(payload.text)
    except MLServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@router.post("/claim/{claim_id}")
def verify_claim(claim_id: int, session: Session = Depends(get_session)):
    claim = session.exec(select(Claim).where(Claim.id == claim_id)).first()

    if not claim:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Claim not found")

    try:
        result = verify_claim_logic(claim.claim_text)
    except MLServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    verdict = Verdict(
        claim_id=claim_id,
        verdict=result["verdict"],
        confidence=result["confidence"],
        evidence=json.dumps(result["sources"], ensure_ascii=False),
    )

    session.add(verdict)
    session.commit()
    session.refresh(verdict)

    return result