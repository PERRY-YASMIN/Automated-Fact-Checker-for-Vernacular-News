from app.services.similarity_service import calculate_similarity
from fastapi import APIRouter, Depends
from sqlmodel import Session, select
from app.services.verification_service import verify_claim_logic
from app.db.session import get_session
from app.models.claim import Claim
from app.models.verdict import Verdict

router = APIRouter(prefix="/verify", tags=["verification"])


@router.post("/claim/{claim_id}")
def verify_claim(claim_id: int, session: Session = Depends(get_session)):

    claim = session.exec(select(Claim).where(Claim.id == claim_id)).first()

    if not claim:
        return {"error": "Claim not found"}

    # get wikipedia evidence
    result = verify_claim_logic(claim.claim_text)
    evidence = result["evidence"]

    # AI similarity calculation
    similarity = calculate_similarity(claim.claim_text, evidence)

    if similarity > 0.75:
        verdict_text = "False"
        confidence = similarity
    elif similarity > 0.5:
        verdict_text = "Needs Review"
        confidence = similarity
    else:
        verdict_text = "Unrelated"
        confidence = similarity

    verdict = Verdict(
        claim_id=claim_id,
        verdict=verdict_text,
        confidence=confidence,
        evidence=evidence
    )

    session.add(verdict)
    session.commit()
    session.refresh(verdict)

    return verdict