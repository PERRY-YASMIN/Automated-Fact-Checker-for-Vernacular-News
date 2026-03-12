from fastapi import APIRouter, Depends
from sqlmodel import Session, select

from app.db.session import get_session
from app.models.post import Post
from app.models.claim import Claim

router = APIRouter()


@router.post("/extract/claims/{post_id}")
def extract_claims(post_id: int, session: Session = Depends(get_session)):

    post = session.exec(select(Post).where(Post.id == post_id)).first()

    if not post:
        return {"error": "Post not found"}

    sentences = post.text.split(".")

    created_claims = []

    for s in sentences:
        s = s.strip()

        if s:
            claim = Claim(
                post_id=post_id,
                claim_text=s,
                language=post.language
            )

            session.add(claim)
            session.commit()
            session.refresh(claim)

            created_claims.append(claim)

    return created_claims