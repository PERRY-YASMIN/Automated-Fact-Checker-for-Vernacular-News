from fastapi import APIRouter, Depends
from sqlmodel import Session
from app.db.session import get_session
from app.models.post import Post

router = APIRouter()


@router.post("/ingest/post")
def ingest_post(post: Post, session: Session = Depends(get_session)):
    session.add(post)
    session.commit()
    session.refresh(post)
    return post