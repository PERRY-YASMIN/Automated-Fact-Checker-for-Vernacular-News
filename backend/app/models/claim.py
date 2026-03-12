from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime


class Claim(SQLModel, table=True):

    id: Optional[int] = Field(default=None, primary_key=True)

    post_id: int
    claim_text: str

    language: Optional[str] = None

    created_at: datetime = Field(default_factory=datetime.utcnow)