from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime


class Verdict(SQLModel, table=True):

    id: Optional[int] = Field(default=None, primary_key=True)

    claim_id: int
    verdict: str
    confidence: float
    evidence: str

    created_at: datetime = Field(default_factory=datetime.utcnow)