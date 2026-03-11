from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime


class Post(SQLModel, table=True):

    id: Optional[int] = Field(default=None, primary_key=True)

    source: str
    text: str
    language: Optional[str] = None

    author: Optional[str] = None
    url: Optional[str] = None

    created_at: datetime = Field(default_factory=datetime.utcnow)