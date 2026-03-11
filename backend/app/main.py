import app.api.routes_verification as routes_verification
from app.models.verdict import Verdict

from app.api.routes_claims import router as claim_router
from app.models.claim import Claim
from fastapi import FastAPI
from sqlmodel import SQLModel
from app.db.session import engine

# IMPORTANT: import the model so SQLModel registers the table
from app.models.post import Post

# router import
from app.api.routes_posts import router as post_router

app = FastAPI(
    title="Automated Vernacular Fact Checking System",
    version="1.0.0"
)

app.include_router(post_router)
app.include_router(claim_router)
app.include_router(routes_verification.router)

@app.on_event("startup")
def on_startup():
    SQLModel.metadata.create_all(engine)


@app.get("/")
def root():
    return {"message": "Fact Checking Backend Running"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}