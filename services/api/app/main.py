import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.settings import REPO_DIR

# One configuration for the whole deploy: <repo>/.env (git-ignored). Real environment
# variables win. The pipeline runs it starts inherit these, so SATIE's settings and
# PUBLIC_API_URL reach the workers. Tests skip it (tests/conftest.py).
if not os.environ.get("MEMORYSCAPES_SKIP_DOTENV"):
    load_dotenv(REPO_DIR / ".env", override=False)

from app import db, pipeline  # noqa: E402 (after .env, which they read)
from app.routers import globe, memories  # noqa: E402


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    db.init_db()  # create tables and the dev user (DATABASE_URL, default a local SQLite file)
    pipeline.recover_interrupted()  # runs from a previous API process are no longer watched
    yield


app = FastAPI(title="MemoryScapes API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_methods=["GET", "POST", "PUT", "PATCH"],  # PUT: uploads
    allow_headers=["*"],
    expose_headers=["Content-Length", "Content-Range"],
)

app.include_router(memories.router, prefix="/memories", tags=["memories"])
app.include_router(globe.router, prefix="/globe", tags=["globe"])


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
