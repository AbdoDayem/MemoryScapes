import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import globe, memories

app = FastAPI(title="MemoryScapes API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_methods=["GET", "POST", "PATCH"],
    allow_headers=["*"],
    expose_headers=["Content-Length", "Content-Range"],
)

app.include_router(memories.router, prefix="/memories", tags=["memories"])
app.include_router(globe.router, prefix="/globe", tags=["globe"])


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
