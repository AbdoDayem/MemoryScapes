from fastapi import FastAPI

from app.routers import globe, memories

app = FastAPI(title="MemoryScapes API", version="0.1.0")

app.include_router(memories.router, prefix="/memories", tags=["memories"])
app.include_router(globe.router, prefix="/globe", tags=["globe"])


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
