# MemoryScapes API

FastAPI service for auth, memory CRUD, presigned uploads, job status (SSE) and globe queries.

```bash
cd services/api
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --reload
```

Endpoints are stubbed and return `501` until they're implemented. See [docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md#3-api-service).
