"""FastAPI entrypoint: CORS for the future Streamlit UI, a health check,
and the /verify route."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes_verify import router as verify_router

app = FastAPI(title="VERITY", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(verify_router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
