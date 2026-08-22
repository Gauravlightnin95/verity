"""FastAPI entrypoint: serves the VERITY web UI and the /verify + /report
API it calls. CORS stays open to the legacy Streamlit origin so the old UI
keeps working during the switchover."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes_verify import router as verify_router

app = FastAPI(title="VERITY", version="0.1.0")

# The web UI is served from this same app, so its calls are same-origin and
# need no CORS entry. These two are only for the Streamlit UI, which runs on
# its own port and is being retired.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(verify_router)

_WEB_DIR = Path(__file__).resolve().parent.parent / "ui" / "web"


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


# Mounted under /ui rather than at "/" so the static handler can never shadow
# an API route: a future endpoint name colliding with a filename would
# otherwise start returning a stylesheet instead of JSON.
if _WEB_DIR.is_dir():
    app.mount("/ui", StaticFiles(directory=str(_WEB_DIR)), name="ui")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(_WEB_DIR / "index.html")
