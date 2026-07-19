"""Central config: API keys, model paths, fusion weights, detector/UI/logging
settings, all loaded from .env. Every field has a safe default so the app
boots with zero environment variables set."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR: Path = Path(__file__).resolve().parent.parent


class FusionWeights(BaseModel):
    evidence_stance: float = 0.45
    archive_match: float = 0.15
    source_credibility: float = 0.10
    image_forensics: float = 0.10
    masthead: float = 0.05
    ai_text_max: float = 0.15


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Cloud API keys - all optional; missing keys degrade individual
    # capabilities rather than crashing the pipeline. Only groq_api_key is
    # really needed for full verdicts; web search falls back to keyless
    # DuckDuckGo, so tavily/gnews/factcheck are pure bonus.
    groq_api_key: str | None = None
    tavily_api_key: str | None = None
    gnews_api_key: str | None = None
    google_factcheck_api_key: str | None = None

    # Groq free tier via console.groq.com - fast, no card required. The
    # pipeline makes only ~4 LLM calls per verify (batched stance), so the
    # free 30-req/min limit is comfortable. If you hit the per-minute token
    # limit, "llama-3.1-8b-instant" has higher throughput (lower quality).
    # Verify the id at console.groq.com/docs/models if it ever 404s.
    llm_model: str = "llama-3.3-70b-versatile"

    device_priority: list[str] = ["NPU", "GPU", "CPU"]

    fusion_weights: FusionWeights = FusionWeights()

    # Streamlit UI -> FastAPI backend. Timeout is generous: a real run does
    # web search + up to ~3 LLM calls, which can take 15-40s cold.
    backend_base_url: str = "http://localhost:8000"
    backend_timeout_seconds: int = 120

    # AI-text detector. detector_local=False (default) means detect_ai_text()
    # never attempts to load a local model - statistics-only, no download.
    # Flipping it on requires `uv sync --extra local-ai-text`.
    detector_local: bool = False
    detector_model_name: str = "roberta-base-openai-detector"
    detector_min_words: int = 150

    streamlit_server_port: int = 8501
    streamlit_server_address: str = "0.0.0.0"

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: Literal["json", "console"] = "console"
    log_file_path: Path = BASE_DIR / "logs" / "verity.log"


settings = Settings()
