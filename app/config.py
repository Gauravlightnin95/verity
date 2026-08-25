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

    # Swytchcode execution kernel (agents/swytchcode_client.py). The token is
    # only needed for registry features (`swytchcode get`); `swytchcode exec`
    # runs offline against the committed .swytchcode/ bundles, so leaving this
    # empty degrades nothing at verify time. swytchcode_dry_run=True makes every
    # kernel call report the request it *would* have made without issuing it -
    # useful for demoing the execution-policy layer without burning API quota.
    swytchcode_token: str | None = None
    swytchcode_dry_run: bool = False

    # Groq free tier via console.groq.com - fast, no card required. The
    # pipeline makes only ~4 LLM calls per verify (batched stance), so the
    # free 30-req/min limit is comfortable. If you hit the per-minute token
    # limit, "openai/gpt-oss-20b" has higher throughput (lower quality).
    #
    # Groq retires model ids without notice - every Llama chat model was
    # decommissioned, which surfaces as a 404 "model does not exist" from
    # claim extraction (not an auth error, so the key is fine). List what a
    # key can actually reach with:
    #   curl https://api.groq.com/openai/v1/models -H "Authorization: Bearer $GROQ_API_KEY"
    # Whatever you pick must support function calling - the three LLM agents
    # all go through with_structured_output(). qwen/qwen3.6-27b does not.
    llm_model: str = "openai/gpt-oss-120b"

    device_priority: list[str] = ["NPU", "GPU", "CPU"]

    fusion_weights: FusionWeights = FusionWeights()

    # Streamlit UI -> FastAPI backend. Timeout is generous: a real run does
    # web search + up to ~3 LLM calls, which can take 15-40s cold.
    backend_base_url: str = "http://localhost:8000"
    backend_timeout_seconds: int = 120

    # AI-text detector. detector_local=False (default) means detect_ai_text()
    # never attempts to load a local model - statistics-only, no download.
    # transformers/torch ship with a plain `uv sync` (the claim extractor's
    # check-worthiness filter needs them), so flipping this on only costs
    # the GPT-2 weights download, not a dependency install.
    detector_local: bool = False
    detector_model_name: str = "roberta-base-openai-detector"
    detector_min_words: int = 150

    streamlit_server_port: int = 8501
    streamlit_server_address: str = "0.0.0.0"

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: Literal["json", "console"] = "console"
    log_file_path: Path = BASE_DIR / "logs" / "verity.log"


settings = Settings()
