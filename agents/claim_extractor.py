"""Claim extraction agent (owned by Member D).

Purpose: one LLM call (structured output) decomposing an article into at
most 5 atomic, checkable claims with entities/location/date, and
classifying genre (report/opinion/satire) - opinion/satire get
checkability="low". Prompt lives in core.prompts.CLAIM_EXTRACTION_PROMPT.
Input: article (ArticleContent). Output: list[Claim] (core.schemas).
"""

from __future__ import annotations

import time
from typing import Optional, Protocol, runtime_checkable

from langchain_groq import ChatGroq
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field, ValidationError

from app.config import settings
from core.prompts import CLAIM_EXTRACTION_PROMPT
from core.schemas import Claim
from utils.logging_conf import get_logger

logger = get_logger(__name__)

# Kept low on purpose: fewer claims -> fewer downstream retrieval/stance
# calls per verify, which is what keeps LLM/search usage sane.
MAX_CLAIMS_PER_ARTICLE = 3


class ClaimExtractionError(Exception):
    """Raised when claim extraction fails after all retries are exhausted."""


@runtime_checkable
class ArticleLike(Protocol):
    """Minimal structural contract for an article this module can process -
    satisfied by core.schemas.ArticleContent without importing it directly."""

    title: Optional[str]
    body: str
    author: Optional[str]
    domain: Optional[str]


class _ClaimExtractionResponse(BaseModel):
    """Wrapper model for structured output - a single top-level object is
    more reliable for tool-calling than a bare list."""

    claims: list[Claim] = Field(default_factory=list, max_length=MAX_CLAIMS_PER_ARTICLE)


_ARTICLE_BLOCK_TEMPLATE = """\
Article title: {title}
Article author: {author}
Article source domain: {domain}

Article body:
\"\"\"
{body}
\"\"\"\
"""


class ClaimExtractor:
    """Extracts factual claims from an article using an LLM.

    Parameters
    ----------
    llm: A LangChain chat model. Defaults to ChatGroq from app.config
        settings. Injectable for testing or to swap models.
    max_claims: Hard cap on claims returned.
    max_retries: Attempts before raising ClaimExtractionError.
    retry_backoff_seconds: Base delay between retries (exponential, capped
        at 30s).
    """

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        max_claims: int = MAX_CLAIMS_PER_ARTICLE,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.0,
    ) -> None:
        self._max_claims = max_claims
        self._max_retries = max_retries
        self._retry_backoff_seconds = retry_backoff_seconds
        self._llm: BaseChatModel = llm or self._build_default_llm()
        self._structured_llm = self._llm.with_structured_output(_ClaimExtractionResponse)

    @staticmethod
    def _build_default_llm() -> BaseChatModel:
        if not settings.groq_api_key:
            raise ClaimExtractionError(
                "GROQ_API_KEY is not set. Configure it in your .env file "
                "before running claim extraction."
            )
        return ChatGroq(
            model=settings.llm_model,
            api_key=settings.groq_api_key,
            temperature=0.0,
        )

    def extract_claims(self, article: ArticleLike) -> list[Claim]:
        self._validate_article(article)

        messages = self._build_messages(article)
        response = self._invoke_with_retry(messages)
        claims = self._enforce_claim_limit(response.claims)

        logger.info("claim extraction succeeded: %d claims for %s", len(claims), article.domain)
        return claims

    def _validate_article(self, article: ArticleLike) -> None:
        if not article.body or not article.body.strip():
            raise ClaimExtractionError("cannot extract claims from an article with an empty body")

    def _build_messages(self, article: ArticleLike) -> list[HumanMessage]:
        article_block = _ARTICLE_BLOCK_TEMPLATE.format(
            title=article.title or "(no title)",
            author=article.author or "(unknown author)",
            domain=article.domain or "(unknown domain)",
            body=article.body,
        )
        prompt = CLAIM_EXTRACTION_PROMPT.format(max_claims=self._max_claims, article_text=article_block)
        return [HumanMessage(content=prompt)]

    def _invoke_with_retry(self, messages: list[HumanMessage]) -> _ClaimExtractionResponse:
        last_error: Optional[Exception] = None

        for attempt in range(1, self._max_retries + 1):
            try:
                result = self._structured_llm.invoke(messages)
                if isinstance(result, _ClaimExtractionResponse):
                    return result
                return _ClaimExtractionResponse.model_validate(result)
            except (ValidationError, ValueError) as exc:
                last_error = exc
                logger.warning("claim extraction returned invalid output (attempt %d): %s", attempt, exc)
            except Exception as exc:  # broad by design, retried
                last_error = exc
                logger.warning("claim extraction call failed (attempt %d): %s", attempt, exc)

            if attempt < self._max_retries:
                backoff = min(self._retry_backoff_seconds * (2 ** (attempt - 1)), 30.0)
                time.sleep(backoff)

        raise ClaimExtractionError(
            f"claim extraction failed after {self._max_retries} attempts: {last_error}"
        ) from last_error

    def _enforce_claim_limit(self, claims: list[Claim]) -> list[Claim]:
        if len(claims) > self._max_claims:
            logger.warning("model returned %d claims, truncating to %d", len(claims), self._max_claims)
            return claims[: self._max_claims]
        return claims


_default_extractor: Optional[ClaimExtractor] = None


def extract_claims(article: ArticleLike) -> list[Claim]:
    """Module-level entrypoint agents/graph.py calls: extract_claims(article) -> list[Claim]."""
    global _default_extractor
    if _default_extractor is None:
        _default_extractor = ClaimExtractor()
    return _default_extractor.extract_claims(article)
