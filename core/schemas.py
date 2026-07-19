"""Shared data contracts for the VERITY pipeline.

Every inter-agent boundary in this project passes one of the Pydantic
models defined here - never a raw dict. This file is the "frozen"
contract all four team workstreams build against; changing a field here
requires updating mocks/ and every consumer in the same change.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class VerdictLabel(str, Enum):
    TRUE = "TRUE"
    MOSTLY_TRUE = "MOSTLY_TRUE"
    MIXED = "MIXED"
    MISLEADING = "MISLEADING"
    FALSE = "FALSE"
    UNVERIFIABLE = "UNVERIFIABLE"
    SATIRE_OPINION = "SATIRE_OPINION"


class InputPayload(BaseModel):
    """What the user sent us: text, a URL, or a path to an uploaded image."""

    model_config = ConfigDict(extra="forbid")

    input_type: Literal["text", "url", "image"]
    text: str | None = None
    url: str | None = None
    image_path: str | None = None

    @model_validator(mode="after")
    def _matching_field_present(self) -> "InputPayload":
        field = {"text": self.text, "url": self.url, "image": self.image_path}[self.input_type]
        if not field:
            raise ValueError(f"input_type is '{self.input_type}' but its field is empty")
        return self


class ArticleContent(BaseModel):
    """A scraped or passed-through article, ready for claim extraction."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = None
    body: str
    author: str | None = None
    publish_date: str | None = None
    domain: str = ""
    partial: bool = False


class OCRRegion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    region_type: Literal["masthead", "dateline", "headline", "body"]
    text: str
    confidence: float = Field(ge=0, le=1)
    bbox: tuple[int, int, int, int]


class OCRResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    regions: list[OCRRegion] = Field(default_factory=list)
    full_text: str = ""
    language: str = "en"
    device_used: Literal["NPU", "GPU", "CPU"] = "CPU"
    duration_ms: int = 0


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str
    text: str
    entities: list[str] = Field(default_factory=list)
    location: str | None = None
    event_date: str | None = None
    checkability: Literal["high", "medium", "low"] = "medium"
    genre: Literal["report", "opinion", "satire"] = "report"


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    claim_id: str
    snippet: str
    url: str
    source_name: str
    source_tier: int = Field(ge=1, le=3)
    published_date: str | None = None
    retrieved_at: datetime


class StanceResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str
    evidence_id: str
    stance: Literal["SUPPORTS", "REFUTES", "NEUTRAL"]
    rationale: str = ""


class Signal(BaseModel):
    """A generic weighted clue - forensics, credibility, ai_text, and the
    fusion-computed evidence_stance all emit this same shape."""

    model_config = ConfigDict(extra="forbid")

    name: str
    score: float = Field(ge=0, le=1)
    weight: float = Field(ge=0, le=1)
    note: str = ""
    extras: dict[str, Any] | None = None


class CheckLog(BaseModel):
    """One entry in the pipeline's transparency trail: which agent ran,
    what happened, how long it took."""

    model_config = ConfigDict(extra="forbid")

    agent_name: str
    status: Literal["ok", "skipped", "failed"]
    duration_ms: int = 0
    note: str = ""


class PerClaimVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str
    label: VerdictLabel
    supporting: int = 0
    refuting: int = 0


class Verdict(BaseModel):
    """The final answer, always shipped with its evidence trail."""

    model_config = ConfigDict(extra="forbid")

    label: VerdictLabel
    confidence: float = Field(ge=0, le=1)
    per_claim: list[PerClaimVerdict] = Field(default_factory=list)
    evidence_citations: list[EvidenceItem] = Field(default_factory=list)
    signals: list[Signal] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    checks_performed: list[CheckLog] = Field(default_factory=list)

    @model_validator(mode="after")
    def _no_citations_implies_unverifiable(self) -> "Verdict":
        # SATIRE_OPINION is a genre classification, not an evidence-backed
        # claim, so it is exempt from the "citations required" rule.
        exempt = {VerdictLabel.UNVERIFIABLE, VerdictLabel.SATIRE_OPINION}
        if not self.evidence_citations and self.label not in exempt:
            raise ValueError(
                "a Verdict with no evidence_citations must have label UNVERIFIABLE "
                f"or SATIRE_OPINION, got {self.label!r}"
            )
        return self


class VerityState(BaseModel):
    """The shared pipeline state ('baton') LangGraph nodes read from and
    write to. Nodes never import each other directly - e.g. the forensics
    agent's masthead check writes `publication_guess` here, and the
    retrieval agent reads it from here, with no direct coupling."""

    model_config = ConfigDict(extra="forbid")

    input_payload: InputPayload | None = None
    article: ArticleContent | None = None
    ocr_result: OCRResult | None = None
    publication_guess: str | None = None
    language: str | None = None
    reject_reason: str | None = None
    claims: list[Claim] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    stance_results: list[StanceResult] = Field(default_factory=list)
    signals: list[Signal] = Field(default_factory=list)
    checks_performed: list[CheckLog] = Field(default_factory=list)
    verdict: Verdict | None = None
