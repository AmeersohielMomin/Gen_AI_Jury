"""Pydantic schemas for the GenAI Jury FastAPI backend."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class CGANSampleRequest(BaseModel):
    diagnosis: str = Field(default="SCC", description="ISIC 2019 diagnosis code (e.g. SCC, MEL, NV).")
    seed: int = Field(default=146, ge=0, le=999_999)
    count: int = Field(default=4, ge=1, le=8)


class QuizGuess(BaseModel):
    panel_id: str
    choice: Literal["real", "synthetic", "unsure"]


class QuizScoreRequest(BaseModel):
    guesses: list[QuizGuess]


class ReportExportRequest(BaseModel):
    case_label: str = Field(default="Uploaded Dermoscopic Image", max_length=160)
    reviewer_notes: str = Field(default="", max_length=1000)
    analysis: dict[str, Any]
