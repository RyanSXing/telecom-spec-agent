from typing import Literal

from pydantic import BaseModel, Field


class Clause(BaseModel):
    spec_id: str
    release: str
    clause_number: str
    title: str
    clause_path: list[str]
    text: str = ""
    tables: list[str] = Field(default_factory=list)
    source_url: str


class Chunk(BaseModel):
    chunk_id: str
    chunk_index: int
    spec_id: str
    release: str
    clause_number: str
    title: str
    clause_path: list[str]
    text: str
    source_url: str


class SearchHit(Chunk):
    fused_rank: int | None = None
    bm25_rank: int | None = None
    vector_rank: int | None = None


class Citation(BaseModel):
    chunk_id: str
    spec_id: str
    release: str
    clause_number: str
    title: str
    quoted_span: str


class Answer(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: Literal["high", "medium", "low"]
    unanswered_aspects: list[str] = Field(default_factory=list)
