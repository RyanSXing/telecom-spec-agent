from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google.auth.exceptions import GoogleAuthError
from google.genai.errors import APIError
from opensearchpy.exceptions import OpenSearchException
from pydantic import BaseModel, Field, field_validator

from spec_copilot.config import Settings
from spec_copilot.models import Answer
from spec_copilot.runtime import (
    DependencyUnavailable,
    build_store,
    default_runtime,
)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    opensearch: Literal["ok", "error"]
    vertex: Literal["configured", "missing"]
    index_document_count: int


def create_app(
    *,
    agent_service=None,
    store=None,
    settings: Settings | None = None,
) -> FastAPI:
    app_settings = settings or Settings()
    app = FastAPI(title="Spec Copilot", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_settings.origin_list,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.post("/query", response_model=Answer)
    async def query(request: QueryRequest) -> Answer:
        try:
            service = agent_service or default_runtime()[0]
            return await service.ask(request.question)
        except (DependencyUnavailable, OpenSearchException, GoogleAuthError, APIError) as error:
            raise HTTPException(
                status_code=503,
                detail="A required dependency is unavailable.",
            ) from error

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        search_store = store
        if search_store is None:
            try:
                search_store = build_store(app_settings)
            except Exception:
                search_store = None
        try:
            count = search_store.count() if search_store is not None else 0
            opensearch_status = "ok"
        except Exception:
            count = 0
            opensearch_status = "error"
        vertex_status = "configured" if app_settings.google_cloud_project else "missing"
        status = "ok" if opensearch_status == "ok" and vertex_status == "configured" else "degraded"
        return HealthResponse(
            status=status,
            opensearch=opensearch_status,
            vertex=vertex_status,
            index_document_count=count,
        )

    return app


app = create_app()
