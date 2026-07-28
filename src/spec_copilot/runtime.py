import os
from functools import lru_cache
from pathlib import Path

from opensearchpy import OpenSearch

from spec_copilot.agent import AgentService, build_agent
from spec_copilot.config import Settings
from spec_copilot.embeddings import EmbeddingCache, EmbeddingService, VertexEmbeddingBackend
from spec_copilot.search import HybridRetriever, OpenSearchStore


class DependencyUnavailable(RuntimeError):
    pass


def build_store(settings: Settings) -> OpenSearchStore:
    return OpenSearchStore(
        OpenSearch(settings.opensearch_url),
        dimensions=settings.spec_copilot_embedding_dimensions,
    )


def build_embedder(settings: Settings) -> EmbeddingService:
    if not settings.google_cloud_project:
        raise DependencyUnavailable("GOOGLE_CLOUD_PROJECT is not configured")
    try:
        backend = VertexEmbeddingBackend(
            model=settings.spec_copilot_embedding_model,
            dimensions=settings.spec_copilot_embedding_dimensions,
            project=settings.google_cloud_project,
            location=settings.google_cloud_location,
        )
    except Exception as error:
        raise DependencyUnavailable("Vertex AI credentials are unavailable") from error

    return EmbeddingService(
        backend,
        EmbeddingCache(Path(".cache/spec-copilot/embeddings.sqlite3")),
    )


def build_retriever(settings: Settings) -> tuple[HybridRetriever, OpenSearchStore]:
    store = build_store(settings)
    embedder = build_embedder(settings)
    return HybridRetriever(store, embedder), store


def build_runtime(settings: Settings) -> tuple[AgentService, OpenSearchStore]:
    if not settings.google_cloud_project:
        raise DependencyUnavailable("GOOGLE_CLOUD_PROJECT is not configured")
    os.environ.update(
        GOOGLE_GENAI_USE_VERTEXAI="true",
        GOOGLE_CLOUD_PROJECT=settings.google_cloud_project,
        GOOGLE_CLOUD_LOCATION=settings.google_cloud_location,
    )
    retriever, store = build_retriever(settings)
    pipeline = build_agent(retriever, model=settings.spec_copilot_model)
    return AgentService(pipeline), store


@lru_cache(maxsize=1)
def default_runtime() -> tuple[AgentService, OpenSearchStore]:
    return build_runtime(Settings())
