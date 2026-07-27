import hashlib
import json
import sqlite3
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types


class EmbeddingCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    model TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    task_type TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    vector TEXT NOT NULL,
                    PRIMARY KEY (model, dimensions, task_type, content_hash)
                )
                """
            )

    def get(
        self,
        model: str,
        dimensions: int,
        task_type: str,
        content_hash: str,
    ) -> list[float] | None:
        with sqlite3.connect(self.path) as connection:
            row = connection.execute(
                """
                SELECT vector FROM embeddings
                WHERE model = ? AND dimensions = ? AND task_type = ? AND content_hash = ?
                """,
                (model, dimensions, task_type, content_hash),
            ).fetchone()
        return list(json.loads(row[0])) if row else None

    def put(
        self,
        model: str,
        dimensions: int,
        task_type: str,
        content_hash: str,
        vector: list[float],
    ) -> None:
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?, ?, ?)",
                (model, dimensions, task_type, content_hash, json.dumps(vector)),
            )


class VertexEmbeddingBackend:
    def __init__(
        self,
        *,
        model: str,
        dimensions: int,
        project: str | None = None,
        location: str = "us-central1",
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model
        self.dimensions = dimensions
        self.client = client or genai.Client(
            vertexai=True,
            project=project,
            location=location,
        )
        self.sleep = sleep

    def embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 32):
            batch = texts[start : start + 32]
            for attempt in range(3):
                try:
                    response = self.client.models.embed_content(
                        model=self.model,
                        contents=batch,
                        config=types.EmbedContentConfig(
                            task_type=task_type,
                            output_dimensionality=self.dimensions,
                        ),
                    )
                    embeddings = response.embeddings or []
                    vectors.extend([list(embedding.values or []) for embedding in embeddings])
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    self.sleep(2**attempt)
        return vectors


class EmbeddingService:
    def __init__(self, backend: Any, cache: EmbeddingCache) -> None:
        self.backend = backend
        self.cache = cache

    def _embed(self, texts: Sequence[str], task_type: str) -> list[list[float]]:
        hashes = [hashlib.sha256(text.encode()).hexdigest() for text in texts]
        results: list[list[float] | None] = [
            self.cache.get(
                self.backend.model,
                self.backend.dimensions,
                task_type,
                content_hash,
            )
            for content_hash in hashes
        ]
        missing_indices = [index for index, vector in enumerate(results) if vector is None]
        if missing_indices:
            missing_vectors = self.backend.embed(
                [texts[index] for index in missing_indices],
                task_type,
            )
            if len(missing_vectors) != len(missing_indices):
                raise ValueError("embedding service returned an unexpected vector count")
            for index, vector in zip(missing_indices, missing_vectors, strict=True):
                results[index] = vector
                self.cache.put(
                    self.backend.model,
                    self.backend.dimensions,
                    task_type,
                    hashes[index],
                    vector,
                )
        return [vector for vector in results if vector is not None]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "RETRIEVAL_QUERY")[0]
