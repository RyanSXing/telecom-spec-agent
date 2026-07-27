from collections.abc import Sequence
from math import inf
from typing import Any

from opensearchpy import helpers

from spec_copilot.models import Chunk, SearchHit


def index_mapping(dimensions: int) -> dict[str, Any]:
    return {
        "settings": {"index.knn": True},
        "mappings": {
            "properties": {
                "chunk_index": {"type": "integer"},
                "spec_id": {"type": "keyword"},
                "release": {"type": "keyword"},
                "clause_number": {"type": "keyword"},
                "clause_path": {"type": "keyword"},
                "title": {"type": "text"},
                "text": {"type": "text"},
                "source_url": {"type": "keyword", "index": False},
                "embedding": {"type": "knn_vector", "dimension": dimensions},
            }
        },
    }


def reciprocal_rank_fusion(
    bm25_hits: Sequence[SearchHit],
    vector_hits: Sequence[SearchHit],
    *,
    limit: int = 8,
    k: int = 60,
) -> list[SearchHit]:
    hits: dict[str, SearchHit] = {}
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}

    for source, source_hits in (("bm25", bm25_hits), ("vector", vector_hits)):
        for rank, hit in enumerate(source_hits, start=1):
            hits.setdefault(hit.chunk_id, hit)
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1 / (k + rank)
            ranks.setdefault(hit.chunk_id, {})[source] = rank

    ordered_ids = sorted(
        hits,
        key=lambda chunk_id: (
            -scores[chunk_id],
            ranks[chunk_id].get("bm25", inf),
            ranks[chunk_id].get("vector", inf),
            chunk_id,
        ),
    )[:limit]
    return [
        hits[chunk_id].model_copy(
            update={
                "fused_rank": fused_rank,
                "bm25_rank": ranks[chunk_id].get("bm25"),
                "vector_rank": ranks[chunk_id].get("vector"),
            }
        )
        for fused_rank, chunk_id in enumerate(ordered_ids, start=1)
    ]


class OpenSearchStore:
    def __init__(
        self,
        client: Any,
        *,
        dimensions: int,
        index_name: str = "spec-clauses-v1",
    ) -> None:
        self.client = client
        self.dimensions = dimensions
        self.index_name = index_name

    def ensure_index(self) -> None:
        if not self.client.indices.exists(index=self.index_name):
            self.client.indices.create(
                index=self.index_name,
                body=index_mapping(self.dimensions),
            )

    def index_chunks(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[list[float]],
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        actions = [
            {
                "_index": self.index_name,
                "_id": chunk.chunk_id,
                "_source": {
                    **chunk.model_dump(exclude={"chunk_id"}),
                    "embedding": vector,
                },
            }
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        if actions:
            helpers.bulk(self.client, actions)

    @staticmethod
    def _hits(response: dict[str, Any]) -> list[SearchHit]:
        return [
            SearchHit(
                chunk_id=hit["_id"],
                **{key: value for key, value in hit["_source"].items() if key != "embedding"},
            )
            for hit in response["hits"]["hits"]
        ]

    def bm25(self, question: str, size: int = 20) -> list[SearchHit]:
        response = self.client.search(
            index=self.index_name,
            body={
                "size": size,
                "query": {
                    "multi_match": {
                        "query": question,
                        "fields": ["title^2", "text"],
                    }
                },
            },
        )
        return self._hits(response)

    def knn(self, vector: list[float], size: int = 20) -> list[SearchHit]:
        response = self.client.search(
            index=self.index_name,
            body={
                "size": size,
                "query": {"knn": {"embedding": {"vector": vector, "k": size}}},
            },
        )
        return self._hits(response)

    def clause_lookup(self, clause_number: str) -> list[SearchHit]:
        response = self.client.search(
            index=self.index_name,
            body={
                "size": 100,
                "query": {"term": {"clause_number": clause_number}},
                "sort": [{"chunk_index": "asc"}],
            },
        )
        return self._hits(response)

    def count(self) -> int:
        return int(self.client.count(index=self.index_name)["count"])


class HybridRetriever:
    def __init__(self, store: Any, embedder: Any) -> None:
        self.store = store
        self.embedder = embedder

    def search(self, question: str, limit: int = 8) -> list[SearchHit]:
        vector = self.embedder.embed_query(question)
        return reciprocal_rank_fusion(
            self.store.bm25(question, size=20),
            self.store.knn(vector, size=20),
            limit=limit,
        )

    def clause_lookup(self, clause_number: str) -> list[SearchHit]:
        return self.store.clause_lookup(clause_number)
