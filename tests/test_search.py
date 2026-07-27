from pathlib import Path
from types import SimpleNamespace

from spec_copilot.embeddings import EmbeddingCache, EmbeddingService, VertexEmbeddingBackend
from spec_copilot.models import Chunk, SearchHit
from spec_copilot.search import (
    HybridRetriever,
    OpenSearchStore,
    index_mapping,
    reciprocal_rank_fusion,
)


class FakeEmbeddingBackend:
    model = "fake-model"
    dimensions = 3

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], str]] = []

    def embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        self.calls.append((texts, task_type))
        return [[float(len(text)), 1.0, 0.0] for text in texts]


def test_embedding_service_reuses_sqlite_cache(tmp_path: Path) -> None:
    backend = FakeEmbeddingBackend()
    service = EmbeddingService(backend, EmbeddingCache(tmp_path / "embeddings.sqlite3"))

    first = service.embed_documents(["AMF", "UPF"])
    second = service.embed_documents(["AMF", "UPF"])
    query = service.embed_query("AMF")

    assert first == second
    assert query == [3.0, 1.0, 0.0]
    assert backend.calls == [
        (["AMF", "UPF"], "RETRIEVAL_DOCUMENT"),
        (["AMF"], "RETRIEVAL_QUERY"),
    ]


def test_vertex_backend_retries_transient_failure_without_cloud_credentials() -> None:
    attempts = 0

    class Models:
        def embed_content(self, **kwargs: object) -> SimpleNamespace:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise RuntimeError("temporary")
            return SimpleNamespace(
                embeddings=[SimpleNamespace(values=[1.0, 2.0]) for _ in kwargs["contents"]]
            )

    backend = VertexEmbeddingBackend(
        client=SimpleNamespace(models=Models()),
        model="gemini-embedding-001",
        dimensions=2,
        sleep=lambda _: None,
    )

    assert backend.embed(["one", "two"], "RETRIEVAL_DOCUMENT") == [
        [1.0, 2.0],
        [1.0, 2.0],
    ]
    assert attempts == 3


def hit(chunk_id: str, clause: str) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        chunk_index=0,
        spec_id="TS 23.501",
        release="Rel-17",
        clause_number=clause,
        title=f"Clause {clause}",
        clause_path=[clause],
        text=f"Text for {clause}",
        source_url="https://example.test/spec.zip",
    )


def test_reciprocal_rank_fusion_is_deterministic_and_deduplicates() -> None:
    bm25 = [hit("a", "1"), hit("b", "2"), hit("c", "3")]
    vector = [hit("b", "2"), hit("a", "1"), hit("d", "4")]

    fused = reciprocal_rank_fusion(bm25, vector, limit=3, k=60)

    assert [item.chunk_id for item in fused] == ["a", "b", "c"]
    assert [(item.bm25_rank, item.vector_rank) for item in fused] == [
        (1, 2),
        (2, 1),
        (3, None),
    ]
    assert [item.fused_rank for item in fused] == [1, 2, 3]


def test_index_mapping_has_searchable_text_and_exact_vector_dimension() -> None:
    mapping = index_mapping(768)
    properties = mapping["mappings"]["properties"]

    assert mapping["settings"]["index.knn"] is True
    assert properties["text"]["type"] == "text"
    assert properties["clause_number"]["type"] == "keyword"
    assert properties["embedding"] == {"type": "knn_vector", "dimension": 768}


def test_opensearch_store_creates_mapping_and_indexes_by_content_hash(
    monkeypatch,
) -> None:
    created: list[dict[str, object]] = []
    actions: list[dict[str, object]] = []

    class Indices:
        def exists(self, index: str) -> bool:
            return False

        def create(self, **kwargs: object) -> None:
            created.append(kwargs)

    client = SimpleNamespace(indices=Indices())
    monkeypatch.setattr(
        "spec_copilot.search.helpers.bulk",
        lambda client, entries: actions.extend(entries),
    )
    store = OpenSearchStore(client, dimensions=3)
    chunk = Chunk(
        chunk_id="hash",
        chunk_index=0,
        spec_id="TS 23.501",
        release="Rel-17",
        clause_number="4.2.6",
        title="Service-based interfaces",
        clause_path=["4", "4.2", "4.2.6"],
        text="Service-based interface text.",
        source_url="https://example.test/spec.zip",
    )

    store.ensure_index()
    store.index_chunks([chunk], [[0.1, 0.2, 0.3]])

    assert created[0]["index"] == "spec-clauses-v1"
    assert created[0]["body"] == index_mapping(3)
    assert actions[0]["_id"] == "hash"
    assert actions[0]["_source"]["embedding"] == [0.1, 0.2, 0.3]


def test_clause_lookup_returns_chunks_in_document_order() -> None:
    class Client:
        def search(self, **kwargs: object) -> dict[str, object]:
            assert kwargs["body"]["query"]["term"] == {"clause_number": "4.2.6"}
            return {
                "hits": {
                    "hits": [
                        {
                            "_id": "part-1",
                            "_source": {
                                "chunk_index": 1,
                                "spec_id": "TS 23.501",
                                "release": "Rel-17",
                                "clause_number": "4.2.6",
                                "title": "Service-based interfaces",
                                "clause_path": ["4", "4.2", "4.2.6"],
                                "text": "Second part.",
                                "source_url": "https://example.test/spec.zip",
                            },
                        }
                    ]
                }
            }

    hits = OpenSearchStore(Client(), dimensions=3).clause_lookup("4.2.6")

    assert [item.chunk_id for item in hits] == ["part-1"]
    assert hits[0].chunk_index == 1


def test_hybrid_retriever_embeds_query_and_fuses_store_results() -> None:
    class Embedder:
        def embed_query(self, text: str) -> list[float]:
            assert text == "What is an AMF?"
            return [0.1, 0.2]

    class Store:
        def bm25(self, question: str, size: int = 20) -> list[SearchHit]:
            assert size == 20
            return [hit("a", "1"), hit("b", "2")]

        def knn(self, vector: list[float], size: int = 20) -> list[SearchHit]:
            assert vector == [0.1, 0.2]
            return [hit("b", "2")]

    results = HybridRetriever(Store(), Embedder()).search("What is an AMF?")

    assert [item.chunk_id for item in results] == ["b", "a"]
