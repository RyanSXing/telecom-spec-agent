from uuid import uuid4

import pytest
from opensearchpy import OpenSearch

from spec_copilot.models import Chunk
from spec_copilot.search import OpenSearchStore, reciprocal_rank_fusion


@pytest.mark.integration
def test_real_opensearch_supports_index_bm25_knn_and_lookup() -> None:
    client = OpenSearch("http://localhost:9200")
    index_name = f"test-spec-copilot-{uuid4().hex}"
    store = OpenSearchStore(client, dimensions=3, index_name=index_name)
    chunks = [
        Chunk(
            chunk_id="mobility",
            chunk_index=0,
            spec_id="TS 23.501",
            release="Rel-17",
            clause_number="5.3",
            title="Mobility management",
            clause_path=["5", "5.3"],
            text="The AMF handles mobility management.",
            source_url="https://example.test/spec.zip",
        ),
        Chunk(
            chunk_id="session",
            chunk_index=0,
            spec_id="TS 23.501",
            release="Rel-17",
            clause_number="5.6",
            title="Session management",
            clause_path=["5", "5.6"],
            text="The SMF handles session management.",
            source_url="https://example.test/spec.zip",
        ),
    ]

    try:
        store.ensure_index()
        store.index_chunks(chunks, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        client.indices.refresh(index=index_name)

        bm25 = store.bm25("mobility", size=2)
        vector = store.knn([1.0, 0.0, 0.0], size=2)
        fused = reciprocal_rank_fusion(bm25, vector)

        assert bm25[0].chunk_id == "mobility"
        assert vector[0].chunk_id == "mobility"
        assert fused[0].chunk_id == "mobility"
        assert store.clause_lookup("5.6")[0].chunk_id == "session"
    finally:
        if client.indices.exists(index=index_name):
            client.indices.delete(index=index_name)
