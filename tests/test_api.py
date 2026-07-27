import pytest
from fastapi.testclient import TestClient
from google.auth.exceptions import GoogleAuthError

from spec_copilot.api import create_app
from spec_copilot.config import Settings
from spec_copilot.models import Answer, Citation
from spec_copilot.runtime import DependencyUnavailable


class FakeAgentService:
    async def ask(self, question: str) -> Answer:
        assert question == "What is an AMF?"
        return Answer(
            answer="The AMF manages access and mobility.",
            citations=[
                Citation(
                    chunk_id="amf",
                    spec_id="TS 23.501",
                    release="Rel-17",
                    clause_number="6.2.1",
                    title="AMF",
                    quoted_span="access and mobility",
                )
            ],
            confidence="high",
        )


class FakeStore:
    def count(self) -> int:
        return 42


def settings(project: str | None = "demo-project") -> Settings:
    return Settings(
        google_cloud_project=project,
        allowed_origins="http://localhost:3000",
    )


def test_query_validates_input_and_serializes_structured_answer() -> None:
    client = TestClient(
        create_app(
            agent_service=FakeAgentService(),
            store=FakeStore(),
            settings=settings(),
        )
    )

    response = client.post("/query", json={"question": "  What is an AMF?  "})

    assert response.status_code == 200
    assert response.json()["confidence"] == "high"
    assert response.json()["citations"][0]["clause_number"] == "6.2.1"
    assert client.post("/query", json={"question": "   "}).status_code == 422
    assert client.post("/query", json={"question": "x" * 2001}).status_code == 422


@pytest.mark.parametrize(
    "failure",
    [
        DependencyUnavailable("Vertex AI is unavailable"),
        GoogleAuthError("ADC refresh failed"),
    ],
)
def test_query_returns_503_for_dependency_failure(failure: Exception) -> None:
    class FailingService:
        async def ask(self, question: str) -> Answer:
            raise failure

    client = TestClient(
        create_app(
            agent_service=FailingService(),
            store=FakeStore(),
            settings=settings(),
        )
    )

    response = client.post("/query", json={"question": "What is an AMF?"})

    assert response.status_code == 503
    assert response.json() == {"detail": "A required dependency is unavailable."}


def test_health_reports_index_count_and_missing_vertex_configuration() -> None:
    configured = TestClient(
        create_app(
            agent_service=FakeAgentService(),
            store=FakeStore(),
            settings=settings(),
        )
    )
    missing = TestClient(
        create_app(
            agent_service=FakeAgentService(),
            store=FakeStore(),
            settings=settings(project=None),
        )
    )

    assert configured.get("/health").json() == {
        "status": "ok",
        "opensearch": "ok",
        "vertex": "configured",
        "index_document_count": 42,
    }
    assert missing.get("/health").json()["status"] == "degraded"
