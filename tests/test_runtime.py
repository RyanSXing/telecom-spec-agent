import os

from spec_copilot import runtime
from spec_copilot.config import Settings


def test_build_runtime_configures_adk_for_vertex(monkeypatch) -> None:
    settings = Settings(
        google_cloud_project="test-project",
        google_cloud_location="northamerica-northeast1",
    )
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
    monkeypatch.setattr(runtime, "build_retriever", lambda _: (object(), object()))

    captured = {}

    def fake_build_agent(retriever, *, model):
        del retriever, model
        captured.update(
            {
                "use_vertex": os.environ.get("GOOGLE_GENAI_USE_VERTEXAI"),
                "project": os.environ.get("GOOGLE_CLOUD_PROJECT"),
                "location": os.environ.get("GOOGLE_CLOUD_LOCATION"),
            }
        )
        return object()

    monkeypatch.setattr(runtime, "build_agent", fake_build_agent)

    runtime.build_runtime(settings)

    assert captured == {
        "use_vertex": "true",
        "project": "test-project",
        "location": "northamerica-northeast1",
    }
