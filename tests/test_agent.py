import json
from types import MethodType, SimpleNamespace

import pytest
from google.adk.agents import LlmAgent, SequentialAgent

from spec_copilot.agent import (
    AgentService,
    build_agent,
    build_langchain_tools,
    capture_tool_result,
    validate_answer,
)
from spec_copilot.models import Answer, Citation, SearchHit


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


class FakeRetriever:
    def search(self, question: str):
        assert question == "What is an AMF?"
        return [hit("amf", "6.2.1")]

    def clause_lookup(self, clause_number: str):
        assert clause_number == "4.2.6"
        return [hit("sbi", "4.2.6")]


def test_langchain_tools_return_json_compatible_search_hits() -> None:
    hybrid_search, clause_lookup = build_langchain_tools(FakeRetriever())

    semantic = hybrid_search.invoke({"question": "What is an AMF?"})
    exact = clause_lookup.invoke({"clause_number": "4.2.6"})

    assert semantic[0]["chunk_id"] == "amf"
    assert exact[0]["chunk_id"] == "sbi"


def test_capture_tool_result_appends_raw_context_to_session_state() -> None:
    context = SimpleNamespace(state={"retrieval_context": "[]"})
    response = [hit("amf", "6.2.1").model_dump()]

    assert (
        capture_tool_result(
            tool=SimpleNamespace(),
            args={},
            tool_context=context,
            tool_response=response,
        )
        is None
    )

    stored = json.loads(context.state["retrieval_context"])
    assert stored[0]["chunk_id"] == "amf"


def test_build_agent_separates_tool_use_from_structured_output() -> None:
    pipeline = build_agent(FakeRetriever(), model="gemini-2.5-flash")

    assert isinstance(pipeline, SequentialAgent)
    assert [agent.name for agent in pipeline.sub_agents] == [
        "spec_researcher",
        "grounded_answer_writer",
    ]
    researcher, writer = pipeline.sub_agents
    assert isinstance(researcher, LlmAgent)
    assert len(researcher.tools) == 2
    assert isinstance(writer, LlmAgent)
    assert writer.tools == []
    assert writer.output_schema is Answer
    assert "exactly one complete sentence" in writer.instruction


def answer(quoted_span: str = "Text for 1") -> Answer:
    return Answer(
        answer="Grounded answer.",
        citations=[
            Citation(
                chunk_id="a",
                spec_id="TS 23.501",
                release="Rel-17",
                clause_number="1",
                title="Clause 1",
                quoted_span=quoted_span,
            )
        ],
        confidence="high",
    )


def test_validate_answer_requires_real_chunk_and_verbatim_span() -> None:
    validate_answer(answer(), [hit("a", "1")])

    with pytest.raises(ValueError, match="quoted span"):
        validate_answer(answer("invented"), [hit("a", "1")])

    with pytest.raises(ValueError, match="unknown chunk"):
        validate_answer(answer(), [hit("different", "1")])


@pytest.mark.asyncio
async def test_agent_service_retries_invalid_output_once() -> None:
    service = AgentService(SimpleNamespace())
    attempts = 0

    async def fake_run_once(self, question: str):
        nonlocal attempts
        attempts += 1
        response = answer("invented") if attempts == 1 else answer()
        return response, [hit("a", "1")]

    service._run_once = MethodType(fake_run_once, service)

    result = await service.ask("question")

    assert result == answer()
    assert attempts == 2


@pytest.mark.asyncio
async def test_agent_service_returns_safe_refusal_after_two_invalid_attempts() -> None:
    service = AgentService(SimpleNamespace())

    async def fake_run_once(self, question: str):
        return answer("invented"), [hit("a", "1")]

    service._run_once = MethodType(fake_run_once, service)

    result = await service.ask("question")

    assert result.confidence == "low"
    assert result.citations == []
    assert result.unanswered_aspects == ["No fully grounded answer could be generated."]
