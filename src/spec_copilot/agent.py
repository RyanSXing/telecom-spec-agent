import json
import re
from typing import Any
from uuid import uuid4

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.integrations.langchain import LangchainTool
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, ValidationError

from spec_copilot.models import Answer, SearchHit


class HybridSearchInput(BaseModel):
    question: str = Field(description="A conceptual question about the specification.")


class ClauseLookupInput(BaseModel):
    clause_number: str = Field(description="An exact clause number such as 4.2.6.")


def build_langchain_tools(retriever: Any) -> tuple[StructuredTool, StructuredTool]:
    def hybrid_search(question: str) -> list[dict[str, Any]]:
        """Search specification clauses semantically and by keyword."""
        return [hit.model_dump() for hit in retriever.search(question)]

    def clause_lookup(clause_number: str) -> list[dict[str, Any]]:
        """Fetch every chunk for one exact specification clause number."""
        return [hit.model_dump() for hit in retriever.clause_lookup(clause_number)]

    return (
        StructuredTool.from_function(
            hybrid_search,
            name="hybrid_search",
            description=(
                "Use for conceptual questions. Runs BM25 and vector search over TS 23.501."
            ),
            args_schema=HybridSearchInput,
        ),
        StructuredTool.from_function(
            clause_lookup,
            name="clause_lookup",
            description=("Use only when the user names an exact clause number such as 4.2.6."),
            args_schema=ClauseLookupInput,
        ),
    )


def capture_tool_result(
    tool: Any,
    args: dict[str, Any],
    context: Any,
    tool_response: dict[str, Any],
) -> None:
    del tool, args
    result = tool_response.get("result", tool_response)
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except json.JSONDecodeError:
            result = []
    stored = json.loads(context.state.get("retrieval_context", "[]"))
    if isinstance(result, list):
        stored.extend(item for item in result if isinstance(item, dict))
    context.state["retrieval_context"] = json.dumps(stored)
    return None


def build_agent(retriever: Any, *, model: str) -> SequentialAgent:
    hybrid_search, clause_lookup = build_langchain_tools(retriever)
    researcher = LlmAgent(
        name="spec_researcher",
        model=model,
        description="Retrieves evidence from TS 23.501.",
        instruction=(
            "Always use a tool before responding. Use clause_lookup only when the user "
            "provides an exact clause number; otherwise use hybrid_search. Do not answer "
            "from memory. Briefly state which evidence was retrieved."
        ),
        tools=[LangchainTool(hybrid_search), LangchainTool(clause_lookup)],
        after_tool_callback=capture_tool_result,
    )
    writer = LlmAgent(
        name="grounded_answer_writer",
        model=model,
        description="Writes a structured answer using retrieved evidence only.",
        instruction=(
            "Answer the user's question using only the JSON evidence below. Cite the exact "
            "chunk_id for every citation and copy quoted_span verbatim from that chunk. "
            "If evidence is insufficient, use low confidence and name the missing aspect.\n\n"
            "Evidence:\n{retrieval_context}"
        ),
        tools=[],
        output_schema=Answer,
    )
    return SequentialAgent(
        name="spec_copilot",
        description="Retrieves specification evidence and writes a grounded answer.",
        sub_agents=[researcher, writer],
    )


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def validate_answer(answer: Answer, hits: list[SearchHit]) -> None:
    if not hits:
        raise ValueError("no retrieved evidence")
    if not answer.citations:
        raise ValueError("grounded answers require at least one citation")

    evidence = {hit.chunk_id: hit for hit in hits}
    for citation in answer.citations:
        hit = evidence.get(citation.chunk_id)
        if hit is None:
            raise ValueError(f"citation references unknown chunk: {citation.chunk_id}")
        if (
            citation.spec_id != hit.spec_id
            or citation.release != hit.release
            or citation.clause_number != hit.clause_number
        ):
            raise ValueError("citation metadata does not match the retrieved chunk")
        quoted_span = _normalized(citation.quoted_span)
        if not quoted_span or quoted_span not in _normalized(hit.text):
            raise ValueError("citation quoted span is not present in the retrieved chunk")


def grounded_refusal() -> Answer:
    return Answer(
        answer="I could not produce a fully grounded answer from the retrieved specification.",
        citations=[],
        confidence="low",
        unanswered_aspects=["No fully grounded answer could be generated."],
    )


class AgentService:
    def __init__(self, pipeline: Any, *, app_name: str = "spec_copilot") -> None:
        self.pipeline = pipeline
        self.app_name = app_name
        # ponytail: in-memory sessions are enough for the local single-process demo;
        # replace with a persistent ADK session service before multi-instance deployment.
        self.sessions = InMemorySessionService()

    async def _run_once(self, question: str) -> tuple[Answer, list[SearchHit]]:
        session_id = uuid4().hex
        user_id = "local-user"
        await self.sessions.create_session(
            app_name=self.app_name,
            user_id=user_id,
            session_id=session_id,
            state={"retrieval_context": "[]"},
        )
        runner = Runner(
            agent=self.pipeline,
            app_name=self.app_name,
            session_service=self.sessions,
        )
        final_text = ""
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=types.Content(
                role="user",
                parts=[types.Part(text=question)],
            ),
        ):
            if event.is_final_response() and event.content and event.content.parts:
                final_text = event.content.parts[0].text or ""

        session = await self.sessions.get_session(
            app_name=self.app_name,
            user_id=user_id,
            session_id=session_id,
        )
        if session is None or not final_text:
            raise ValueError("agent returned no final response")
        hits = [
            SearchHit.model_validate(item)
            for item in json.loads(session.state.get("retrieval_context", "[]"))
        ]
        return Answer.model_validate_json(final_text), hits

    async def ask(self, question: str) -> Answer:
        for _ in range(2):
            try:
                answer, hits = await self._run_once(question)
                validate_answer(answer, hits)
                return answer
            except (ValidationError, ValueError):
                pass
        return grounded_refusal()
