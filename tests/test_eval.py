import json
from pathlib import Path

from spec_copilot.evaluation import EvaluationCase, evaluate, write_report
from spec_copilot.models import SearchHit


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


def test_evaluate_calculates_recall_at_five_and_mrr() -> None:
    class Retriever:
        def search(self, question: str, limit: int = 8) -> list[SearchHit]:
            return {
                "first": [hit("x", "9"), hit("a", "1")],
                "second": [hit("b", "2")],
            }[question]

    report = evaluate(
        Retriever(),
        [
            EvaluationCase(question="first", relevant_clauses=["1"]),
            EvaluationCase(question="second", relevant_clauses=["2", "3"]),
        ],
    )

    assert report["question_count"] == 2
    assert report["recall_at_5"] == 0.75
    assert report["mrr"] == 0.75
    assert report["cases"][0]["first_relevant_rank"] == 2


def test_write_report_creates_machine_readable_json(tmp_path: Path) -> None:
    output = tmp_path / "reports" / "latest.json"

    write_report({"question_count": 0, "recall_at_5": 0.0, "mrr": 0.0}, output)

    assert json.loads(output.read_text())["mrr"] == 0.0
