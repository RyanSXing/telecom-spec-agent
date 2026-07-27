import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class EvaluationCase(BaseModel):
    question: str
    relevant_clauses: list[str] = Field(min_length=1)


def load_cases(path: Path) -> list[EvaluationCase]:
    return [
        EvaluationCase.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def evaluate(retriever: Any, cases: Sequence[EvaluationCase]) -> dict[str, Any]:
    case_reports = []
    recalls = []
    reciprocal_ranks = []

    for case in cases:
        hits = retriever.search(case.question)
        retrieved = [hit.clause_number for hit in hits[:5]]
        relevant = set(case.relevant_clauses)
        recall = len(relevant.intersection(retrieved)) / len(relevant)
        first_relevant_rank = next(
            (rank for rank, clause in enumerate(retrieved, start=1) if clause in relevant),
            None,
        )
        recalls.append(recall)
        reciprocal_ranks.append(1 / first_relevant_rank if first_relevant_rank else 0.0)
        case_reports.append(
            {
                "question": case.question,
                "relevant_clauses": case.relevant_clauses,
                "retrieved_clauses": retrieved,
                "recall_at_5": round(recall, 6),
                "first_relevant_rank": first_relevant_rank,
            }
        )

    count = len(cases)
    return {
        "question_count": count,
        "recall_at_5": round(sum(recalls) / count, 6) if count else 0.0,
        "mrr": round(sum(reciprocal_ranks) / count, 6) if count else 0.0,
        "cases": case_reports,
    }


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{json.dumps(report, indent=2)}\n")
