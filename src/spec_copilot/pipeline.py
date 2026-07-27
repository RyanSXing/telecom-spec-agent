from pathlib import Path
from time import perf_counter
from typing import Any

from spec_copilot.config import Settings
from spec_copilot.ingest import (
    SPEC_URL,
    ClauseNodeParser,
    fetch_pinned_spec,
    parse_docx,
)
from spec_copilot.runtime import build_embedder, build_retriever, build_store


def ingest_path(
    path: Path,
    source_url: str,
    embedder: Any,
    store: Any,
) -> dict[str, int | float]:
    started = perf_counter()
    clauses = parse_docx(path, "TS 23.501", "Rel-17", source_url)
    chunks = ClauseNodeParser(chunk_size=800).chunk_clauses(clauses)
    vectors = embedder.embed_documents([chunk.text for chunk in chunks])
    store.ensure_index()
    store.index_chunks(chunks, vectors)
    return {
        "documents": 1,
        "clauses": len(clauses),
        "chunks": len(chunks),
        "embedded": len(vectors),
        "elapsed_seconds": round(perf_counter() - started, 3),
    }


def run_ingest(settings: Settings) -> dict[str, int | float]:
    path = fetch_pinned_spec()
    return ingest_path(
        path,
        SPEC_URL,
        build_embedder(settings),
        build_store(settings),
    )


def run_evaluation(settings: Settings) -> dict[str, Any]:
    from spec_copilot.evaluation import evaluate, load_cases, write_report

    retriever, _ = build_retriever(settings)
    report = evaluate(retriever, load_cases(Path("eval/golden_set.jsonl")))
    write_report(report, Path("eval/reports/latest.json"))
    return report
