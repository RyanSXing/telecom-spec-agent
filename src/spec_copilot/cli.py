import argparse
import json
import sys
from collections.abc import Sequence

from spec_copilot.config import Settings
from spec_copilot.pipeline import run_evaluation, run_ingest
from spec_copilot.runtime import DependencyUnavailable


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="spec-copilot",
        description="Ingest and evaluate public 3GPP specifications.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "ingest",
        help="Download, parse, embed, and index the pinned TS 23.501 document.",
    )
    subparsers.add_parser(
        "eval",
        help="Evaluate retrieval against the ten-question golden set.",
    )
    arguments = parser.parse_args(argv)

    try:
        report = (
            run_ingest(Settings()) if arguments.command == "ingest" else run_evaluation(Settings())
        )
    except DependencyUnavailable as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
