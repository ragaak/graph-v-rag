"""CLI entry point: python -m graph_vlm_rag <command>"""

import argparse
import sys

from .config import get_settings


def main():
    """Parse CLI arguments and dispatch commands."""
    parser = argparse.ArgumentParser(
        prog="graph_vlm_rag",
        description="Multi-Modal Hybrid GraphRAG Pipeline",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a PDF file")
    ingest_parser.add_argument(
        "pdf_path",
        help="Path to PDF file to ingest",
    )
    ingest_parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear databases before ingesting (default: append)",
    )

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the knowledge graph")
    query_parser.add_argument(
        "question",
        help="Question to ask",
    )
    query_parser.add_argument(
        "--max-results",
        type=int,
        default=5,
        help="Maximum number of results to retrieve (default: 5)",
    )
    query_parser.add_argument(
        "--document",
        type=str,
        default=None,
        help="Filter by specific document name (default: all documents)",
    )

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Run evaluation")
    eval_parser.add_argument(
        "--questions-file",
        default="data/eval_questions.json",
        help="Path to evaluation questions (default: data/eval_questions.json)",
    )

    args = parser.parse_args()

    if args.version:
        from . import __version__
        print(f"graph_vlm_rag {__version__}")
        return 0

    if not args.command:
        parser.print_help()
        return 1

    # Dispatch commands
    if args.command == "ingest":
        from .ingest import ingest_pdf
        try:
            result = ingest_pdf(args.pdf_path, clear=args.clear)
            print(f"✅ Ingested: {result}")
            return 0
        except Exception as e:
            print(f"❌ Failed to ingest: {e}")
            return 1

    elif args.command == "query":
        from .query import answer_query
        try:
            result = answer_query(
                args.question,
                max_results=args.max_results,
                document_name=args.document,
            )
            print(f"🤖 {result}")
            return 0
        except Exception as e:
            print(f"❌ Failed to query: {e}")
            return 1

    elif args.command == "eval":
        from .eval import run_evaluation
        try:
            report = run_evaluation(args.questions_file)
            print(report)
            return 0
        except Exception as e:
            print(f"❌ Failed to run evaluation: {e}")
            return 1

    return 1


if __name__ == "__main__":
    sys.exit(main())