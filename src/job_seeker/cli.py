"""Command-line interface for the job_seeker application."""

import argparse
import json

__all__ = ["build_parser", "main"]


def cmd_crawl(_: argparse.Namespace) -> None:
    from job_seeker.crawler import crawl_jobs

    crawl_jobs()


def cmd_extract_resume(args: argparse.Namespace) -> None:
    from job_seeker.resume import process_resume

    extracted = process_resume(pdf_path=args.pdf)
    print(json.dumps(extracted, ensure_ascii=False, indent=2))


def cmd_init_vector_db(args: argparse.Namespace) -> None:
    from job_seeker.vector_db.qdrant import ensure_collection

    collection = ensure_collection(collection=args.collection, recreate=args.recreate)
    print(f"Collection ready: {collection}")


def cmd_ingest(args: argparse.Namespace) -> None:
    from job_seeker.vector_db.ingest import ingest_jobs

    count = ingest_jobs(path=args.path, recreate=args.recreate, collection=args.collection)
    print(f"Ingested {count} points")


def cmd_search(args: argparse.Namespace) -> None:
    from job_seeker.vector_db.search import search_jobs, search_jobs_expanded

    if args.expand:
        results, expanded = search_jobs_expanded(
            args.query, top_k=args.top_k, collection=args.collection
        )
        print(f"# Expanded BM25 query: {expanded}")
    else:
        results = search_jobs(args.query, top_k=args.top_k, collection=args.collection)
    print(json.dumps(results, ensure_ascii=False, indent=2))


def cmd_rag(args: argparse.Namespace) -> None:
    from job_seeker.rag import rag_search

    result = rag_search(
        args.query,
        top_k=args.top_k,
        rerank=not args.no_rerank,
        resume_path=args.resume,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="job-seeker", description="Job Seeker pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("crawl", help="Crawl JobsDB for analyst-programmer jobs")

    extract_parser = subparsers.add_parser("extract-resume", help="Convert resume PDF to markdown and extract structured JSON via LLM")
    extract_parser.add_argument("--pdf", default=None, help="Path to the resume PDF (overrides RESUME_PDF_PATH)")

    vector_parser = subparsers.add_parser("init-vector-db", help="Create the hybrid Qdrant collection (HNSW dense + BM25 sparse)")
    vector_parser.add_argument("--collection", default=None, help="Collection name (overrides QDRANT_COLLECTION)")
    vector_parser.add_argument("--recreate", action="store_true", help="Drop and recreate the collection")

    ingest_parser = subparsers.add_parser("ingest", help="Chunk, embed, and upsert crawled jobs into Qdrant")
    ingest_parser.add_argument("--path", default=None, help="Jobs JSON file (defaults to JOBSDB_OUTPUT_FILE)")
    ingest_parser.add_argument("--collection", default=None, help="Collection name (overrides QDRANT_COLLECTION)")
    ingest_parser.add_argument("--recreate", action="store_true", help="Recreate the collection before ingesting")

    search_parser = subparsers.add_parser("search", help="Hybrid RRF search over jobs (dense + BM25)")
    search_parser.add_argument("query", help="Natural language / keyword query")
    search_parser.add_argument("--top-k", type=int, default=None, help="Number of jobs to return (overrides SEARCH_LIMIT)")
    search_parser.add_argument("--collection", default=None, help="Collection name (overrides QDRANT_COLLECTION)")
    search_parser.add_argument("--expand", action="store_true", help="Expand the query keywords with the LLM for the BM25 branch")

    rag_parser = subparsers.add_parser("rag", help="Retrieve jobs with hybrid search and rerank them with the LLM against your resume")
    rag_parser.add_argument("query", help="Natural language / keyword query")
    rag_parser.add_argument("--top-k", type=int, default=10, help="Number of jobs to retrieve")
    rag_parser.add_argument("--resume", default=None, help="Path to the extracted resume JSON (defaults to RESUME_JSON_OUTPUT)")
    rag_parser.add_argument("--no-rerank", action="store_true", help="Skip LLM reranking and return hybrid results only")

    return parser


HANDLERS = {
    "crawl": cmd_crawl,
    "extract-resume": cmd_extract_resume,
    "init-vector-db": cmd_init_vector_db,
    "ingest": cmd_ingest,
    "search": cmd_search,
    "rag": cmd_rag,
}


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        HANDLERS[args.command](args)
    finally:
        from job_seeker.vector_db.qdrant import close_client

        close_client()


if __name__ == "__main__":
    main()
