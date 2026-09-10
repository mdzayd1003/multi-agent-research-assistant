"""CLI: ``ingest`` builds the index, ``ask`` runs the agents, ``serve`` opens the UI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from .config import Config
from .embeddings import build_embedder
from .ingest import Retriever, build_index
from .llm import build_llm
from .orchestrator import Orchestrator
from .trace import Tracer
from .vectorstore import VectorStore, format_context


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="research_assistant", description=__doc__)
    parser.add_argument("--index-dir", default=None, help="where the vector index lives")
    parser.add_argument("--embedder", default=None, choices=["hashing", "minilm"])
    parser.add_argument("--offline", action="store_true", help="disable every network tool")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="chunk and embed a corpus")
    ingest.add_argument("corpus", help="file or directory to ingest")
    ingest.add_argument("--chunk-size", type=int, default=None)
    ingest.add_argument("--chunk-overlap", type=int, default=None)
    ingest.add_argument("--store", default=None, choices=["auto", "numpy", "faiss"])

    ask = sub.add_parser("ask", help="run the multi-agent pipeline on a question")
    ask.add_argument("question")
    ask.add_argument("--fake-llm", action="store_true", help="deterministic offline backend")
    ask.add_argument("--model", default=None)
    ask.add_argument("--top-k", type=int, default=None)
    ask.add_argument("--max-steps", type=int, default=None)
    ask.add_argument("--trace", default=None, help="path for the JSONL trace")
    ask.add_argument("--json", action="store_true", help="print the full report as JSON")

    retrieve = sub.add_parser("retrieve", help="inspect what retrieval returns, no LLM involved")
    retrieve.add_argument("query")
    retrieve.add_argument("--top-k", type=int, default=None)

    sub.add_parser("serve", help="launch the Streamlit UI")
    return parser


def _load_retriever(config: Config) -> Retriever:
    if not (config.index_dir / "chunks.json").exists():
        raise SystemExit(f"no index at {config.index_dir}; run `ingest` first")
    store = VectorStore.load(config.index_dir)
    return Retriever(store, build_embedder(config.embedder, dim=store.dim), default_k=config.top_k)


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = Config()
    if args.index_dir:
        config.index_dir = Path(args.index_dir)
    if args.embedder:
        config.embedder = args.embedder
    if args.offline:
        config.offline = True

    if args.command == "ingest":
        embedder = build_embedder(config.embedder)
        store = build_index(
            args.corpus,
            embedder=embedder,
            chunk_size=args.chunk_size or config.chunk_size,
            chunk_overlap=args.chunk_overlap or config.chunk_overlap,
            backend=args.store or config.store_backend,
        )
        store.save(config.index_dir)
        print(f"indexed {len(store)} chunks from {args.corpus} into {config.index_dir} ({store.backend})")
        return 0

    if args.command == "retrieve":
        retriever = _load_retriever(config)
        hits = retriever.retrieve(args.query, k=args.top_k or config.top_k)
        if not hits:
            print("no hits")
            return 0
        context, citations = format_context(hits)
        print(context)
        print("\ncitations:", ", ".join(citations))
        return 0

    if args.command == "ask":
        if args.top_k:
            config.top_k = args.top_k
        if args.max_steps:
            config.max_steps = args.max_steps
        retriever = _load_retriever(config)
        llm = (
            build_llm("fake")
            if args.fake_llm
            else build_llm(config.llm_backend, model=args.model or config.model, host=config.ollama_host)
        )
        trace_path = Path(args.trace) if args.trace else config.trace_dir / "run.jsonl"
        tracer = Tracer(trace_path)
        orchestrator = Orchestrator(
            llm, retriever, tracer=tracer, offline=config.offline, max_steps=config.max_steps, context_k=config.top_k
        )
        report = orchestrator.run(args.question)
        if args.json:
            print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
        else:
            print(report.answer)
            if report.citations:
                print("\nsources: " + ", ".join(report.citations))
            print(f"\ntrace: {trace_path} ({sum(tracer.counts().values())} events)")
        return 0 if report.answer else 1

    if args.command == "serve":  # pragma: no cover - launches a server
        from streamlit.web import cli as stcli

        sys.argv = ["streamlit", "run", str(Path(__file__).with_name("app.py"))]
        return int(stcli.main() or 0)

    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
