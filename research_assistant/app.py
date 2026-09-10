"""Streamlit UI. Run it with ``python -m research_assistant serve``."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from research_assistant.config import Config
from research_assistant.embeddings import build_embedder
from research_assistant.ingest import Retriever, build_index
from research_assistant.llm import build_llm
from research_assistant.orchestrator import Orchestrator
from research_assistant.trace import Tracer
from research_assistant.vectorstore import VectorStore


def load_retriever(config: Config) -> Retriever | None:
    if not (config.index_dir / "chunks.json").exists():
        return None
    store = VectorStore.load(config.index_dir)
    return Retriever(store, build_embedder(config.embedder, dim=store.dim), default_k=config.top_k)


def main() -> None:
    st.set_page_config(page_title="Multi-agent research assistant", layout="wide")
    st.title("Multi-agent research assistant")

    config = Config()
    with st.sidebar:
        st.header("Setup")
        corpus = st.text_input("Corpus folder", "sample_corpus")
        backend = st.selectbox("LLM backend", ["fake", "ollama"])
        model = st.text_input("Ollama model", config.model)
        config.top_k = st.slider("Retrieved chunks", 1, 10, config.top_k)
        offline = st.checkbox("Offline (no web search)", value=True)
        if st.button("Re-ingest corpus"):
            store = build_index(corpus, embedder=build_embedder(config.embedder))
            store.save(config.index_dir)
            st.success(f"indexed {len(store)} chunks")

    retriever = load_retriever(config)
    if retriever is None:
        st.warning("No index yet — set a corpus folder in the sidebar and press *Re-ingest corpus*.")
        return

    question = st.text_input("Question", "How does top-k retrieval work in this project?")
    if not st.button("Ask"):
        return

    tracer = Tracer(Path(config.trace_dir) / "ui.jsonl")
    llm = build_llm(backend, model=model, host=config.ollama_host)
    report = Orchestrator(llm, retriever, tracer=tracer, offline=offline, context_k=config.top_k).run(question)

    st.subheader("Answer")
    st.write(report.answer or "_no answer_")
    if report.citations:
        st.caption("Sources: " + ", ".join(report.citations))

    left, right = st.columns(2)
    with left:
        st.subheader("Plan")
        st.json(report.plan)
        st.subheader("Critique")
        st.json(report.critique)
    with right:
        st.subheader("Subtask results")
        st.json(report.subtask_results)

    with st.expander(f"Trace ({len(tracer.events)} events)"):
        st.json([event.to_json() for event in tracer.events])


if __name__ == "__main__":  # pragma: no cover
    main()
