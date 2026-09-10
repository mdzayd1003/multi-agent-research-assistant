import pytest

from research_assistant.ingest import build_index, chunk_documents, iter_documents


def test_only_text_files_are_picked_up(tmp_path):
    (tmp_path / "a.md").write_text("hello", encoding="utf-8")
    (tmp_path / "b.png").write_bytes(b"\x89PNG")
    assert [p.name for p in iter_documents(tmp_path)] == ["a.md"]


def test_chunks_carry_a_relative_source_and_index(tmp_path):
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "doc.md").write_text("paragraph one.\n\nparagraph two.", encoding="utf-8")
    chunks = chunk_documents(iter_documents(tmp_path), root=tmp_path)
    assert chunks[0].source == "sub/doc.md"
    assert chunks[0].citation == "sub/doc.md#0"
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_building_an_index_covers_every_source_document(corpus_dir, embedder):
    store = build_index(corpus_dir, embedder=embedder, chunk_size=500, chunk_overlap=80, backend="numpy")
    assert len(store) > 3
    assert {c.source for c in store.chunks} == {"retrieval.md", "agents.md", "tracing.md"}


def test_an_empty_corpus_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_index(tmp_path)


def test_retriever_finds_the_document_that_covers_the_question(retriever):
    hits = retriever.retrieve("how are parse failures handled?", k=3)
    assert hits
    assert any(h.chunk.source == "agents.md" for h in hits)


def test_a_blank_query_retrieves_nothing(retriever):
    assert retriever.retrieve("   ") == []
