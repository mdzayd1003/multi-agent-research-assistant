import numpy as np
import pytest

from research_assistant.vectorstore import Chunk, VectorStore, build_store, format_context


def make_store(embedder, texts):
    store = VectorStore(embedder.dim)
    chunks = [Chunk(id=f"d#{i}", text=t, source="d.md", index=i) for i, t in enumerate(texts)]
    store.add(chunks, embedder.encode(texts))
    return store


def test_search_ranks_the_relevant_chunk_first(embedder):
    store = make_store(
        embedder,
        [
            "FAISS builds an index for fast nearest neighbour search",
            "cats are small domestic animals kept as pets",
            "top-k search returns the k closest vectors to a query",
        ],
    )
    hits = store.search(embedder.encode(["how does top-k vector search work"])[0], k=2)
    assert len(hits) == 2
    assert "top-k search" in hits[0].chunk.text
    assert hits[0].score >= hits[1].score


def test_search_on_an_empty_store_returns_nothing(embedder):
    assert VectorStore(embedder.dim).search(np.zeros(embedder.dim), k=3) == []


def test_k_is_clamped_to_the_corpus_size(embedder):
    store = make_store(embedder, ["one", "two"])
    assert len(store.search(embedder.encode(["one"])[0], k=99)) == 2


def test_dimension_mismatch_is_rejected(embedder):
    store = VectorStore(embedder.dim)
    with pytest.raises(ValueError):
        store.add([Chunk(id="a", text="a", source="s", index=0)], np.zeros((1, embedder.dim + 1), dtype=np.float32))


def test_chunk_count_must_match_vector_count(embedder):
    store = VectorStore(embedder.dim)
    with pytest.raises(ValueError):
        store.add([Chunk(id="a", text="a", source="s", index=0)], np.zeros((2, embedder.dim), dtype=np.float32))


def test_round_trip_through_disk_preserves_results(embedder, tmp_path):
    store = make_store(embedder, ["alpha document", "beta document", "gamma document"])
    store.save(tmp_path / "index")
    reloaded = VectorStore.load(tmp_path / "index")

    query = embedder.encode(["beta"])[0]
    assert len(reloaded) == len(store)
    assert [h.chunk.id for h in reloaded.search(query, k=3)] == [h.chunk.id for h in store.search(query, k=3)]


def test_auto_backend_stays_on_numpy_for_a_small_corpus(embedder):
    assert build_store(embedder.dim, "auto", corpus_size=10).backend == "numpy"


def test_unknown_backend_is_rejected(embedder):
    with pytest.raises(ValueError):
        build_store(embedder.dim, "annoy")


def test_format_context_emits_tags_and_respects_the_budget(embedder):
    store = make_store(embedder, ["a" * 400, "b" * 400, "c" * 400])
    hits = store.search(embedder.encode(["a"])[0], k=3)
    context, citations = format_context(hits, max_chars=500)
    assert citations and all(c.startswith("d.md#") for c in citations)
    assert f"[{citations[0]}]" in context
    assert len(citations) < 3
