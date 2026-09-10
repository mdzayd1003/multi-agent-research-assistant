import numpy as np

from research_assistant.embeddings import HashingEmbedder, l2_normalise, tokenize


def test_vectors_are_unit_norm(embedder):
    vectors = embedder.encode(["hello world", "another document about FAISS"])
    assert vectors.shape == (2, 256)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_encoding_is_deterministic(embedder):
    a = embedder.encode(["top-k search over vectors"])
    b = HashingEmbedder(dim=256).encode(["top-k search over vectors"])
    np.testing.assert_allclose(a, b)


def test_related_text_scores_above_unrelated_text(embedder):
    query = embedder.encode(["how does top-k vector search work"])[0]
    related = embedder.encode(["top-k search returns the k closest vectors"])[0]
    unrelated = embedder.encode(["cats are small domestic animals"])[0]
    assert float(query @ related) > float(query @ unrelated)


def test_empty_text_gives_a_finite_vector(embedder):
    vector = embedder.encode([""])
    assert np.isfinite(vector).all()


def test_l2_normalise_leaves_zero_rows_alone():
    out = l2_normalise(np.zeros((1, 4), dtype=np.float32))
    assert np.isfinite(out).all() and float(out.sum()) == 0.0


def test_tokenizer_lowercases_and_drops_punctuation():
    assert tokenize("Top-K, Search!") == ["top", "k", "search"]
