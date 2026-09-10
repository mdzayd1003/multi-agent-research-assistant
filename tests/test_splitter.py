import pytest

from research_assistant.splitter import RecursiveTextSplitter


def test_chunks_respect_the_size_budget():
    splitter = RecursiveTextSplitter(chunk_size=100, chunk_overlap=0)
    text = "\n\n".join("sentence number %d is here. " % i * 3 for i in range(20))
    chunks = splitter.split(text)
    assert chunks
    assert all(len(c) <= 100 for c in chunks)


def test_overlap_carries_the_tail_of_the_previous_chunk():
    splitter = RecursiveTextSplitter(chunk_size=60, chunk_overlap=15)
    chunks = splitter.split("alpha beta gamma. " * 12)
    assert len(chunks) > 1
    for previous, chunk in zip(chunks, chunks[1:]):
        assert chunk.startswith(previous[-15:].strip()[:5])


def test_prefers_paragraph_boundaries_over_hard_cuts():
    splitter = RecursiveTextSplitter(chunk_size=40, chunk_overlap=0)
    chunks = splitter.split("first para here.\n\nsecond para here.\n\nthird para here.")
    assert len(chunks) >= 2
    assert any(c.startswith("first para") for c in chunks)


def test_text_shorter_than_a_chunk_is_returned_whole():
    assert RecursiveTextSplitter(chunk_size=500).split("short text") == ["short text"]


def test_empty_input_produces_no_chunks():
    assert RecursiveTextSplitter().split("   ") == []


def test_no_content_is_dropped_when_there_is_no_overlap():
    splitter = RecursiveTextSplitter(chunk_size=50, chunk_overlap=0)
    text = "word " * 60
    joined = "".join(splitter.split(text)).replace(" ", "")
    assert joined == text.replace(" ", "")


@pytest.mark.parametrize("size,overlap", [(0, 0), (10, 10), (10, 20)])
def test_invalid_settings_are_rejected(size, overlap):
    with pytest.raises(ValueError):
        RecursiveTextSplitter(chunk_size=size, chunk_overlap=overlap)
