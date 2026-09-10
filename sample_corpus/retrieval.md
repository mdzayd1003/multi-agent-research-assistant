# Retrieval in this project

The retrieval pipeline has four stages and none of them are hidden behind a
framework abstraction.

## Splitting

Documents are split by `RecursiveTextSplitter`, which walks a list of separators
from coarse to fine: blank lines first, then single newlines, then sentence
punctuation, then words, and finally a hard character cut. Adjacent pieces are
packed greedily up to `chunk_size` and then re-overlapped by `chunk_overlap`
characters so a sentence that straddles a boundary is still retrievable from
both sides.

## Embedding

The default embedder is a hashing embedder. Word unigrams and character
4-grams are hashed into a fixed number of buckets with a sign drawn from a
second hash, counts are scaled sublinearly with log1p, and the vector is L2
normalised. It needs no download, which is what makes the offline mode and the
test suite possible. `SentenceTransformerEmbedder` swaps in MiniLM when the
optional extra is installed.

## Indexing

Vectors go into a store. The numpy store keeps a single float32 matrix and does
an exact cosine search; because every vector is unit norm, cosine similarity is
just a matrix product. FAISS `IndexFlatIP` is used instead when it is installed
and the corpus is at least 2000 chunks, which is roughly where the exact search
stops being free.

## Top-k search

Top-k search scores the query against every stored vector and returns the k
highest. The implementation uses `argpartition` to find the k best without
sorting the whole corpus, then sorts only those k. The retrieved chunks are
rendered into the prompt as `[<source>#<chunk>] text`, and that citation tag is what
agents are asked to carry through into their answers.
