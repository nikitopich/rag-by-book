from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from config import COLLECTION_NAME, TOP_K
from indexer import get_chroma_client, get_embeddings

try:
    import pymorphy3
    _morph = pymorphy3.MorphAnalyzer()

    def _tokenize(text: str) -> list[str]:
        return [_morph.parse(w)[0].normal_form for w in text.lower().split()]
except ImportError:
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()

BM25_TOP_N = 15
VECTOR_TOP_N = 15
RRF_K = 60

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_CANDIDATES = 30

_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker

_bm25_cache: tuple[list[str], list[str], "BM25Okapi"] | None = None


def _get_bm25(collection) -> tuple[list[str], list[str], "BM25Okapi"]:
    global _bm25_cache
    corpus = collection.get(include=["documents"])
    all_ids: list[str] = corpus["ids"]
    all_docs: list[str] = corpus["documents"]
    if _bm25_cache is None or _bm25_cache[0] != all_ids:
        tokenized = [_tokenize(doc) for doc in all_docs]
        _bm25_cache = (all_ids, all_docs, BM25Okapi(tokenized))
    return _bm25_cache


def retrieve(query: str, top_k: int = TOP_K) -> dict:
    client = get_chroma_client()
    collection = client.get_collection(name=COLLECTION_NAME)

    query_embedding = get_embeddings([query], prefix="query: ")[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "distances"],
    )

    return {
        "documents": results["documents"][0],
        "distances": results["distances"][0],
        "ids": results["ids"][0],
    }


def hybrid_retrieve(query: str, top_k: int = TOP_K) -> dict:
    client = get_chroma_client()
    collection = client.get_collection(name=COLLECTION_NAME)

    all_ids, all_docs, bm25 = _get_bm25(collection)

    # BM25 search (with morphological normalization for Russian)
    bm25_scores = bm25.get_scores(_tokenize(query))

    bm25_top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:BM25_TOP_N]
    bm25_ranks = {all_ids[i]: rank for rank, i in enumerate(bm25_top_indices)}

    # Vector search
    query_embedding = get_embeddings([query], prefix="query: ")[0]
    vector_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=VECTOR_TOP_N,
        include=["documents", "distances"],
    )
    vector_ids = vector_results["ids"][0]
    vector_distances = vector_results["distances"][0]
    vector_ranks = {doc_id: rank for rank, doc_id in enumerate(vector_ids)}

    # RRF fusion: score = Σ 1 / (k + rank_i)
    # absent docs get corpus_size as rank → contribution ≈ 0
    corpus_size = len(all_ids)
    candidate_ids = set(bm25_ranks) | set(vector_ranks)

    def rrf_score(doc_id: str) -> float:
        b_rank = bm25_ranks.get(doc_id, corpus_size)
        v_rank = vector_ranks.get(doc_id, corpus_size)
        return 1 / (RRF_K + b_rank) + 1 / (RRF_K + v_rank)

    ranked_ids = sorted(candidate_ids, key=rrf_score, reverse=True)[:top_k]

    id_to_doc = dict(zip(all_ids, all_docs))
    id_to_dist = dict(zip(vector_ids, vector_distances))

    return {
        "documents": [id_to_doc[doc_id] for doc_id in ranked_ids],
        "distances": [id_to_dist.get(doc_id, -1.0) for doc_id in ranked_ids],
        "ids": ranked_ids,
    }


def hybrid_retrieve_reranked(query: str, top_k: int = TOP_K) -> dict:
    client = get_chroma_client()
    collection = client.get_collection(name=COLLECTION_NAME)

    all_ids, all_docs, bm25 = _get_bm25(collection)

    bm25_scores = bm25.get_scores(_tokenize(query))
    bm25_top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:RERANKER_CANDIDATES]
    bm25_ranks = {all_ids[i]: rank for rank, i in enumerate(bm25_top_indices)}

    query_embedding = get_embeddings([query], prefix="query: ")[0]
    vector_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=RERANKER_CANDIDATES,
        include=["documents", "distances"],
    )
    vector_ids = vector_results["ids"][0]
    vector_distances = vector_results["distances"][0]
    vector_ranks = {doc_id: rank for rank, doc_id in enumerate(vector_ids)}

    corpus_size = len(all_ids)
    candidate_ids = set(bm25_ranks) | set(vector_ranks)

    def rrf_score(doc_id: str) -> float:
        b_rank = bm25_ranks.get(doc_id, corpus_size)
        v_rank = vector_ranks.get(doc_id, corpus_size)
        return 1 / (RRF_K + b_rank) + 1 / (RRF_K + v_rank)

    # Take top RERANKER_CANDIDATES by RRF, then rerank with cross-encoder
    rrf_top_ids = sorted(candidate_ids, key=rrf_score, reverse=True)[:RERANKER_CANDIDATES]

    id_to_doc = dict(zip(all_ids, all_docs))
    id_to_dist = dict(zip(vector_ids, vector_distances))

    candidate_docs = [id_to_doc[doc_id] for doc_id in rrf_top_ids]
    reranker = get_reranker()
    ce_scores = reranker.predict([(query, doc) for doc in candidate_docs])

    ranked_pairs = sorted(zip(rrf_top_ids, candidate_docs, ce_scores), key=lambda x: x[2], reverse=True)[:top_k]

    return {
        "documents": [doc for _, doc, _ in ranked_pairs],
        "distances": [id_to_dist.get(doc_id, -1.0) for doc_id, _, _ in ranked_pairs],
        "ids": [doc_id for doc_id, _, _ in ranked_pairs],
        "reranker_scores": [float(score) for _, _, score in ranked_pairs],
    }
