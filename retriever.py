from config import COLLECTION_NAME, TOP_K
from indexer import get_chroma_client, get_embeddings


def retrieve(query: str, top_k: int = TOP_K) -> dict:
    client = get_chroma_client()
    collection = client.get_collection(name=COLLECTION_NAME)

    query_embedding = get_embeddings([query])[0]

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
