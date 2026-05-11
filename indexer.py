import chromadb
from chromadb.errors import NotFoundError
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, CHROMA_DB_PATH, COLLECTION_NAME

_model = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def get_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_DB_PATH)


def get_embeddings(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False)
    return embeddings.tolist()


def index_document(chunks: list[dict]) -> int:
    chroma_client = get_chroma_client()

    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except (ValueError, NotFoundError):
        pass
    collection = chroma_client.create_collection(name=COLLECTION_NAME)

    batch_size = 64
    total = 0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [chunk["text"] for chunk in batch]
        ids = [chunk["id"] for chunk in batch]
        metadatas = [chunk["metadata"] for chunk in batch]

        embeddings = get_embeddings(texts)

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        total += len(batch)
        print(f"  Проиндексировано: {total}/{len(chunks)} чанков")

    return total
