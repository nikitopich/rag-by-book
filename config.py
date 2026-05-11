import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "claude-haiku-4-5"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "book_chunks"
