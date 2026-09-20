from pathlib import Path

BASE_DIR = Path(__file__).parent
DOCUMENTS_DIR = BASE_DIR / "documents"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "safety_docs"
# Cosine is the right metric for text embeddings; Chroma defaults to squared L2.
COLLECTION_METADATA = {"hnsw:space": "cosine"}

EMBED_MODEL = "nomic-embed-text"
# nomic-embed-text is trained with asymmetric task prefixes; documents and
# queries must use their matching one or retrieval quality drops.
DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "
LLM_MODEL = "llama3.2:3b"  # swap to "llama3.1:8b" once that pull succeeds / on stronger hardware

RETRIEVAL_TOP_K = 3
