"""Query the ingested safety knowledge base for the sections most relevant to a query."""
import chromadb
import ollama

from config import (
    CHROMA_DIR,
    COLLECTION_METADATA,
    COLLECTION_NAME,
    EMBED_MODEL,
    QUERY_PREFIX,
    RETRIEVAL_TOP_K,
)


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(COLLECTION_NAME, metadata=COLLECTION_METADATA)


def retrieve(query: str, k: int = RETRIEVAL_TOP_K) -> list[dict]:
    """Return the top-k most relevant sections as [{text, title, source, section, distance}, ...]."""
    collection = get_collection()
    if collection.count() == 0:
        raise RuntimeError("Knowledge base is empty — run `python ingest.py` first.")

    query_embedding = ollama.embeddings(model=EMBED_MODEL, prompt=QUERY_PREFIX + query)["embedding"]
    results = collection.query(query_embeddings=[query_embedding], n_results=k)

    hits = []
    for text, metadata, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        hits.append({"text": text, "distance": distance, **metadata})
    return hits


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "unauthorized entry into a restricted lab after hours"
    for hit in retrieve(query):
        print(f"[{hit['distance']:.3f}] {hit['section']}")
        print(hit["text"][:200].replace("\n", " ") + "...\n")
