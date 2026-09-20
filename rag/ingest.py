"""Chunk documents/*.md by section, embed each section, and load them into Chroma.

Run this once up front, and again whenever a document under documents/ changes:

    python ingest.py
"""
import re

import chromadb
import ollama

from config import CHROMA_DIR, COLLECTION_METADATA, COLLECTION_NAME, DOCUMENTS_DIR, EMBED_MODEL

SECTION_PATTERN = re.compile(r"^##\s+(.*)$", re.MULTILINE)
TITLE_PATTERN = re.compile(r"^#\s+(.*)$", re.MULTILINE)


def split_into_sections(markdown_text: str) -> list[tuple[str, str]]:
    """Return [(section_heading, section_body), ...] for a doc split on '## ' headers."""
    parts = SECTION_PATTERN.split(markdown_text)[1:]  # drop preamble before first '##'
    # re.split with a capturing group interleaves heading, body, heading, body, ...
    headings, bodies = parts[0::2], parts[1::2]
    return [(h.strip(), b.strip()) for h, b in zip(headings, bodies)]


def load_documents() -> list[dict]:
    """Return one record per section across every .md file in documents/."""
    records = []
    for path in sorted(DOCUMENTS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title_match = TITLE_PATTERN.search(text)
        title = title_match.group(1).strip() if title_match else path.stem

        for heading, body in split_into_sections(text):
            records.append(
                {
                    "id": f"{path.stem}::{heading}",
                    "text": f"{heading}\n\n{body}",
                    "title": title,
                    "source": path.name,
                    "section": f"{title}, {heading}",
                }
            )
    return records


def main():
    records = load_documents()
    if not records:
        raise SystemExit(f"No sections found under {DOCUMENTS_DIR} — nothing to ingest.")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # Rebuilt from scratch each run so renamed/deleted sections don't linger.
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(COLLECTION_NAME, metadata=COLLECTION_METADATA)

    ids, texts, metadatas, embeddings = [], [], [], []
    for record in records:
        embedding = ollama.embeddings(model=EMBED_MODEL, prompt=record["text"])["embedding"]
        ids.append(record["id"])
        texts.append(record["text"])
        metadatas.append({"title": record["title"], "source": record["source"], "section": record["section"]})
        embeddings.append(embedding)

    collection.upsert(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
    print(f"Ingested {len(records)} sections from {len(list(DOCUMENTS_DIR.glob('*.md')))} documents into '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    main()
