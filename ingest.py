"""Read a document, embed its chunks with Ollama, and store them in ChromaDB."""

from pathlib import Path
from typing import Any

import chromadb
import ollama


BASE_DIR = Path(__file__).parent
DOCUMENT_PATH = BASE_DIR / "documents" / "hush_policy.txt"
CHROMA_PATH = BASE_DIR / "chroma_db"
COLLECTION_NAME = "hush_policy"
EMBEDDING_MODEL = "nomic-embed-text"


def read_document(path: Path) -> str:
    """Read a UTF-8 document and reject missing or empty input."""
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Document is empty: {path}")
    return text


def split_into_chunks(text: str) -> list[str]:
    """Use paragraphs as simple, meaningful chunks for this learning project."""
    chunks = [paragraph.strip() for paragraph in text.split("\n\n")]
    return [chunk for chunk in chunks if chunk]


def get_model_name(model: Any) -> str:
    """Get a model name from either an Ollama object or dict response."""
    if isinstance(model, dict):
        return str(model.get("name", model.get("model", "")))
    return str(getattr(model, "model", getattr(model, "name", "")))


def ensure_ollama_model(model: str) -> None:
    """Give a useful message when Ollama is stopped or the model is not pulled."""
    try:
        response = ollama.list()
    except Exception as error:
        raise RuntimeError("Ollama is not running. Start Ollama, then try again.") from error

    models = response.get("models", []) if isinstance(response, dict) else response.models
    available_names = {get_model_name(item).split(":")[0] for item in models}
    if model.split(":")[0] not in available_names:
        raise RuntimeError(f"Ollama model '{model}' is unavailable. Pull it with: ollama pull {model}")


def embed_chunks(chunks: list[str]) -> list[list[float]]:
    """Turn each chunk into a vector using the local Ollama embedding model."""
    try:
        response = ollama.embed(model=EMBEDDING_MODEL, input=chunks)
    except ollama.ResponseError as error:
        raise RuntimeError(f"Could not create embeddings with Ollama: {error}") from error
    return response["embeddings"] if isinstance(response, dict) else response.embeddings


def main() -> None:
    try:
        ensure_ollama_model(EMBEDDING_MODEL)
        document_text = read_document(DOCUMENT_PATH)
        chunks = split_into_chunks(document_text)
        embeddings = embed_chunks(chunks)

        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
        collection.upsert(
            ids=[f"hush-policy-{index}" for index in range(len(chunks))],
            documents=chunks,
            embeddings=embeddings,
            metadatas=[
                {
                    "source": str(DOCUMENT_PATH.relative_to(BASE_DIR)),
                    "document_name": DOCUMENT_PATH.name,
                    "chunk_index": index,
                }
                for index in range(len(chunks))
            ],
        )
        print(f"Stored {len(chunks)} chunks in ChromaDB at {CHROMA_PATH}.")
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"Ingestion error: {error}")


if __name__ == "__main__":
    main()
