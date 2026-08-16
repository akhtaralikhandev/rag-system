"""Ask questions against the locally stored Hush policy chunks."""

from pathlib import Path

import chromadb
import ollama


BASE_DIR = Path(__file__).parent
CHROMA_PATH = BASE_DIR / "chroma_db"
COLLECTION_NAME = "hush_policy"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"
TOP_K = 3


def ensure_ollama_model(model: str) -> None:
    """Check that Ollama is reachable and the requested model is installed."""
    try:
        response = ollama.list()
    except Exception as error:
        raise RuntimeError("Ollama is not running. Start Ollama and try again.") from error

    models = response.get("models", []) if isinstance(response, dict) else response.models
    names = set()
    for item in models:
        name = item.get("name", item.get("model", "")) if isinstance(item, dict) else getattr(item, "model", "")
        names.add(str(name).split(":")[0])
    if model.split(":")[0] not in names:
        raise RuntimeError(f"Ollama model '{model}' is unavailable. Pull it with: ollama pull {model}")


def embed_question(question: str) -> list[float]:
    """Embed the question in the same vector space as the document chunks."""
    try:
        response = ollama.embed(model=EMBEDDING_MODEL, input=question)
    except ollama.ResponseError as error:
        raise RuntimeError(f"Could not embed the question: {error}") from error
    embeddings = response["embeddings"] if isinstance(response, dict) else response.embeddings
    return embeddings[0]


def retrieve_chunks(question_embedding: list[float]) -> list[dict[str, str]]:
    """Search ChromaDB for the chunks nearest to the question vector."""
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as error:
        raise RuntimeError("ChromaDB collection not found. Run 'python ingest.py' first.") from error

    result = collection.query(query_embeddings=[question_embedding], n_results=TOP_K)
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    return [{"text": document, "metadata": str(metadata)} for document, metadata in zip(documents, metadatas)]


def answer_question(question: str, retrieved_chunks: list[dict[str, str]]) -> str:
    """Ask the LLM to answer only from the retrieved context."""
    context = "\n\n".join(chunk["text"] for chunk in retrieved_chunks)
    prompt = f"""You answer questions using only the supplied context.
Do not invent facts or use outside knowledge.
If the answer is not contained in the context, say: "It cannot be determined from the provided documents."

Context:
{context}

Question: {question}
Answer:"""
    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "Be concise and grounded in the provided context."},
                {"role": "user", "content": prompt},
            ],
        )
    except ollama.ResponseError as error:
        raise RuntimeError(f"Could not generate an answer with Ollama: {error}") from error
    message = response["message"] if isinstance(response, dict) else response.message
    return message["content"] if isinstance(message, dict) else message.content


def main() -> None:
    question = input("Ask a question (or press Enter to quit): ").strip()
    if not question:
        print("Question cannot be empty.")
        return

    try:
        ensure_ollama_model(EMBEDDING_MODEL)
        ensure_ollama_model(LLM_MODEL)
        retrieved_chunks = retrieve_chunks(embed_question(question))
        print(f"\nQuestion: {question}\n\nRetrieved chunks:")
        for index, chunk in enumerate(retrieved_chunks, start=1):
            print(f"\n[{index}] {chunk['metadata']}\n{chunk['text']}")
        print(f"\nFinal answer:\n{answer_question(question, retrieved_chunks)}")
    except RuntimeError as error:
        print(f"RAG error: {error}")


if __name__ == "__main__":
    main()
