"""Walk DATA_DIR, chunk documents, and rebuild the Chroma collection."""

from app.config import settings
from app.loaders import load_directory
from app.rag import chunk_documents, persist_documents


def main() -> None:
    print(f"Loading documents from {settings.data_dir}")
    documents = load_directory(settings.data_dir)
    if not documents:
        print(
            "No PDF, CSV, or TXT files found. Add files under data/ and run ingest again."
        )
        return

    chunks = chunk_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks")

    vectordb = persist_documents(chunks)
    count = vectordb._collection.count()
    print(f"Persisted {count} embeddings to {settings.chroma_persist_dir}")


if __name__ == "__main__":
    main()
