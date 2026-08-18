"""Load PDF, CSV, and TXT files into LangChain Documents (from the original notebook)."""

from pathlib import Path

import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

SUPPORTED_EXTENSIONS = {".pdf", ".csv", ".txt"}


def load_pdf(file_path: Path) -> list[Document]:
    loader = PyPDFLoader(str(file_path))
    return loader.load()


def load_csv_file(file_path: Path) -> list[Document]:
    df = pd.read_csv(file_path)
    documents: list[Document] = []
    for _, row in df.iterrows():
        content = " ".join(map(str, row.values))
        documents.append(
            Document(page_content=content, metadata={"source": str(file_path)})
        )
    return documents


def load_text_file(file_path: Path) -> list[Document]:
    content = file_path.read_text(encoding="utf-8")
    return [Document(page_content=content, metadata={"source": str(file_path)})]


def load_file(file_path: Path) -> list[Document]:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        print(f"Loading .pdf: {file_path}")
        return load_pdf(file_path)
    if suffix == ".csv":
        print(f"Loading .csv: {file_path}")
        return load_csv_file(file_path)
    if suffix == ".txt":
        print(f"Loading .txt: {file_path}")
        return load_text_file(file_path)
    return []


def load_directory(data_dir: Path) -> list[Document]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    documents: list[Document] = []
    for path in sorted(data_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            documents.extend(load_file(path))
    return documents
