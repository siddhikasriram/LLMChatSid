"""Embeddings, Chroma persistence, and the RetrievalQA chain from final.py."""

import shutil
from pathlib import Path

from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings


def build_embeddings() -> HuggingFaceInstructEmbeddings:
    return HuggingFaceInstructEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": settings.embedding_device},
    )


def chunk_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", "(?<=\\. )", " ", ""],
    )
    return text_splitter.split_documents(documents)


def _reset_persist_dir(persist_dir: Path) -> None:
    persist_dir.mkdir(parents=True, exist_ok=True)
    for child in persist_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def persist_documents(chunks) -> Chroma:
    persist_dir = Path(settings.chroma_persist_dir)
    _reset_persist_dir(persist_dir)
    embeddings = build_embeddings()
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_dir),
    )
    if hasattr(vectordb, "persist"):
        vectordb.persist()
    return vectordb


def chroma_is_ready() -> bool:
    persist_dir = Path(settings.chroma_persist_dir)
    if not persist_dir.exists():
        return False
    return any(persist_dir.iterdir())


def load_vectordb() -> Chroma:
    embeddings = build_embeddings()
    return Chroma(
        persist_directory=str(settings.chroma_persist_dir),
        embedding_function=embeddings,
    )


def build_qa_chain() -> RetrievalQA:
    vectordb = load_vectordb()
    llm = ChatOpenAI(
        model_name=settings.openai_model,
        openai_api_key=settings.openai_api_key or None,
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": settings.retriever_k})
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
    )
