from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from ..config import get_settings
from ..utils.logging import logger


class PolicyRAGTool:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self._persist_directory = Path(self._settings.vector_db_path)
        self._persist_directory.mkdir(parents=True, exist_ok=True)

    def _vectorstore(self) -> Chroma:
        return Chroma(
            collection_name="policies",
            embedding_function=self._embedding,
            persist_directory=str(self._persist_directory),
        )

    def ensure_index(self) -> Dict[str, Any]:
        if not any(self._persist_directory.iterdir()):
            return self.ingest(refresh=False)
        return {"status": "cached"}

    def ingest(self, refresh: bool = False) -> Dict[str, Any]:
        policy_dir = Path(self._settings.policy_documents_dir)
        if not policy_dir.exists():
            raise FileNotFoundError(f"Policy directory not found: {policy_dir}")

        vectorstore = self._vectorstore()
        if refresh:
            if self._persist_directory.exists():
                for item in self._persist_directory.iterdir():
                    if item.is_file():
                        item.unlink()
                    else:
                        shutil.rmtree(item)
            vectorstore = self._vectorstore()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
        total_chunks = 0
        for pdf_path in policy_dir.glob("*.pdf"):
            loader = PyPDFLoader(str(pdf_path))
            documents = loader.load()
            splits = text_splitter.split_documents(documents)
            for chunk in splits:
                chunk.metadata.setdefault("source", pdf_path.name)
            vectorstore.add_documents(splits)
            total_chunks += len(splits)
            logger.info(
                "policy_rag.ingested",
                file=pdf_path.name,
                chunks=len(splits),
            )

        vectorstore.persist()
        return {"status": "indexed", "chunks_indexed": total_chunks}

    def query(self, question: str, top_k: int = 4) -> Dict[str, Any]:
        vectorstore = self._vectorstore()
        matches = vectorstore.similarity_search(question, k=top_k)

        response: List[Dict[str, Any]] = []
        for doc in matches:
            response.append(
                {
                    "policy": doc.metadata.get("source"),
                    "page": doc.metadata.get("page"),
                    "score": doc.metadata.get("score"),
                    "text": doc.page_content.strip(),
                }
            )

        return {"question": question, "matches": response}
