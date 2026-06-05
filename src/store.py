from __future__ import annotations

import os
from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
        persist_directory: str | None = None,
        use_chroma: bool | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        should_use_chroma = (
            use_chroma
            if use_chroma is not None
            else os.getenv("EMBEDDING_STORE_BACKEND", "").strip().lower() == "chroma"
        )

        if not should_use_chroma:
            return

        try:
            import chromadb

            db_path = persist_directory or os.getenv("CHROMA_DB_PATH", "./chroma_db")
            client = chromadb.PersistentClient(path=db_path)
            self._collection = client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        metadata = dict(doc.metadata or {})
        metadata.setdefault("doc_id", doc.id)

        record_id = f"{doc.id}-{self._next_index}"
        self._next_index += 1

        return {
            "id": record_id,
            "doc_id": doc.id,
            "content": doc.content,
            "metadata": metadata,
            "embedding": self._embedding_fn(doc.content),
        }

    def add_documents(self, docs: list[Document]) -> None:
        if self._use_chroma:
            ids = [doc.id for doc in docs]
            texts = [doc.content for doc in docs]
            embeddings = [self._embedding_fn(doc.content) for doc in docs]
            metadatas = [dict(doc.metadata) or None for doc in docs]
            self._collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )
        else:
            for doc in docs:
                record = self._make_record(doc)
                self._store.append(record)

    def get_collection_size(self) -> int:
        if self._use_chroma:
            return self._collection.count()
        return len(self._store)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if self._use_chroma:
            q_vec = self._embedding_fn(query)
            res = self._collection.query(
                query_embeddings=[q_vec],
                n_results=min(top_k, self._collection.count()),
            )
            results = []
            for i, doc_id in enumerate(res["ids"][0]):
                results.append({
                    "id":       doc_id,
                    "content":  res["documents"][0][i],
                    "metadata": res["metadatas"][0][i] if res["metadatas"] else {},
                    "score":    1 - res["distances"][0][i],  # cosine: distance → similarity
                })
            return results
        return self._search_records(query, self._store, top_k)

    def _search_records(self, query: str, records: list[dict], top_k: int) -> list[dict]:
        if not records:
            return []
        query_embedding = self._embedding_fn(query)
        scored: list[dict] = []
        for record in records:
            score = _dot(query_embedding, record["embedding"])
            scored.append({
                "id": record["id"],
                "content": record["content"],
                "metadata": record.get("metadata", {}),
                "score": score,
            })

        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:max(0, top_k)]

    def search_with_filter(self, query: str, top_k: int = 3,
                        metadata_filter: dict | None = None) -> list[dict[str, Any]]:
        if self._use_chroma:
            q_vec = self._embedding_fn(query)
            kwargs = dict(
                query_embeddings=[q_vec],
                n_results=min(top_k, self._collection.count()),
            )
            if metadata_filter:
                # Chroma dùng $eq operator
                if len(metadata_filter) == 1:
                    k, v = next(iter(metadata_filter.items()))
                    kwargs["where"] = {k: {"$eq": v}}
                else:
                    kwargs["where"] = {
                        "$and": [{k: {"$eq": v}} for k, v in metadata_filter.items()]
                    }
            res = self._collection.query(**kwargs)
            return [
                {
                    "id":       res["ids"][0][i],
                    "content":  res["documents"][0][i],
                    "metadata": res["metadatas"][0][i] if res["metadatas"] else {},
                    "score":    1 - res["distances"][0][i],
                }
                for i in range(len(res["ids"][0]))
            ]
        # fallback in-memory
        if not metadata_filter:
            return self.search(query, top_k)

        filtered = []
        for r in self._store:
            metadata = r.get("metadata", {})
            if all(metadata.get(k) == v for k, v in metadata_filter.items()):
                filtered.append(r)

        return self._search_records(query, filtered, top_k)

    def delete_document(self, doc_id: str) -> bool:
        if self._use_chroma:
            before = self._collection.count()
            # Thử xóa theo id trực tiếp
            try:
                self._collection.delete(ids=[doc_id])
            except Exception:
                return False
            return self._collection.count() < before
        before = len(self._store)
        self._store = [
            r for r in self._store
            if r["id"] != doc_id and r["metadata"].get("doc_id") != doc_id
        ]
        return len(self._store) < before
