from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        # 1. Retrieve top-k relevant chunks
        results = self.store.search(question, top_k=top_k)

        # 2. If no results, ask LLM to say it doesn't know
        if not results:
            prompt = (
                "No relevant context was found in the knowledge base.\n"
                f"Question: {question}\n"
                "If you do not know, say you do not know."
            )
            return self.llm_fn(prompt)

        # 3. Build context blocks with metadata and scores
        context_blocks = []
        for index, result in enumerate(results, start=1):
            metadata = result.get("metadata", {})
            source = metadata.get("source", metadata.get("doc_id", "unknown"))
            score = result.get("score", 0.0)
            content = result.get("content", "")

            context_blocks.append(
                f"[Chunk {index} | source={source} | score={score:.4f}]\n"
                f"{content}"
            )

        context = "\n\n".join(context_blocks)

        prompt = (
            "You are a knowledge-base assistant. "
            "Answer the question using only the context below.\n"
            "If the answer is not supported by the context, say you do not know.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

        return self.llm_fn(prompt)
