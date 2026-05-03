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
        # 1. Retrieve chunks liên quan nhất
        results = self.store.search(question, top_k=top_k)
        
        # 2. Build prompt với context
        context_parts = [r["content"] for r in results]
        context = "\n\n".join(context_parts)
        
        prompt = f"""Dựa trên thông tin sau đây, hãy trả lời câu hỏi.

    Context:
    {context}

    Câu hỏi: {question}

    Trả lời:"""
        
        # 3. Gọi LLM
        return self.llm_fn(prompt)
