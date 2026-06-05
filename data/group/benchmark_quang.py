"""
Benchmark strategy cua Quang: MetadataAwareChunker + Gemini embedding.

Chay tu thu muc data/group:
    python benchmark_quang.py
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from dotenv import load_dotenv

load_dotenv()

from src import Document, EmbeddingStore, GeminiEmbedder


DATA_FILES = [
    ("data/quy_che_dao_tao.md", {"category": "academics", "chapter": "I"}),
    ("data/dang_ky_hoc_phan.md", {"category": "academics", "chapter": "II"}),
    ("data/danh_gia_ket_qua_hoc_tap.md", {"category": "assessment", "chapter": "III"}),
    ("data/tot_nghiep_va_bang_cap.md", {"category": "graduation", "chapter": "III"}),
    ("data/nghi_hoc_chuyen_nganh.md", {"category": "transfer", "chapter": "IV"}),
    ("data/ky_luat_sinh_vien.md", {"category": "discipline", "chapter": "IV"}),
]

QUERIES = [
    {
        "q": "Sinh viên bị cảnh báo học vụ khi nào?",
        "expected_file": "danh_gia_ket_qua_hoc_tap",
        "filter": {"category": "assessment"},
    },
    {
        "q": "Điều kiện để được xét tốt nghiệp là gì?",
        "expected_file": "tot_nghiep_va_bang_cap",
        "filter": {"category": "graduation"},
    },
    {
        "q": "Số tín chỉ tối đa có thể đăng ký trong một học kỳ chính là bao nhiêu?",
        "expected_file": "dang_ky_hoc_phan",
        "filter": {"category": "academics"},
    },
    {
        "q": "Sinh viên thi hộ bị xử lý kỷ luật như thế nào?",
        "expected_file": "ky_luat_sinh_vien",
        "filter": {"category": "discipline"},
    },
    {
        "q": "Hạng tốt nghiệp bị giảm một mức trong trường hợp nào?",
        "expected_file": "tot_nghiep_va_bang_cap",
        "filter": {"category": "graduation"},
    },
]


class CachedRateLimitedEmbedder:
    def __init__(self, embedder, delay: float = 0.65) -> None:
        self._embedder = embedder
        self._cache: dict[str, list[float]] = {}
        self._delay = delay
        self._backend_name = getattr(embedder, "_backend_name", "unknown")

    def __call__(self, text: str) -> list[float]:
        if text not in self._cache:
            time.sleep(self._delay)
            self._cache[text] = self._embedder(text)
        return self._cache[text]


class MetadataAwareChunker:
    """Split by article first, then section headings if an article is too long."""

    def __init__(self, max_chars: int = 900) -> None:
        self.max_chars = max_chars

    def chunk_with_metadata(self, text: str, base_meta: dict) -> list[tuple[str, dict]]:
        matches = list(re.finditer(r"(?m)^## Điều\s+(\d+)\.", text))
        if not matches:
            return [(text.strip(), dict(base_meta))]

        chunks: list[tuple[str, dict]] = []
        for index, match in enumerate(matches):
            start = match.start()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            article = text[start:end].strip()
            article_number = match.group(1)
            article_title = article.splitlines()[0].lstrip("# ").strip()

            meta = dict(base_meta)
            meta["article_number"] = article_number
            meta["section_title"] = article_title

            if len(article) <= self.max_chars:
                chunks.append((article, meta))
                continue

            parts = re.split(r"(?m)(?=^###\s+)", article)
            current = ""
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                candidate = f"{current}\n\n{part}".strip() if current else part
                if len(candidate) <= self.max_chars:
                    current = candidate
                else:
                    if current:
                        chunks.append((current, meta))
                    current = part
            if current:
                chunks.append((current, meta))

        return chunks


def main() -> None:
    embedder = CachedRateLimitedEmbedder(GeminiEmbedder(), delay=0.65)
    chunker = MetadataAwareChunker(max_chars=900)
    store = EmbeddingStore(collection_name="quang_metadata_aware", embedding_fn=embedder)

    docs: list[Document] = []
    for path, base_meta in DATA_FILES:
        text = Path(path).read_text(encoding="utf-8")
        file_id = Path(path).stem
        for i, (content, meta) in enumerate(chunker.chunk_with_metadata(text, base_meta)):
            meta["source"] = path
            meta["file_id"] = file_id
            meta["chunk_index"] = i
            docs.append(Document(id=f"{file_id}_q{i}", content=content, metadata=meta))

    store.add_documents(docs)
    print(f"Embedding backend: {embedder._backend_name}")
    print("STRATEGY: Quang - MetadataAwareChunker(max_chars=900)")
    print(f"Total chunks indexed: {len(docs)}")

    hits = 0
    scores: list[float] = []
    for qi, q_info in enumerate(QUERIES, 1):
        results = store.search_with_filter(
            q_info["q"],
            top_k=3,
            metadata_filter=q_info["filter"],
        )
        top1 = results[0] if results else {}
        top1_src = top1.get("metadata", {}).get("file_id", "?")
        top1_score = float(top1.get("score", 0.0))
        top3_srcs = [r.get("metadata", {}).get("file_id", "?") for r in results]
        expected = q_info["expected_file"]
        relevant = expected in top3_srcs
        if relevant:
            hits += 1
        scores.append(top1_score)
        rank = top3_srcs.index(expected) + 1 if expected in top3_srcs else "miss"
        preview = top1.get("content", "")[:90].replace("\n", " ")
        print(f"\nQ{qi}: {q_info['q']}")
        print(f"  Filter   : {q_info['filter']}")
        print(f"  Expected : {expected}")
        print(f"  Top-1    : {top1_src} (score={top1_score:.4f})")
        print(f"  Rank     : {rank}")
        print(f"  Relevant : {'yes' if relevant else 'no'}")
        print(f"  Preview  : {preview}...")

    avg_score = sum(scores) / len(scores)
    print(f"\nPrecision@3: {hits}/{len(QUERIES)} = {hits / len(QUERIES):.0%}")
    print(f"Avg Score Q1-Q5: {avg_score:.3f}")


if __name__ == "__main__":
    main()
