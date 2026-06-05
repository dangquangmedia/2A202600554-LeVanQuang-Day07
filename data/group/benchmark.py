"""
Benchmark 3 chunking strategies trên dataset quy che DHSPKTHY.
Dung Gemini embedding de co ket qua co ngu nghia thuc.
"""
import sys, os, time
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")
from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
from src import (
    FixedSizeChunker, SentenceChunker, RecursiveChunker,
    EmbeddingStore, Document, GeminiEmbedder, compute_similarity,
)

DATA_FILES = [
    ("data/quy_che_dao_tao.md",        {"category": "academics",   "chapter": "I"}),
    ("data/dang_ky_hoc_phan.md",       {"category": "academics",   "chapter": "II"}),
    ("data/danh_gia_ket_qua_hoc_tap.md",{"category": "academics",  "chapter": "III"}),
    ("data/tot_nghiep_va_bang_cap.md", {"category": "graduation",  "chapter": "III"}),
    ("data/nghi_hoc_chuyen_nganh.md",  {"category": "transfer",    "chapter": "IV"}),
    ("data/ky_luat_sinh_vien.md",      {"category": "discipline",  "chapter": "IV"}),
]

QUERIES = [
    {
        "q": "Sinh viên bị cảnh báo học vụ khi nào?",
        "gold": "Khi tín chỉ không đạt > 50% đăng ký, hoặc nợ > 24 tín chỉ; điểm TB học kỳ < 0.8 (kỳ 1) / < 1.0; điểm TB tích lũy dưới ngưỡng theo năm. Tối đa 4 lần, không quá 2 lần liên tiếp.",
        "expected_file": "danh_gia_ket_qua_hoc_tap",
        "filter": None,
    },
    {
        "q": "Điều kiện để được xét tốt nghiệp là gì?",
        "gold": "Tích lũy đủ tín chỉ; điểm TB tích lũy ≥ 2.0; không bị truy cứu hình sự hoặc đình chỉ; có đơn xin xét tốt nghiệp nếu tốt nghiệp sớm/muộn.",
        "expected_file": "tot_nghiep_va_bang_cap",
        "filter": {"category": "graduation"},
    },
    {
        "q": "Số tín chỉ tối đa có thể đăng ký trong một học kỳ chính là bao nhiêu?",
        "gold": "Cử nhân: tối đa 26 tín chỉ. Kỹ sư: tối đa 30 tín chỉ (kỹ sư tài năng: 33). Học kỳ hè: tối đa 8 tín chỉ.",
        "expected_file": "dang_ky_hoc_phan",
        "filter": None,
    },
    {
        "q": "Sinh viên thi hộ bị xử lý kỷ luật như thế nào?",
        "gold": "Lần 1: đình chỉ học 1 năm. Lần 2: buộc thôi học. Áp dụng cả người thi hộ và người nhờ thi hộ.",
        "expected_file": "ky_luat_sinh_vien",
        "filter": {"category": "discipline"},
    },
    {
        "q": "Hạng tốt nghiệp bị giảm một mức trong trường hợp nào?",
        "gold": "Khi khối lượng học lại > 5% tổng tín chỉ chương trình, hoặc đã bị kỷ luật từ mức cảnh cáo trở lên trong thời gian học.",
        "expected_file": "tot_nghiep_va_bang_cap",
        "filter": {"category": "graduation"},
    },
]

STRATEGIES = {
    "FixedSize(500, overlap=50)":   FixedSizeChunker(chunk_size=500, overlap=50),
    "Sentence(max=3)":              SentenceChunker(max_sentences_per_chunk=3),
    "Recursive(300)":               RecursiveChunker(chunk_size=300),
}


class CachedRateLimitedEmbedder:
    """Wraps an embedder with an in-memory cache and per-call delay to respect rate limits."""

    def __init__(self, embedder, delay: float = 0.65):
        self._embedder = embedder
        self._cache: dict[str, list[float]] = {}
        self._delay = delay
        self._backend_name = getattr(embedder, "_backend_name", "unknown")

    def __call__(self, text: str) -> list[float]:
        if text not in self._cache:
            time.sleep(self._delay)
            self._cache[text] = self._embedder(text)
        return self._cache[text]


raw_embedder = GeminiEmbedder()
embedder = CachedRateLimitedEmbedder(raw_embedder, delay=0.65)
print(f"Embedding backend: {embedder._backend_name}\n")
print("=" * 70)

all_results = {}

for strategy_name, chunker in STRATEGIES.items():
    print(f"\n{'='*70}")
    print(f"STRATEGY: {strategy_name}")
    print(f"{'='*70}")

    store = EmbeddingStore(collection_name=strategy_name, embedding_fn=embedder)
    docs = []
    total_chunks = 0

    for path, base_meta in DATA_FILES:
        text = Path(path).read_text(encoding="utf-8")
        chunks = chunker.chunk(text)
        total_chunks += len(chunks)
        file_id = Path(path).stem
        for i, chunk_text in enumerate(chunks):
            meta = dict(base_meta)
            meta["source"] = path
            meta["file_id"] = file_id   # preserve original file stem
            meta["chunk_index"] = i
            docs.append(Document(
                id=f"{file_id}_chunk{i}",
                content=chunk_text,
                metadata=meta,
            ))

    store.add_documents(docs)
    print(f"  Total chunks indexed: {total_chunks} (avg {total_chunks//len(DATA_FILES)}/file)")

    hits = 0
    strategy_results = []

    for qi, q_info in enumerate(QUERIES, 1):
        query = q_info["q"]
        expected = q_info["expected_file"]
        mfilter = q_info["filter"]

        if mfilter:
            results = store.search_with_filter(query, top_k=3, metadata_filter=mfilter)
        else:
            results = store.search(query, top_k=3)

        top1 = results[0] if results else {}
        top1_src = top1.get("metadata", {}).get("file_id", "?")
        top1_score = top1.get("score", 0)
        top1_preview = top1.get("content", "")[:80].replace("\n", " ")

        top3_srcs = [r.get("metadata", {}).get("file_id", "?") for r in results]
        relevant = expected in top3_srcs
        if relevant:
            hits += 1

        rank = top3_srcs.index(expected) + 1 if expected in top3_srcs else "miss"
        tag = "✓" if relevant else "✗"

        print(f"\n  Q{qi}: {query}")
        print(f"    Expected : {expected}")
        print(f"    Top-1    : {top1_src} (score={top1_score:.4f}) {tag}")
        print(f"    Rank     : {rank}")
        print(f"    Preview  : {top1_preview}...")

        strategy_results.append({
            "query": query,
            "expected": expected,
            "top1_src": top1_src,
            "top1_score": top1_score,
            "relevant": relevant,
            "rank": rank,
            "top1_preview": top1_preview,
        })

    precision = hits / len(QUERIES)
    print(f"\n  >> Precision@3: {hits}/{len(QUERIES)} = {precision:.0%}")
    all_results[strategy_name] = {
        "total_chunks": total_chunks,
        "avg_chunks_per_file": total_chunks // len(DATA_FILES),
        "hits": hits,
        "precision": precision,
        "details": strategy_results,
    }

print(f"\n{'='*70}")
print("TONG KET SO SANH")
print(f"{'='*70}")
print(f"{'Strategy':<30} {'Chunks':>8} {'Precision':>12}")
print("-" * 55)
for name, res in all_results.items():
    print(f"{name:<30} {res['total_chunks']:>8} {res['precision']:>11.0%}")
