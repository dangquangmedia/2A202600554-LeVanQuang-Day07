from src import EmbeddingStore, Document

store = EmbeddingStore("my_kb")
print("Using Chroma:", store._use_chroma)  # True nếu đã cài chromadb

store.add_documents([
    Document("doc1", "Chinh sach hoan tien trong 7 ngay.", {"category": "support"}),
    Document("doc2", "Giao hang mien phi don tren 500k.", {"category": "shipping"}),
])

print("Total docs:", store.get_collection_size())

results = store.search("hoan tien", top_k=2)
for r in results:
    print(f"[{r['score']:.3f}] {r['content']}")