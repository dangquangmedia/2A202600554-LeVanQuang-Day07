# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** [Điền tên sinh viên]  
**Nhóm:** [Điền tên nhóm]  
**Ngày:** 12/05/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**  
High cosine similarity nghĩa là hai vector có hướng gần nhau, nên hai đoạn text thường có ý nghĩa hoặc chủ đề gần nhau. Điểm càng gần 1 thì mức độ tương đồng theo embedding càng cao.

**Ví dụ HIGH similarity:**
- Sentence A: Python is used for machine learning and data analysis.
- Sentence B: Data scientists use Python to train models and analyze data.
- Tại sao tương đồng: Cả hai câu đều nói về Python trong bối cảnh AI/data.

**Ví dụ LOW similarity:**
- Sentence A: Vector stores retrieve similar embeddings.
- Sentence B: The recipe requires fresh basil and tomatoes.
- Tại sao khác: Hai câu thuộc hai domain khác nhau, một câu về retrieval system và một câu về nấu ăn.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**  
Cosine similarity tập trung vào hướng của vector, phù hợp khi ta quan tâm đến ý nghĩa hơn là độ lớn tuyệt đối. Với text embeddings đã normalize, cosine similarity giúp so sánh semantic similarity ổn định hơn.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**  
Step size = 500 - 50 = 450.  
Chunk count = ceil((10000 - 500) / 450) + 1 = ceil(9500 / 450) + 1 = 23 chunks.

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**  
Step size = 500 - 100 = 400, nên chunk count = ceil(9500 / 400) + 1 = 25 chunks. Overlap nhiều hơn giúp giữ ngữ cảnh ở ranh giới chunk, nhưng làm tăng số chunk và chi phí lưu/truy vấn.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Internal knowledge assistant / RAG documentation.

**Tại sao nhóm chọn domain này?**  
Domain này phù hợp với Lab 7 vì có nhiều tài liệu dạng markdown/text, có nhu cầu semantic search rõ ràng, và dễ kiểm thử bằng benchmark queries. Nội dung cũng bao gồm cả tài liệu tiếng Anh và tiếng Việt nên có thể quan sát failure cases khi embedding chưa đủ tốt cho multilingual retrieval.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | python_intro | data/python_intro.txt | 1944 | source, extension, domain |
| 2 | vector_store_notes | data/vector_store_notes.md | 2123 | source, extension, domain |
| 3 | rag_system_design | data/rag_system_design.md | 2391 | source, extension, domain |
| 4 | customer_support_playbook | data/customer_support_playbook.txt | 1692 | source, extension, domain |
| 5 | chunking_experiment_report | data/chunking_experiment_report.md | 1987 | source, extension, domain |
| 6 | vi_retrieval_notes | data/vi_retrieval_notes.md | 1667 | source, extension, domain |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| source | string | data/vector_store_notes.md | Giúp trace kết quả về file gốc. |
| extension | string | .md | Phân biệt markdown và plain text. |
| domain | string | internal_knowledge | Lọc theo bộ tài liệu/domain đang index. |
| language | string | en / vi | Hữu ích khi cần lọc tài liệu theo ngôn ngữ. |
| department | string | support / engineering | Tránh lấy nhầm tài liệu từ phòng ban không liên quan. |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` với `chunk_size=500`:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| python_intro | FixedSizeChunker (`fixed_size`) | 5 | 428.8 | Trung bình, có thể cắt ngang câu. |
| python_intro | SentenceChunker (`by_sentences`) | 5 | 387.0 | Tốt, giữ câu nguyên vẹn. |
| python_intro | RecursiveChunker (`recursive`) | 5 | 387.2 | Tốt, giữ đoạn/câu tự nhiên. |
| vector_store_notes | FixedSizeChunker (`fixed_size`) | 5 | 464.6 | Trung bình. |
| vector_store_notes | SentenceChunker (`by_sentences`) | 8 | 263.6 | Tốt nhưng nhiều chunk nhỏ hơn. |
| vector_store_notes | RecursiveChunker (`recursive`) | 7 | 301.6 | Tốt nhất cho markdown có heading/paragraph. |
| rag_system_design | FixedSizeChunker (`fixed_size`) | 6 | 440.2 | Trung bình. |
| rag_system_design | SentenceChunker (`by_sentences`) | 5 | 476.0 | Tốt nhưng chunk hơi dài. |
| rag_system_design | RecursiveChunker (`recursive`) | 7 | 339.9 | Cân bằng giữa kích thước và ngữ cảnh. |

### Strategy Của Tôi

**Loại:** RecursiveChunker.

**Mô tả cách hoạt động:**  
`RecursiveChunker` tách văn bản theo danh sách separator ưu tiên: paragraph, dòng mới, câu, khoảng trắng, rồi cuối cùng là ký tự. Nếu một đoạn đã ngắn hơn `chunk_size` thì giữ nguyên. Nếu đoạn còn quá dài, thuật toán tiếp tục tách bằng separator nhỏ hơn.

**Tại sao tôi chọn strategy này cho domain nhóm?**  
Tài liệu nhóm gồm markdown, report, playbook và ghi chú tiếng Việt nên có cấu trúc paragraph/heading rõ ràng. Recursive chunking giữ được nhiều ngữ cảnh hơn fixed-size và ổn định hơn sentence-only chunking khi câu dài.

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| vector_store_notes | FixedSizeChunker | 5 | 464.6 | Có thể trả về đoạn bị cắt ngang. |
| vector_store_notes | **RecursiveChunker** | 7 | 301.6 | Tốt hơn vì chunk bám theo heading/paragraph. |
| rag_system_design | SentenceChunker | 5 | 476.0 | Dễ đọc nhưng một số chunk dài. |
| rag_system_design | **RecursiveChunker** | 7 | 339.9 | Cân bằng hơn cho retrieval. |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | RecursiveChunker | 8 | Giữ paragraph/context tốt. | Cần tuning separator và chunk size. |
| Thành viên A | FixedSizeChunker | 6 | Đơn giản, predictable. | Có thể cắt ngang ý. |
| Thành viên B | SentenceChunker | 7 | Chunk dễ đọc. | Chunk count/kích thước không đều. |

**Strategy nào tốt nhất cho domain này? Tại sao?**  
RecursiveChunker là lựa chọn tốt nhất vì tài liệu có cấu trúc rõ theo đoạn và heading. Nó giảm nguy cơ mất ngữ cảnh ở ranh giới chunk nhưng vẫn kiểm soát được độ dài.

---

## 4. My Approach — Cá nhân (10 điểm)

### Chunking Functions

**`SentenceChunker.chunk` — approach:**  
Hàm dùng regex để tách câu theo dấu `.`, `!`, `?` kết hợp whitespace/newline, sau đó loại câu rỗng và gom tối đa `max_sentences_per_chunk` câu vào một chunk. Edge case quan trọng là text rỗng và giá trị max sentence nhỏ hơn 1.

**`RecursiveChunker.chunk` / `_split` — approach:**  
Base case là text đã ngắn hơn `chunk_size` thì trả về ngay. Nếu còn quá dài, hàm thử tách bằng separator hiện tại, gom các phần nhỏ lại, và gọi đệ quy với separator tiếp theo khi cần.

### EmbeddingStore

**`add_documents` + `search` — approach:**  
Mặc định store dùng in-memory list để test deterministic và không phụ thuộc ChromaDB cũ trên disk. Mỗi document được lưu cùng content, metadata và embedding; query được embed rồi so điểm dot product với các vector đã normalize.

**`search_with_filter` + `delete_document` — approach:**  
Với in-memory backend, filter metadata được áp dụng trước khi tính similarity để giảm candidate không liên quan. `delete_document` xóa record theo `id` hoặc metadata `doc_id` và trả về `True/False` tùy có xóa được hay không.

### KnowledgeBaseAgent

**`answer` — approach:**  
Agent retrieve top-k chunks từ `EmbeddingStore`, nối content thành phần `Context`, rồi đưa vào prompt cùng câu hỏi. LLM chỉ nhận prompt đã có retrieved context nên câu trả lời được grounded vào tài liệu đã tìm thấy.

### Test Results

```text
pytest tests/ -v
collected 42 items
42 passed in 0.13s
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

Kết quả dùng deterministic `MockEmbedder`, nên score không phản ánh semantic embedding thật như OpenAI/SentenceTransformer.

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Python supports machine learning workflows. | Python is used for AI and data science. | high | 0.0940 | Một phần |
| 2 | Vector stores retrieve similar embeddings. | A database can rank vectors by semantic similarity. | high | -0.1997 | Không |
| 3 | Metadata filters improve retrieval precision. | The recipe requires fresh basil and tomatoes. | low | 0.0084 | Có |
| 4 | Recursive chunking preserves paragraph context. | Recursive splitting tries larger separators before smaller ones. | high | 0.1266 | Một phần |
| 5 | Billing errors should be escalated when documents are missing. | Password recovery starts from the account settings page. | low | -0.0150 | Có |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**  
Pair 2 bất ngờ nhất vì hai câu cùng nói về vector store nhưng score âm. Điều này cho thấy `MockEmbedder` chỉ phù hợp để test interface deterministic, không phù hợp để đánh giá chất lượng semantic retrieval thật.

---

## 6. Results — Cá nhân (10 điểm)

### Benchmark Queries & Gold Answers

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | What is Python commonly used for in production? | Python is used for APIs, data pipelines, internal tools, model serving, data analysis, and AI workflows. |
| 2 | Why does metadata matter in vector stores? | Metadata narrows retrieval by source, language, department, product area, date, and access level. |
| 3 | What should a RAG assistant do when retrieved evidence is weak? | It should state uncertainty or escalate instead of pretending the answer is complete. |
| 4 | Which chunking strategy worked best for mixed technical documentation? | Recursive chunking gave the best balance of context preservation and size control. |
| 5 | Vì sao chunking ảnh hưởng đến retrieval? | Chunk quá ngắn thiếu ngữ cảnh, chunk quá dài gộp nhiều ý không liên quan và làm giảm độ chính xác. |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | What is Python commonly used for in production? | `python_intro`: Python used for automation, backend, data analysis, scientific computing, ML. | 0.2010 | Yes | Trả lời dựa trên context về Python production use. |
| 2 | Why does metadata matter in vector stores? | `vector_store_notes`: vector store workflow and metadata section. | 0.2272 | Yes | Nêu metadata giúp lưu source/language/department và filter retrieval. |
| 3 | What should a RAG assistant do when retrieved evidence is weak? | `customer_support_playbook`: insufficient retrieval should escalate instead of improvising. | 0.2499 | Yes | Trả lời rằng nên uncertainty/escalation khi thiếu tài liệu. |
| 4 | Which chunking strategy worked best for mixed technical documentation? | `chunking_experiment_report`: comparison of fixed, sentence, recursive chunking. | 0.1241 | Yes | Nêu recursive chunking cân bằng context và size. |
| 5 | Vì sao chunking ảnh hưởng đến retrieval? | `customer_support_playbook`: support assistant content, not the Vietnamese retrieval note. | 0.1764 | No | Failure case: mock embedding không retrieve đúng tài liệu tiếng Việt. |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5  
**Bao nhiêu queries có Top-1 relevant?** 4 / 5

**Failure case chính:**  
Query tiếng Việt có `vi_retrieval_notes` trong top-3 nhưng không đứng top-1. Nguyên nhân có khả năng là `MockEmbedder` không học semantic multilingual thật, nên các score chỉ phục vụ kiểm thử hành vi hệ thống chứ không đảm bảo chất lượng retrieval.

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**  
Fixed-size chunking dễ debug vì chunk count predictable, nhưng không nên dùng một mình cho tài liệu có nhiều câu/paragraph dài. Cách so sánh nhiều strategy trên cùng query giúp thấy rõ trade-off giữa độ dài chunk và khả năng giữ context.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**  
Một nhóm khác dùng metadata filter theo department/language để giảm nhiễu retrieval. Điều đó cho thấy chất lượng RAG không chỉ phụ thuộc embedding mà còn phụ thuộc schema dữ liệu.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**  
Tôi sẽ thêm metadata `language`, `department`, `audience`, và `updated_at` cho từng tài liệu. Tôi cũng sẽ dùng embedding model thật cho benchmark cuối, ví dụ SentenceTransformer hoặc OpenAI embedding, để đánh giá semantic retrieval chính xác hơn.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 9 / 10 |
| Chunking strategy | Nhóm | 14 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 4 / 5 |
| Results | Cá nhân | 9 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 4 / 5 |
| **Tổng** | | **85 / 100** |
