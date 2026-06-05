# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Lê Văn Quang  
**Nhóm:** B2  
**Ngày:** 05/06/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**  
High cosine similarity nghĩa là hai vector embedding có hướng gần nhau trong không gian vector. Với văn bản, điều này thường cho thấy hai câu hoặc hai đoạn có ý nghĩa gần nhau, dù cách dùng từ có thể khác nhau.

**Ví dụ HIGH similarity:**
- Sentence A: "Sinh viên bị cảnh báo học vụ khi điểm trung bình dưới quy định."
- Sentence B: "Kết quả học tập thấp có thể khiến sinh viên bị cảnh báo."
- Tại sao tương đồng: Cả hai câu đều nói về cảnh báo học vụ do kết quả học tập thấp.

**Ví dụ LOW similarity:**
- Sentence A: "Sinh viên bị cảnh báo học vụ khi điểm trung bình dưới quy định."
- Sentence B: "Thư viện mở cửa từ 7 giờ sáng đến 9 giờ tối."
- Tại sao khác: Hai câu thuộc hai chủ đề khác nhau, một câu về học vụ và một câu về dịch vụ thư viện.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**  
Cosine similarity tập trung vào hướng của vector thay vì độ lớn tuyệt đối. Điều này phù hợp với text embeddings vì ta quan tâm hai văn bản có cùng ý nghĩa hay không hơn là vector dài/ngắn bao nhiêu. Euclidean distance dễ bị ảnh hưởng bởi magnitude của vector, nhất là khi văn bản có độ dài khác nhau.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**  
Step size = 500 - 50 = 450.  
Số chunks = ceil((10000 - 500) / 450) + 1 = ceil(9500 / 450) + 1 = 23 chunks.

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**  
Step size = 500 - 100 = 400.  
Số chunks = ceil((10000 - 500) / 400) + 1 = 25 chunks. Overlap lớn hơn giúp giữ ngữ cảnh ở ranh giới chunk tốt hơn, nhưng làm tăng số chunk, thời gian embedding và chi phí truy vấn.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Quy định học vụ — Quy chế đào tạo trình độ đại học của Trường Đại học Sư phạm Kỹ thuật Hưng Yên.

**Tại sao nhóm chọn domain này?**  
Domain quy định học vụ phù hợp với RAG vì tài liệu có cấu trúc rõ theo chương, điều, mục và bảng. Người dùng có nhu cầu hỏi đáp thực tế như đăng ký học phần, cảnh báo học tập, tốt nghiệp, kỷ luật, nghỉ học hoặc chuyển ngành. Các gold answers cũng dễ kiểm chứng trực tiếp từ tài liệu gốc.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | quy_che_dao_tao.md | QĐ 952/QĐ-ĐHSPKT 2021, Điều 1-5 | 3.709 | category=academics, chapter=I, language=vi |
| 2 | dang_ky_hoc_phan.md | QĐ 952/QĐ-ĐHSPKT 2021, Điều 9-12 | 3.894 | category=academics, chapter=II, language=vi |
| 3 | danh_gia_ket_qua_hoc_tap.md | QĐ 952/QĐ-ĐHSPKT 2021, Điều 13-16 | 4.634 | category=assessment, chapter=III, language=vi |
| 4 | tot_nghiep_va_bang_cap.md | QĐ 952/QĐ-ĐHSPKT 2021, Điều 18-20 | 3.670 | category=graduation, chapter=III, language=vi |
| 5 | nghi_hoc_chuyen_nganh.md | QĐ 952/QĐ-ĐHSPKT 2021, Điều 21-25 | 4.211 | category=transfer, chapter=IV, language=vi |
| 6 | ky_luat_sinh_vien.md | QĐ 952/QĐ-ĐHSPKT 2021, Điều 26-27 | 3.582 | category=discipline, chapter=IV, language=vi |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|-----------------|------|---------------|--------------------------------|
| category | string | academics / assessment / graduation / discipline / transfer | Lọc theo nhóm chủ đề trước khi rank similarity |
| chapter | string | I / II / III / IV | Giữ liên kết với cấu trúc chương của quy chế |
| article_number | string | 10 / 16 / 20 / 26 | Xác định điều quy chế chứa thông tin |
| section_title | string | so_luong_tin_chi / hang_tot_nghiep | Phân biệt các mục nhỏ trong cùng một điều |
| source | string | data/tot_nghiep_va_bang_cap.md | Truy vết kết quả về tài liệu gốc |
| file_id | string | tot_nghiep_va_bang_cap | So sánh top-k với expected file trong benchmark |
| language | string | vi | Đảm bảo benchmark chạy trên tài liệu tiếng Việt |

---

## 3. Strategy Design — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Nhóm dùng `ChunkingStrategyComparator().compare()` để quan sát baseline chunking. Đây chỉ là baseline chung để hiểu dữ liệu. Strategy cá nhân của tôi là **MetadataAwareChunker(max_chars=900) + metadata schema/filtering**.

| Tài liệu | Strategy | Chunk Count | Avg Length | Nhận xét |
|----------|----------|-------------|------------|----------|
| quy_che_dao_tao.md | fixed_size | 19 | 195.21 | Dễ cắt ngang câu hoặc bullet |
| quy_che_dao_tao.md | by_sentences | 6 | 616.0 | Giữ câu nhưng chunk dài |
| quy_che_dao_tao.md | paragraph-based | 29 | 126.28 | Tôn trọng paragraph tốt hơn |
| danh_gia_ket_qua_hoc_tap.md | fixed_size | 24 | 193.08 | Có thể cắt ngang bảng điểm |
| danh_gia_ket_qua_hoc_tap.md | by_sentences | 4 | 1156.75 | Chunk quá lớn, nhiều ý lẫn nhau |
| danh_gia_ket_qua_hoc_tap.md | paragraph-based | 34 | 134.53 | Tách nhỏ, bám đoạn |
| tot_nghiep_va_bang_cap.md | fixed_size | 19 | 193.16 | Dễ mất context bảng hạng tốt nghiệp |
| tot_nghiep_va_bang_cap.md | by_sentences | 5 | 731.6 | Ít chunk nhưng quá rộng |
| tot_nghiep_va_bang_cap.md | paragraph-based | 28 | 129.18 | Tốt cho paragraph, nhưng chưa gắn metadata điều/mục |

### Strategy Của Tôi

**Loại:** Metadata-aware chunking strategy.

**Chunking method:** `MetadataAwareChunker` tách tài liệu theo heading `## Điều ...` trước. Nếu một Điều dài hơn `max_chars`, chunker tách tiếp theo heading `###`. Cách này giữ ngữ cảnh Điều/Mục tốt hơn việc cắt cứng theo số ký tự.

**Tham số:** `max_chars=900`.

**Metadata schema:** mỗi chunk được gán:
- `category`: nhóm chủ đề, ví dụ `academics`, `assessment`, `graduation`, `discipline`, `transfer`
- `chapter`: chương trong quy chế
- `article_number`: số điều, ví dụ `10`, `16`, `20`, `26`
- `section_title`: tiêu đề điều hoặc mục nhỏ
- `source`, `file_id`, `chunk_index`, `language`

**Tại sao tôi chọn strategy này?**  
Tài liệu học vụ có nhiều từ chung như "sinh viên", "học kỳ", "tín chỉ", "điểm". Nếu chỉ search semantic trên toàn bộ chunks, kết quả có thể bị nhiễu giữa đăng ký học phần, tốt nghiệp và cảnh báo học vụ. Metadata-aware chunking giúp mỗi chunk vừa có nội dung đủ lớn để giữ context, vừa có nhãn để filter đúng vùng tài liệu trước khi tính similarity.

**Code minh họa:**

```python
chunker = MetadataAwareChunker(max_chars=900)

meta = {
    "category": "graduation",
    "chapter": "III",
    "article_number": "20",
    "section_title": "Cong Nhan Tot Nghiep",
    "source": "data/tot_nghiep_va_bang_cap.md",
    "language": "vi",
}

results = store.search_with_filter(
    "Điều kiện để được xét tốt nghiệp là gì?",
    top_k=3,
    metadata_filter={"category": "graduation"},
)
```

### So Sánh Với Thành Viên Khác

Benchmark chạy trên cùng 5 queries và cùng bộ 6 tài liệu. Kết quả của Quang được chạy lại bằng `data/group/benchmark_quang.py`, Gemini embedding `models/gemini-embedding-001`.

| Thành viên | Strategy | Chunks (6 files) | Avg Score Q1-Q5 | Precision@3 / Score | Điểm mạnh | Điểm yếu |
|------------|----------|------------------|-----------------|---------------------|-----------|----------|
| Thu Linh | RecursiveChunker(300) | 107 | 0.853 | 5/5 = 100% | Score cao nhất, chunk khớp cấu trúc Điều/Khoản | Nhiều chunk nhỏ (107), tốc độ index chậm hơn |
| Bút | Semantic Chunking | - | - | 10/10 | Lọc cực kỳ thông minh, chunk ngữ nghĩa trọn vẹn | Thời gian chạy cực kỳ chậm vì embed từng câu |
| Đức | ArticleSectionChunker(1400) + VietnameseTextEmbedder | - | - | 10/10 | Giữ article context, top-1 expected file 5/5, không cần dependency ngoài | Lexical embedder chưa hiểu paraphrase xa như semantic embedding thật |
| Hải An | FixedSize(500, overlap=50) | 56 | 0.790 | 5/5 = 100% | Kiểm soát kích thước tốt | Score thấp hơn các strategy semantic/recursive, overlap tạo chunk dư thừa |
| **Quang** | **MetadataAwareChunker(900) + metadata schema/filter** | **33** | **0.644** | **5/5 = 100%** | Tư tưởng metadata phù hợp pháp luật, filter theo danh mục/điều số, dễ trace nguồn | Score thấp nhất; filter hiện chưa đủ chi tiết nên bỏ lỡ nhiều kết quả liên quan |

**Chi tiết score theo query của Quang:**

| Query | Metadata filter | Top-1 Retrieved | Score | Rank |
|-------|-----------------|-----------------|-------|------|
| Q1: Cảnh báo học vụ | `category=assessment` | `danh_gia_ket_qua_hoc_tap` | 0.6789 | 1 |
| Q2: Điều kiện tốt nghiệp | `category=graduation` | `tot_nghiep_va_bang_cap` | 0.6299 | 1 |
| Q3: Tín chỉ đăng ký tối đa | `category=academics` | `dang_ky_hoc_phan` | 0.6982 | 1 |
| Q4: Sinh viên thi hộ | `category=discipline` | `ky_luat_sinh_vien` | 0.6643 | 1 |
| Q5: Hạng tốt nghiệp giảm | `category=graduation` | `tot_nghiep_va_bang_cap` | 0.5483 | 1 |
| **Precision@3** |  |  |  | **100%** |

**Ranking theo domain pháp luật:**

1. **RecursiveChunker(300) - Thu Linh:** tốt nhất về metric. Score cao nhất (0.853), Precision@3 = 100%, chunk theo cấu trúc tự nhiên của văn bản pháp luật (Điều -> Khoản -> điểm). Nhược điểm 107 chunks là chấp nhận được vì domain pháp luật cần granularity cao khi người dùng hỏi cụ thể từng điều/khoản.
2. **ArticleSectionChunker(1400) - Đức:** phù hợp về ngữ nghĩa. Strategy giữ nguyên context của từng điều, tránh cắt giữa chừng một quy định. Điểm yếu là chunk 1400 lớn có thể trả về quá nhiều nội dung không liên quan, và lexical embedder yếu khi câu hỏi paraphrase xa.
3. **Semantic Chunking - Bút:** tốt nhưng không thực tế cho production lớn. Chunk ngữ nghĩa trọn vẹn và lọc thông minh, nhưng thời gian chạy rất chậm vì phải embed/tính similarity từng câu. Phù hợp làm baseline đánh giá chất lượng hơn là deploy thật.
4. **FixedSize(500) - Hải An:** ổn định nhưng không phù hợp nhất với domain pháp luật. Score 0.790 thấp hơn các strategy trên; cắt cố định không tôn trọng ranh giới Điều/Khoản nên có nguy cơ chunk chứa nửa điều này, nửa điều khác.
5. **MetadataAwareChunker - Quang:** ý tưởng metadata rất hay cho pháp luật vì có thể tra cứu theo điều số và danh mục. Tuy nhiên kết quả hiện tại có score thấp nhất (0.644) vì filter mới dùng category-level và chưa đủ chi tiết, nên có thể bỏ lỡ nhiều kết quả liên quan.

**Strategy nào tốt nhất cho domain này? Tại sao?**  
Xét riêng metric retrieval trên benchmark nhóm, RecursiveChunker(300) của Thu Linh là strategy tốt nhất vì score cao nhất và vẫn đạt Precision@3 = 100%. Tuy nhiên, strategy MetadataAwareChunker của tôi vẫn có giá trị thực tế: nó giúp giải thích nguồn, filter theo danh mục/điều số, và tạo nền tảng tốt nếu metadata được làm mịn hơn ở các bước sau.

---

## 4. My Approach — Cá nhân (10 điểm)

### Chunking Functions

**`SentenceChunker.chunk` — approach:**  
Hàm dùng regex để tách câu theo dấu `.`, `!`, `?` kết hợp whitespace/newline, lọc câu rỗng rồi gom tối đa `max_sentences_per_chunk` câu vào mỗi chunk.

**Paragraph-based chunking implementation — approach:**  
Hàm `_split` dùng base case khi text đã ngắn hơn `chunk_size`; nếu text còn dài thì thử tách theo separator ưu tiên như paragraph, newline, câu, khoảng trắng, rồi cuối cùng mới cắt theo ký tự.

### EmbeddingStore

**`add_documents` + `search` — approach:**  
`add_documents` lưu từng `Document` cùng content, metadata và embedding. `search` embed query, tính similarity với từng vector đã lưu, sort giảm dần theo score rồi trả top-k.

**`search_with_filter` + `delete_document` — approach:**  
`search_with_filter` lọc metadata trước khi tính similarity, phù hợp với strategy metadata của tôi. `delete_document` xóa document theo id hoặc `metadata["doc_id"]`, trả `True/False` tùy có xóa được hay không.

### KnowledgeBaseAgent

**`answer` — approach:**  
Agent retrieve top-k chunks từ store, nối nội dung thành context rồi tạo prompt gồm context và question. LLM chỉ trả lời dựa trên context đã retrieve để giảm hallucination.

### Test Results

```text
pytest tests/ -v
collected 42 items
42 passed
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

Kết quả dưới đây dùng `MockEmbedder`, nên score deterministic nhưng không phản ánh semantic similarity thật như Gemini/OpenAI embeddings.

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|------------|------------|---------|--------------|-------|
| 1 | Sinh viên bị cảnh báo học vụ khi điểm trung bình dưới quy định. | Kết quả học tập thấp sẽ dẫn đến cảnh báo từ nhà trường. | high | 0.1341 | Một phần |
| 2 | Học phí tính theo số tín chỉ đăng ký trong học kỳ. | Tiền học phải đóng phụ thuộc vào số môn học đã chọn. | high | 0.0369 | Không rõ |
| 3 | Sinh viên bị buộc thôi học nếu bị cảnh báo nhiều lần. | Sinh viên xuất sắc đạt GPA trên 3.6 được học bổng. | low | -0.1245 | Có |
| 4 | Điều kiện tốt nghiệp là tích lũy đủ tín chỉ và GPA từ 2.0 trở lên. | Sinh viên phải hoàn thành đồ án tốt nghiệp để được ra trường. | high | -0.0163 | Không |
| 5 | Sinh viên thi hộ bị đình chỉ học tập một năm. | Vi phạm thi cử lần thứ hai sẽ bị buộc thôi học. | high | 0.0443 | Một phần |

**Kết quả nào bất ngờ nhất? Điều này nói gì về embeddings?**  
Pair 4 bất ngờ nhất vì hai câu đều liên quan tốt nghiệp nhưng score lại âm. Điều này cho thấy `MockEmbedder` chỉ phù hợp để test interface deterministic, không phù hợp để đánh giá ngữ nghĩa. Khi đánh giá retrieval thật, nhóm dùng Gemini embedding để có semantic signal tốt hơn.

---

## 6. Results — Cá nhân (10 điểm)

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Sinh viên bị cảnh báo học vụ khi nào? | Khi tín chỉ không đạt vượt 50% khối lượng đăng ký, hoặc nợ quá 24 tín chỉ; GPA học kỳ dưới ngưỡng; hoặc GPA tích lũy dưới ngưỡng theo năm học. |
| 2 | Điều kiện để được xét tốt nghiệp là gì? | Tích lũy đủ học phần/tín chỉ, hoàn thành chuẩn đầu ra, GPA toàn khóa từ 2.0 trở lên, không bị truy cứu hình sự hoặc đình chỉ, và có đơn nếu tốt nghiệp sớm/muộn. |
| 3 | Số tín chỉ tối đa có thể đăng ký trong một học kỳ chính là bao nhiêu? | Cử nhân tối đa 26 tín chỉ; kỹ sư tối đa 30 tín chỉ; kỹ sư tài năng tối đa 33 tín chỉ. Học kỳ hè tối đa 8 tín chỉ. |
| 4 | Sinh viên thi hộ bị xử lý kỷ luật như thế nào? | Lần thứ nhất đình chỉ học tập 1 năm; lần thứ hai buộc thôi học; áp dụng cho cả người thi hộ và người nhờ thi hộ. |
| 5 | Hạng tốt nghiệp bị giảm một mức trong trường hợp nào? | Khi khối lượng học lại vượt 5% tổng tín chỉ chương trình hoặc sinh viên từng bị kỷ luật từ mức cảnh cáo trở lên. |

### Kết Quả Của Tôi — Quang

**Strategy:** `MetadataAwareChunker(max_chars=900) + Gemini embedding + metadata filter`.

| # | Query | Metadata filter | Top-1 Retrieved | Score | Rank | Relevant? |
|---|-------|-----------------|-----------------|-------|------|-----------|
| 1 | Sinh viên bị cảnh báo học vụ khi nào? | category=assessment | danh_gia_ket_qua_hoc_tap | 0.6789 | 1 | Có |
| 2 | Điều kiện để được xét tốt nghiệp là gì? | category=graduation | tot_nghiep_va_bang_cap | 0.6299 | 1 | Có |
| 3 | Số tín chỉ tối đa có thể đăng ký trong một học kỳ chính là bao nhiêu? | category=academics | dang_ky_hoc_phan | 0.6982 | 1 | Có |
| 4 | Sinh viên thi hộ bị xử lý kỷ luật như thế nào? | category=discipline | ky_luat_sinh_vien | 0.6643 | 1 | Có |
| 5 | Hạng tốt nghiệp bị giảm một mức trong trường hợp nào? | category=graduation | tot_nghiep_va_bang_cap | 0.5483 | 1 | Có |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5 = 100%  
**Bao nhiêu queries có Top-1 relevant?** 5 / 5 = 100%  
**Số chunks indexed:** 33 chunks  
**Avg Score Q1-Q5:** 0.644

**Nhận xét:**  
Strategy của tôi retrieve đúng expected file ở rank 1 cho cả 5 queries. Điểm mạnh là metadata filter giúp thu hẹp không gian search trước khi rank vector, nhất là với query về tốt nghiệp và kỷ luật. Query khó nhất là Q5 vì score thấp nhất (0.5483): thông tin "hạng tốt nghiệp bị giảm một mức" nằm trong phần bảng/điều kiện, cần context tốt và metadata chi tiết hơn như `section_title=hang_tot_nghiep`.

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**  
Tôi thấy RecursiveChunker(300) của Thu Linh phù hợp nhất với domain pháp luật về metric vì chunk nhỏ bám đúng cấu trúc Điều/Khoản và đạt score cao nhất. Từ Đức, tôi học được lợi ích của việc giữ article context để không cắt giữa một quy định. Từ Bút, tôi thấy semantic chunking rất tốt về chất lượng ngữ nghĩa nhưng chi phí chạy cao. Từ Hải An, tôi thấy fixed-size dễ kiểm soát kích thước nhưng không tôn trọng ranh giới quy định.

**Điều hay nhất tôi học được từ nhóm khác hoặc qua demo:**  
Metadata không chỉ là thông tin phụ; nếu thiết kế tốt, metadata trở thành một lớp retrieval control. Với tài liệu quy chế, filter theo `category` hoặc `article_number` giúp hệ thống trả lời ổn định hơn và giải thích nguồn dễ hơn.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**  
Tôi sẽ giữ ý tưởng metadata-aware retrieval nhưng cần làm mịn schema hơn. Cụ thể, mỗi chunk nên có `article_number`, `section_title`, `has_table`, `policy_action` và `target_user`. Tôi cũng sẽ dùng filter kết hợp nhiều field, ví dụ `category=graduation` + `section_title=hang_tot_nghiep`, thay vì chỉ filter category-level. Nếu kết hợp metadata filter với chunk nhỏ theo Điều/Khoản, score của strategy có thể cải thiện gần với nhóm top hơn.

### Failure Analysis (Ex 3.5)

**Query khó nhất:** Q5 "Hạng tốt nghiệp bị giảm một mức trong trường hợp nào?".

**Lý do:**
- Score Q5 của tôi thấp nhất (0.5483), và avg score cả strategy cũng thấp nhất trong nhóm (0.644).
- Filter `category=graduation` đưa query vào đúng file, nhưng chưa đủ mịn để nhảy thẳng tới mục `Hạng Tốt Nghiệp`.
- Metadata idea đúng với domain pháp luật, nhưng filter hiện tại chưa đủ chi tiết nên không khai thác hết lợi thế của `article_number` và `section_title`.
- Nếu dùng thêm `section_title=hang_tot_nghiep` hoặc xử lý bảng Markdown riêng, retrieval có thể tốt hơn.

**Đề xuất cải thiện:**
- Thêm `section_title` cho mọi chunk và bắt buộc gán đúng cho các bảng/mục quan trọng.
- Tách bảng Markdown thành chunk riêng.
- Hỗ trợ filter nhiều metadata field thay vì chỉ filter theo `category`.
- Kết hợp MetadataAwareChunker với chunk nhỏ hơn theo Điều/Khoản để tăng score nhưng vẫn giữ khả năng trace metadata.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 9 / 10 |
| Strategy design | Nhóm | 13 / 15 |
| My approach | Cá nhân | 9 / 10 |
| Similarity predictions | Cá nhân | 4 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 4 / 5 |
| **Tổng** | | **84 / 100** |
