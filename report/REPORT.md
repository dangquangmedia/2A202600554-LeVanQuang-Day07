# Bao Cao Lab 7: Embedding & Vector Store

**Ho ten:** Le Van Quang  
**Nhom:** B2  
**Ngay:** 05/06/2026

---

## 1. Warm-up (5 diem)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghia la gi?**  
High cosine similarity nghia la hai vector embedding co huong gan nhau trong khong gian vector. Voi van ban, dieu nay thuong cho thay hai cau hoac hai doan co y nghia gan nhau, du cach dung tu co the khac nhau.

**Vi du HIGH similarity:**
- Sentence A: "Sinh vien bi canh bao hoc vu khi diem trung binh duoi quy dinh."
- Sentence B: "Ket qua hoc tap thap co the khien sinh vien bi canh bao."
- Tai sao tuong dong: Ca hai cau deu noi ve canh bao hoc vu do ket qua hoc tap thap.

**Vi du LOW similarity:**
- Sentence A: "Sinh vien bi canh bao hoc vu khi diem trung binh duoi quy dinh."
- Sentence B: "Thu vien mo cua tu 7 gio sang den 9 gio toi."
- Tai sao khac: Hai cau thuoc hai chu de khac nhau, mot cau ve hoc vu va mot cau ve dich vu thu vien.

**Tai sao cosine similarity duoc uu tien hon Euclidean distance cho text embeddings?**  
Cosine similarity tap trung vao huong cua vector thay vi do lon tuyet doi. Dieu nay phu hop voi text embeddings vi ta quan tam hai van ban co cung y nghia hay khong hon la vector dai ngan bao nhieu. Euclidean distance de bi anh huong boi magnitude cua vector, nhat la khi van ban co do dai khac nhau.

### Chunking Math (Ex 1.2)

**Document 10,000 ky tu, chunk_size=500, overlap=50. Bao nhieu chunks?**  
Step size = 500 - 50 = 450.  
So chunks = ceil((10000 - 500) / 450) + 1 = ceil(9500 / 450) + 1 = 23 chunks.

**Neu overlap tang len 100, chunk count thay doi the nao? Tai sao muon overlap nhieu hon?**  
Step size = 500 - 100 = 400.  
So chunks = ceil((10000 - 500) / 400) + 1 = 25 chunks. Overlap lon hon giup giu ngu canh o ranh gioi chunk tot hon, nhung lam tang so chunk, thoi gian embedding va chi phi truy van.

---

## 2. Document Selection - Nhom (10 diem)

### Domain & Ly Do Chon

**Domain:** Quy dinh hoc vu - Quy che dao tao trinh do dai hoc cua Truong Dai hoc Su pham Ky thuat Hung Yen.

**Tai sao nhom chon domain nay?**  
Domain quy dinh hoc vu phu hop voi RAG vi tai lieu co cau truc ro theo chuong, dieu, muc va bang. Nguoi dung co nhu cau hoi dap thuc te nhu dang ky hoc phan, canh bao hoc tap, tot nghiep, ky luat, nghi hoc hoac chuyen nganh. Cac gold answers cung de kiem chung truc tiep tu tai lieu goc.

### Data Inventory

| # | Ten tai lieu | Nguon | So ky tu | Metadata da gan |
|---|--------------|-------|----------|-----------------|
| 1 | quy_che_dao_tao.md | QD 952/QD-DHSPKT 2021, Dieu 1-5 | 3.709 | category=academics, chapter=I, language=vi |
| 2 | dang_ky_hoc_phan.md | QD 952/QD-DHSPKT 2021, Dieu 9-12 | 3.894 | category=academics, chapter=II, language=vi |
| 3 | danh_gia_ket_qua_hoc_tap.md | QD 952/QD-DHSPKT 2021, Dieu 13-16 | 4.634 | category=assessment, chapter=III, language=vi |
| 4 | tot_nghiep_va_bang_cap.md | QD 952/QD-DHSPKT 2021, Dieu 18-20 | 3.670 | category=graduation, chapter=III, language=vi |
| 5 | nghi_hoc_chuyen_nganh.md | QD 952/QD-DHSPKT 2021, Dieu 21-25 | 4.211 | category=transfer, chapter=IV, language=vi |
| 6 | ky_luat_sinh_vien.md | QD 952/QD-DHSPKT 2021, Dieu 26-27 | 3.582 | category=discipline, chapter=IV, language=vi |

### Metadata Schema

| Truong metadata | Kieu | Vi du gia tri | Tai sao huu ich cho retrieval? |
|-----------------|------|---------------|--------------------------------|
| category | string | academics / assessment / graduation / discipline / transfer | Loc theo nhom chu de truoc khi rank similarity |
| chapter | string | I / II / III / IV | Giu lien ket voi cau truc chuong cua quy che |
| article_number | string | 10 / 16 / 20 / 26 | Xac dinh dieu quy che chua thong tin |
| section_title | string | so_luong_tin_chi / hang_tot_nghiep | Phan biet cac muc nho trong cung mot dieu |
| source | string | data/tot_nghiep_va_bang_cap.md | Truy vet ket qua ve tai lieu goc |
| file_id | string | tot_nghiep_va_bang_cap | So sanh top-k voi expected file trong benchmark |
| language | string | vi | Dam bao benchmark chay tren tai lieu tieng Viet |

---

## 3. Strategy Design - Ca nhan chon, nhom so sanh (15 diem)

### Baseline Analysis

Nhom dung `ChunkingStrategyComparator().compare()` de quan sat baseline chunking. Day chi la baseline chung de hieu du lieu. Strategy ca nhan cua toi la **MetadataAwareChunker(max_chars=900) + metadata schema/filtering**.

| Tai lieu | Strategy | Chunk Count | Avg Length | Nhan xet |
|----------|----------|-------------|------------|----------|
| quy_che_dao_tao.md | fixed_size | 19 | 195.21 | De cat ngang cau hoac bullet |
| quy_che_dao_tao.md | by_sentences | 6 | 616.0 | Giu cau nhung chunk dai |
| quy_che_dao_tao.md | paragraph-based | 29 | 126.28 | Ton trong paragraph tot hon |
| danh_gia_ket_qua_hoc_tap.md | fixed_size | 24 | 193.08 | Co the cat ngang bang diem |
| danh_gia_ket_qua_hoc_tap.md | by_sentences | 4 | 1156.75 | Chunk qua lon, nhieu y lan nhau |
| danh_gia_ket_qua_hoc_tap.md | paragraph-based | 34 | 134.53 | Tach nho, bam doan |
| tot_nghiep_va_bang_cap.md | fixed_size | 19 | 193.16 | De mat context bang hang tot nghiep |
| tot_nghiep_va_bang_cap.md | by_sentences | 5 | 731.6 | It chunk nhung qua rong |
| tot_nghiep_va_bang_cap.md | paragraph-based | 28 | 129.18 | Tot cho paragraph, nhung chua gan metadata dieu/muc |

### Strategy Cua Toi

**Loai:** Metadata-aware chunking strategy.

**Chunking method:** `MetadataAwareChunker` tach tai lieu theo heading `## Dieu ...` truoc. Neu mot Dieu dai hon `max_chars`, chunker tach tiep theo heading `###`. Cach nay giu ngu canh Dieu/Muc tot hon viec cat cung theo so ky tu.

**Tham so:** `max_chars=900`.

**Metadata schema:** moi chunk duoc gan:
- `category`: nhom chu de, vi du `academics`, `assessment`, `graduation`, `discipline`, `transfer`
- `chapter`: chuong trong quy che
- `article_number`: so dieu, vi du `10`, `16`, `20`, `26`
- `section_title`: tieu de dieu hoac muc nho
- `source`, `file_id`, `chunk_index`, `language`

**Tai sao toi chon strategy nay?**  
Tai lieu hoc vu co nhieu tu chung nhu "sinh vien", "hoc ky", "tin chi", "diem". Neu chi search semantic tren toan bo chunks, ket qua co the bi nhieu giua dang ky hoc phan, tot nghiep va canh bao hoc vu. Metadata-aware chunking giup moi chunk vua co noi dung du lon de giu context, vua co nhan de filter dung vung tai lieu truoc khi tinh similarity.

**Code minh hoa:**

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
    "Dieu kien de duoc xet tot nghiep la gi?",
    top_k=3,
    metadata_filter={"category": "graduation"},
)
```

### So Sanh Voi Thanh Vien Khac

Benchmark chay tren cung 5 queries va cung bo 6 tai lieu. Ket qua cua Quang duoc chay lai bang `data/group/benchmark_quang.py`, Gemini embedding `models/gemini-embedding-001`.

| Thanh vien | Strategy | Chunks (6 files) | Avg Score Q1-Q5 | Precision@3 | Diem manh | Diem yeu |
|------------|----------|------------------|-----------------|-------------|-----------|----------|
| **Quang** | **MetadataAwareChunker(900) + metadata schema/filter** | **33** | **0.644** | **5/5 = 100%** | Gan metadata `category`, `article_number`, `section_title`; filter truoc khi rank; de trace nguon | Filter hien moi dung category-level nen Q5 con score thap |
| Thanh vien 1 | Paragraph-based chunking | 107 | Khong dung lam ket qua ca nhan | 5/5 = 100% | Chunk nho, bam cau truc Dieu/Khoan | Nhieu chunk nho, index cham hon |
| Thanh vien 2 | Semantic Chunking | - | - | 10/10 | Chunk ngu nghia tron y, loc thong minh | Chay cham vi can embed/tinh similarity nhieu cau |
| Thanh vien 3 | ArticleSectionChunker(1400) + VietnameseTextEmbedder | - | - | 10/10 | Giu article context, top-1 expected file tot | Lexical embedder chua hieu paraphrase xa nhu semantic embedding that |
| Baseline | FixedSize(500, overlap=50) | 56 | 0.793 | 5/5 = 100% | Don gian, kich thuoc on dinh | Co the cat ngang bang hoac bullet |

**Chi tiet score theo query cua Quang:**

| Query | Metadata filter | Top-1 Retrieved | Score | Rank |
|-------|-----------------|-----------------|-------|------|
| Q1: Canh bao hoc vu | `category=assessment` | `danh_gia_ket_qua_hoc_tap` | 0.6789 | 1 |
| Q2: Dieu kien tot nghiep | `category=graduation` | `tot_nghiep_va_bang_cap` | 0.6299 | 1 |
| Q3: Tin chi dang ky toi da | `category=academics` | `dang_ky_hoc_phan` | 0.6982 | 1 |
| Q4: Sinh vien thi ho | `category=discipline` | `ky_luat_sinh_vien` | 0.6643 | 1 |
| Q5: Hang tot nghiep giam | `category=graduation` | `tot_nghiep_va_bang_cap` | 0.5483 | 1 |
| **Precision@3** |  |  |  | **100%** |

**Strategy nao tot nhat cho domain nay? Tai sao?**  
Voi domain quy che hoc vu, strategy cua toi phu hop khi uu tien kha nang giai thich va kiem soat retrieval. Metadata-aware chunking khong chi tach van ban ma con gan nhan theo `category`, `article_number`, `section_title`, giup he thong thu hep vung tim kiem truoc khi tinh similarity. Strategy nay dat Precision@3 = 100%, dung it chunk, index nhanh hon va de kiem chung nguon hon trong demo.

---

## 4. My Approach - Ca nhan (10 diem)

### Chunking Functions

**`SentenceChunker.chunk` - approach:**  
Ham dung regex de tach cau theo dau `.`, `!`, `?` ket hop whitespace/newline, loc cau rong roi gom toi da `max_sentences_per_chunk` cau vao moi chunk.

**Paragraph-based chunking implementation - approach:**  
Ham `_split` dung base case khi text da ngan hon `chunk_size`; neu text con dai thi thu tach theo separator uu tien nhu paragraph, newline, cau, khoang trang, roi cuoi cung moi cat theo ky tu.

### EmbeddingStore

**`add_documents` + `search` - approach:**  
`add_documents` luu tung `Document` cung content, metadata va embedding. `search` embed query, tinh similarity voi tung vector da luu, sort giam dan theo score roi tra top-k.

**`search_with_filter` + `delete_document` - approach:**  
`search_with_filter` loc metadata truoc khi tinh similarity, phu hop voi strategy metadata cua toi. `delete_document` xoa document theo id hoac `metadata["doc_id"]`, tra `True/False` tuy co xoa duoc hay khong.

### KnowledgeBaseAgent

**`answer` - approach:**  
Agent retrieve top-k chunks tu store, noi noi dung thanh context roi tao prompt gom context va question. LLM chi tra loi dua tren context da retrieve de giam hallucination.

### Test Results

```text
pytest tests/ -v
collected 42 items
42 passed
```

**So tests pass:** 42 / 42

---

## 5. Similarity Predictions - Ca nhan (5 diem)

Ket qua duoi day dung `MockEmbedder`, nen score deterministic nhung khong phan anh semantic similarity that nhu Gemini/OpenAI embeddings.

| Pair | Sentence A | Sentence B | Du doan | Actual Score | Dung? |
|------|------------|------------|---------|--------------|-------|
| 1 | Sinh vien bi canh bao hoc vu khi diem trung binh duoi quy dinh. | Ket qua hoc tap thap se dan den canh bao tu nha truong. | high | 0.1341 | Mot phan |
| 2 | Hoc phi tinh theo so tin chi dang ky trong hoc ky. | Tien hoc phai dong phu thuoc vao so mon hoc da chon. | high | 0.0369 | Khong ro |
| 3 | Sinh vien bi buoc thoi hoc neu bi canh bao nhieu lan. | Sinh vien xuat sac dat GPA tren 3.6 duoc hoc bong. | low | -0.1245 | Co |
| 4 | Dieu kien tot nghiep la tich luy du tin chi va GPA tu 2.0 tro len. | Sinh vien phai hoan thanh do an tot nghiep de duoc ra truong. | high | -0.0163 | Khong |
| 5 | Sinh vien thi ho bi dinh chi hoc tap mot nam. | Vi pham thi cu lan thu hai se bi buoc thoi hoc. | high | 0.0443 | Mot phan |

**Ket qua nao bat ngo nhat? Dieu nay noi gi ve embeddings?**  
Pair 4 bat ngo nhat vi hai cau deu lien quan tot nghiep nhung score lai am. Dieu nay cho thay `MockEmbedder` chi phu hop de test interface deterministic, khong phu hop de danh gia ngu nghia. Khi danh gia retrieval that, nhom dung Gemini embedding de co semantic signal tot hon.

---

## 6. Results - Ca nhan (10 diem)

### Benchmark Queries & Gold Answers (nhom thong nhat)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Sinh vien bi canh bao hoc vu khi nao? | Khi tin chi khong dat vuot 50% khoi luong dang ky, hoac no qua 24 tin chi; GPA hoc ky duoi nguong; hoac GPA tich luy duoi nguong theo nam hoc. |
| 2 | Dieu kien de duoc xet tot nghiep la gi? | Tich luy du hoc phan/tin chi, hoan thanh chuan dau ra, GPA toan khoa tu 2.0 tro len, khong bi truy cuu hinh su hoac dinh chi, va co don neu tot nghiep som/muon. |
| 3 | So tin chi toi da co the dang ky trong mot hoc ky chinh la bao nhieu? | Cu nhan toi da 26 tin chi; ky su toi da 30 tin chi; ky su tai nang toi da 33 tin chi. Hoc ky he toi da 8 tin chi. |
| 4 | Sinh vien thi ho bi xu ly ky luat nhu the nao? | Lan thu nhat dinh chi hoc tap 1 nam; lan thu hai buoc thoi hoc; ap dung cho ca nguoi thi ho va nguoi nho thi ho. |
| 5 | Hang tot nghiep bi giam mot muc trong truong hop nao? | Khi khoi luong hoc lai vuot 5% tong tin chi chuong trinh hoac sinh vien tung bi ky luat tu muc canh cao tro len. |

### Ket Qua Cua Toi - Quang

**Strategy:** `MetadataAwareChunker(max_chars=900) + Gemini embedding + metadata filter`.

| # | Query | Metadata filter | Top-1 Retrieved | Score | Rank | Relevant? |
|---|-------|-----------------|-----------------|-------|------|-----------|
| 1 | Sinh vien bi canh bao hoc vu khi nao? | category=assessment | danh_gia_ket_qua_hoc_tap | 0.6789 | 1 | Co |
| 2 | Dieu kien de duoc xet tot nghiep la gi? | category=graduation | tot_nghiep_va_bang_cap | 0.6299 | 1 | Co |
| 3 | So tin chi toi da co the dang ky trong mot hoc ky chinh la bao nhieu? | category=academics | dang_ky_hoc_phan | 0.6982 | 1 | Co |
| 4 | Sinh vien thi ho bi xu ly ky luat nhu the nao? | category=discipline | ky_luat_sinh_vien | 0.6643 | 1 | Co |
| 5 | Hang tot nghiep bi giam mot muc trong truong hop nao? | category=graduation | tot_nghiep_va_bang_cap | 0.5483 | 1 | Co |

**Bao nhieu queries tra ve chunk relevant trong top-3?** 5 / 5 = 100%  
**Bao nhieu queries co Top-1 relevant?** 5 / 5 = 100%  
**So chunks indexed:** 33 chunks  
**Avg Score Q1-Q5:** 0.644

**Nhan xet:**  
Strategy cua toi retrieve dung expected file o rank 1 cho ca 5 queries. Diem manh la metadata filter giup thu hep khong gian search truoc khi rank vector, nhat la voi query ve tot nghiep va ky luat. Query kho nhat la Q5 vi score thap nhat (0.5483): thong tin "hang tot nghiep bi giam mot muc" nam trong phan bang/dieu kien, can context tot va metadata chi tiet hon nhu `section_title=hang_tot_nghiep`.

---

## 7. What I Learned (5 diem - Demo)

**Dieu hay nhat toi hoc duoc tu thanh vien khac trong nhom:**  
Toi thay chunk nho theo cau truc paragraph/Dieu co the tang cosine score ro ret, semantic chunking co the tao chunk tron y hon nhung chi phi embedding cao hon, con article-level chunking giu context tot va de kiem chung expected file.

**Dieu hay nhat toi hoc duoc tu nhom khac hoac qua demo:**  
Metadata khong chi la thong tin phu; neu thiet ke tot, metadata tro thanh mot lop retrieval control. Voi tai lieu quy che, filter theo `category` hoac `article_number` giup he thong tra loi on dinh hon va giai thich nguon de hon.

**Neu lam lai, toi se thay doi gi trong data strategy?**  
Toi se gan metadata min hon cho tung chunk: `article_number`, `section_title`, `has_table`, `policy_action` va `target_user`. Toi cung se dung filter ket hop nhieu field neu backend ho tro tot, vi du `category=graduation` + `section_title=hang_tot_nghiep`, de cai thien nhung query nhu Q5.

### Failure Analysis (Ex 3.5)

**Query kho nhat:** Q5 "Hang tot nghiep bi giam mot muc trong truong hop nao?".

**Ly do:**
- Score thap nhat trong 5 queries vi noi dung nam trong phan hang tot nghiep va cac bullet dieu kien.
- Filter `category=graduation` dua query vao dung file, nhung chua du min de nhay thang toi muc `Hang Tot Nghiep`.
- Neu dung them `section_title=hang_tot_nghiep` hoac xu ly bang Markdown rieng, retrieval co the tot hon.

**De xuat cai thien:**
- Them `section_title` cho moi chunk.
- Tach bang Markdown thanh chunk rieng.
- Ho tro filter nhieu metadata field thay vi chi filter theo `category`.

---

## Tu Danh Gia

| Tieu chi | Loai | Diem tu danh gia |
|----------|------|------------------|
| Warm-up | Ca nhan | 5 / 5 |
| Document selection | Nhom | 9 / 10 |
| Strategy design | Nhom | 13 / 15 |
| My approach | Ca nhan | 9 / 10 |
| Similarity predictions | Ca nhan | 4 / 5 |
| Results | Ca nhan | 10 / 10 |
| Core implementation (tests) | Ca nhan | 30 / 30 |
| Demo | Nhom | 4 / 5 |
| **Tong** | | **84 / 100** |
