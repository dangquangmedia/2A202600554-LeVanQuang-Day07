# Group Work — Day 07: Embedding & Vector Store

**Nhóm:** [Tên nhóm]  
**Thành viên:**
- Đoàn Thị Thu Linh (12423020) — RecursiveChunker(300)
- [Thành viên 2] — SentenceChunker(max=3)
- [Thành viên 3] — FixedSizeChunker(500, overlap=50)

**Ngày:** 05/06/2026

---

## Mô tả

Phần bài làm nhóm gồm dataset domain chung và script benchmark so sánh 3 chunking strategies với Gemini embedding.

## Cấu trúc thư mục

```
group/
├── data/
│   ├── quy_che_dao_tao.md           # Điều 1–5, Chương I
│   ├── dang_ky_hoc_phan.md          # Điều 9–12, Chương II
│   ├── danh_gia_ket_qua_hoc_tap.md  # Điều 13–16, Chương III
│   ├── tot_nghiep_va_bang_cap.md    # Điều 18–20, Chương III
│   ├── nghi_hoc_chuyen_nganh.md     # Điều 21–25, Chương IV
│   └── ky_luat_sinh_vien.md         # Điều 26–27, Chương IV
├── benchmark.py                     # So sánh 3 strategies, tính Precision@3
└── README.md
```

## Domain đã chọn

**Quy chế đào tạo trình độ đại học — Trường ĐH Sư phạm Kỹ thuật Hưng Yên**  
Nguồn: QĐ 952/QĐ-ĐHSPKT năm 2021

6 tài liệu, tổng ~23.700 ký tự, bao phủ các chủ đề: đăng ký học phần, đánh giá kết quả, tốt nghiệp, kỷ luật, chuyển ngành.

## 5 Benchmark Queries (nhóm thống nhất)

| # | Query | Expected File | Filter |
|---|-------|---------------|--------|
| 1 | Sinh viên bị cảnh báo học vụ khi nào? | danh_gia_ket_qua_hoc_tap | — |
| 2 | Điều kiện để được xét tốt nghiệp là gì? | tot_nghiep_va_bang_cap | category=graduation |
| 3 | Số tín chỉ tối đa có thể đăng ký trong một học kỳ chính là bao nhiêu? | dang_ky_hoc_phan | — |
| 4 | Sinh viên thi hộ bị xử lý kỷ luật như thế nào? | ky_luat_sinh_vien | category=discipline |
| 5 | Hạng tốt nghiệp bị giảm một mức trong trường hợp nào? | tot_nghiep_va_bang_cap | category=graduation |

## Kết quả Benchmark (Gemini embedding `models/gemini-embedding-001`)

| Strategy | Chunks (6 files) | Precision@3 | Avg Cosine Score |
|----------|-----------------|-------------|-----------------|
| FixedSize(500, overlap=50) | 56 | 100% | 0.793 |
| Sentence(max=3) | 33 | 100% | 0.802 |
| **Recursive(300)** | **107** | **100%** | **0.853** |

### Score chi tiết theo query

| Query | FixedSize(500) | Sentence(3) | Recursive(300) |
|-------|----------------|-------------|----------------|
| Q1: Cảnh báo học vụ | 0.8161 | 0.8189 | **0.8414** |
| Q2: Điều kiện tốt nghiệp | 0.8178 | 0.8170 | **0.8573** |
| Q3: Tín chỉ đăng ký tối đa | 0.8107 | **0.8396** | 0.8400 |
| Q4: Sinh viên thi hộ | 0.7462 | 0.7598 | **0.8807** |
| Q5: Hạng tốt nghiệp giảm | 0.7611 | 0.7548 | **0.7884** |

### Kết luận nhóm

> Cả 3 strategy đều đạt Precision@3 = 100%, cho thấy Gemini embedding rất mạnh cho tiếng Việt. **RecursiveChunker(300)** dẫn đầu về cosine score vì tôn trọng cấu trúc phân cấp của văn bản pháp quy. **SentenceChunker(max=3)** tạo ít chunk nhất (33) — phù hợp khi cần tốc độ index nhanh. **FixedSizeChunker** dễ implement nhất nhưng có thể cắt ngang bảng/danh sách.

## Chạy benchmark

```bash
# Cần GEMINI_API_KEY trong file .env ở thư mục gốc dự án
pip install google-genai python-dotenv

# Chạy từ thư mục gốc dự án (không phải trong group/)
python benchmark.py
```
