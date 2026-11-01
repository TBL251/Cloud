# NHẬT KÝ CÔNG VIỆC TUẦN 4 (WEEK 4 WORKLOG)
**Môn học:** Cloud - Đợt 1 - 2026-2027  
**Sinh viên thực hiện:** Lê Bá Tâm - MSSV: 24110055  
**Hạn mức thời gian:** 9.0 giờ  

---

### Danh mục công việc hoàn thành trong tuần:
1. **Đối soát & Phân tích chi phí AWS Academy (1.5 giờ):**
   - Kiểm tra số dư và chi tiêu thực tế trên Dashboard Learner Lab: Ghi nhận **`Used $18.1 of $50`** (ngân sách còn dư $31.90 USD).
   - Đánh giá hiệu quả kinh tế và mức độ tuân thủ hạn mức ngân sách của toàn bộ dự án.
2. **Xây dựng Bảng đối chuẩn kỹ thuật (Benchmarking) (2.0 giờ):**
   - So sánh toàn diện Batch Transform (`ml.c5.xlarge`, ~$0.013/lượt, Acc 90%) và Real-time Endpoint (`ml.m5.large`, Latency 212.39 ms, ~$65/tháng nếu bật 24/7).
   - Phân tích kịch bản ứng dụng nông nghiệp thực tế: Quét ảnh flycam theo lô (Batch) vs Ứng dụng di động chẩn đoán tại ruộng (Real-time).
3. **Quy trình thu dọn tài nguyên (Final Teardown) (1.0 giờ):**
   - Chạy script `scripts/teardown.py` xác nhận 0 Endpoint, 0 Config, 0 Model.
   - Dừng Notebook Instance `ml.t3.medium` để đóng băng hệ thống an toàn.
4. **Xây dựng kịch bản, quay và biên tập Video Demo (2.5 giờ):**
   - Thực hiện quay video demo thực tế ~9 phút bao quát toàn bộ 5 phần: Kiến trúc & GitHub repo, Chuẩn bị dữ liệu & Huấn luyện MobileNetV2 736s 90%, Đánh giá Batch Transform 400 ảnh & Confusion Matrix, Triển khai Transient Endpoint 212.39ms, và Teardown + Đối soát ngân sách $18.1.
5. **Soạn thảo và chốt Báo cáo Tiến độ Tuần 4 (`docs/BAO_CAO_TUAN_4.md`) (2.0 giờ):**
   - Hoàn thành đầy đủ 5 mục theo chuẩn nhận xét của GVHD Huỳnh Xuân Phụng, văn phong kỹ sư mộc mạc và gắn link minh chứng đầy đủ.
