# BÁO CÁO TIẾN ĐỘ TUẦN 1 (NỘP LẠI)
**Môn học:** Cloud - Đợt 1 - 2026-2027  
**Giảng viên hướng dẫn (GVHD):** Huỳnh Xuân Phụng  
**Đề tài:** Phân loại ảnh bằng Transfer Learning trên SageMaker (Nhóm: AI/ML)  

---

## 1. THÔNG TIN THỰC HIỆN & LINK GITHUB PUBLIC
* **Sinh viên thực hiện:** Lê Bá Tâm - MSSV: 24110055 - Lớp: CLCO332779_05UTExMC  
* **Link GitHub Repository (Public):** https://github.com/TBL251/Cloud  
*(Repository ở chế độ Public để GVHD truy cập chấm bài trực tiếp).*

---

## 2. BẢNG PHÂN BỔ CÔNG VIỆC VÀ THỜI LƯỢNG (≥ 8 GIỜ/TUẦN/SV)

| STT | Hạng mục công việc Tuần 1 | Thời lượng | Trạng thái |
| :---: | :--- | :---: | :---: |
| 1 | Khảo sát dịch vụ AWS (SageMaker, S3, IAM LabRole), thuật toán Transfer Learning và các ràng buộc Learner Lab | 2.0 giờ | Hoàn thành |
| 2 | Xác định chủ đề, bộ nhãn lá cây ngô bệnh/khỏe (PlantVillage) và thông số kỹ thuật (input 224x224, $\le$ 2.000 ảnh) | 2.0 giờ | Hoàn thành |
| 3 | Lập kế hoạch chi tiết 4 tuần tối ưu chi phí (< 3$ / 100$ credit) và kịch bản video demo $\le$ 10 phút | 2.5 giờ | Hoàn thành |
| 4 | Xây dựng cấu trúc thư mục chuẩn đồ án MLOps, khởi tạo Git và đẩy mã nguồn lên GitHub Public | 2.0 giờ | Hoàn thành |

* **Tổng thời gian thực hiện Tuần 1:** **8.5 giờ** (Đạt yêu cầu quy định $\ge$ 8 giờ/tuần).

---

## 3. NỘI DUNG CÔNG VIỆC ĐÃ HOÀN THÀNH KÈM LINK FILE TRỰC TIẾP TRÊN GITHUB

Sinh viên đã hoàn thành 100% các đầu việc Tuần 1, toàn bộ mã nguồn và tài liệu đã được đẩy lên GitHub Public:

1. **Khảo sát & Lập Kế hoạch tổng thể 4 tuần:**
   * Nghiên cứu kiến trúc Transfer Learning trên CPU instance (`ml.c5.xlarge`), phương án phục vụ (Batch Transform + Transient Endpoint) và dự toán chi phí chi tiết:  
     👉 Link file: [docs/PROJECT_PLAN_RECOMMENDED.md](https://github.com/TBL251/Cloud/blob/main/docs/PROJECT_PLAN_RECOMMENDED.md)

2. **Xác định chủ đề & Định nghĩa nhãn dữ liệu:**
   * Xác định bộ 4 nhãn phân loại lá ngô (Khỏe mạnh, Gỉ sắt, Đốm lá xám, Cháy lá lớn) kèm thông số chuẩn hóa ImageNet:  
     👉 Link file: [data/class_labels.json](https://github.com/TBL251/Cloud/blob/main/data/class_labels.json)

3. **Thiết kế kiến trúc hệ thống AWS & Luồng dữ liệu:**
   * Soạn thảo tài liệu kiến trúc, sơ đồ luồng dữ liệu Mermaid từ S3 đến SageMaker Script Mode:  
     👉 Link file: [architecture/README.md](https://github.com/TBL251/Cloud/blob/main/architecture/README.md)

4. **Xây dựng cấu trúc đồ án & Hướng dẫn tổng quan:**
   * Khởi tạo cấu trúc repository chuẩn MLOps (`data/`, `src/`, `notebooks/`, `scripts/`, `docs/`, `architecture/`):  
     👉 Link file: [README.md](https://github.com/TBL251/Cloud/blob/main/README.md)

5. **Nhật ký công việc Tuần 1 (Worklog Week 1):**
   * Ghi nhận danh mục công việc Tuần 1:  
     👉 Link file: [docs/worklogs/week_1_worklog.md](https://github.com/TBL251/Cloud/blob/main/docs/worklogs/week_1_worklog.md)

6. **Điều chỉnh tiến độ cho Tuần 2, 3, 4:**
   * Lập sẵn nhật ký công việc điều chỉnh cho các tuần tiếp theo:  
     👉 Link file Tuần 2: [docs/worklogs/week_2_worklog.md](https://github.com/TBL251/Cloud/blob/main/docs/worklogs/week_2_worklog.md)  
     👉 Link file Tuần 3: [docs/worklogs/week_3_worklog.md](https://github.com/TBL251/Cloud/blob/main/docs/worklogs/week_3_worklog.md)  
     👉 Link file Tuần 4: [docs/worklogs/week_4_worklog.md](https://github.com/TBL251/Cloud/blob/main/docs/worklogs/week_4_worklog.md)

---

## 4. LỊCH SỬ COMMIT TRÊN GITHUB PHẢN ÁNH TIẾN ĐỘ CÔNG VIỆC

Sinh viên thực hiện phân chia công việc và commit mã nguồn theo từng giai đoạn rõ ràng:

* `khoi tao cau truc thu muc du an va file readme`  
  👉 Link commit: https://github.com/TBL251/Cloud/commit/93816b0
* `dinh nghia cac lop benh la cay va thong so chuan hoa`  
  👉 Link commit: https://github.com/TBL251/Cloud/commit/cebcf67
* `thiet ke kien truc he thong sagemaker va ve so do luong du lieu`  
  👉 Link commit: https://github.com/TBL251/Cloud/commit/aa636aa
* `len ke hoach trien khai 4 tuan va du toan chi phi`  
  👉 Link commit: https://github.com/TBL251/Cloud/commit/12d8534
* `them nhat ky cong viec cac tuan tu tuan 1 den tuan 4`  
  👉 Link commit: https://github.com/TBL251/Cloud/commit/0cb3446
* `viet script tien xu ly anh va script kiem tra quota aws`  
  👉 Link commit: https://github.com/TBL251/Cloud/commit/d03fafc
* `viet script train.py transfer learning mobilenetv2 chay tren sagemaker cpu`  
  👉 Link commit: https://github.com/TBL251/Cloud/commit/1a26b7a

---

## 5. KHAI BÁO SỬ DỤNG AI (AI DECLARATION) & MINH CHỨNG CHỈNH SỬA

Tuân thủ nghiêm ngặt quy chế đề bài về việc minh bạch khi có AI hỗ trợ:

1. **Công cụ AI sử dụng:** ChatGPT / Claude (tra cứu tài liệu kỹ thuật).
2. **Nội dung có AI hỗ trợ:**
   * Tra cứu tài liệu và gợi ý cú pháp Mermaid diagram để vẽ sơ đồ luồng dữ liệu kiến trúc AWS.
   * Tra cứu tài liệu cú pháp AWS SDK (boto3) khi viết script tương tác với S3.
3. **Minh chứng sinh viên tự can thiệp, chỉnh sửa và làm chủ:**
   * Sinh viên tự nghiên cứu tài liệu môn học, tự tay thiết lập cấu trúc đồ án và điều chỉnh các thông số phù hợp với ràng buộc của AWS Learner Lab (chọn mô hình MobileNetV2 chạy trên CPU, giới hạn dataset dưới 2.000 ảnh, cấu hình LabRole).
   * Sinh viên tự viết và tinh chỉnh các file cấu hình, tài liệu kế hoạch và mã nguồn qua từng commit riêng biệt.
   * Minh chứng qua lịch sử chỉnh sửa mã nguồn: 👉 [Xem Commit Diff tại 12d8534](https://github.com/TBL251/Cloud/commit/12d8534)
4. **Cam kết của sinh viên:**
   * Sinh viên hiểu rõ 100% cấu hình, tài liệu và mã nguồn đã cam kết trên GitHub; tự tin giải thích và phản biện trực tiếp trước GVHD Huỳnh Xuân Phụng khi vấn đáp.

---
*(Báo cáo được nộp trực tiếp dưới dạng văn bản trong nội dung nộp bài — Tuyệt đối không sử dụng tệp đính kèm .docx).*
