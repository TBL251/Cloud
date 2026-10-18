# KẾ HOẠCH TỔNG THỂ ĐỒ ÁN (MASTER PROJECT PLAN)
**Đề tài:** Phân loại ảnh bằng Transfer Learning trên SageMaker (Nhóm: AI/ML)  
**Môn học:** Cloud - Đợt 1 - 2026-2027  
**Giảng viên hướng dẫn (GVHD):** Huỳnh Xuân Phụng  
**Sinh viên thực hiện:** Lê Bá Tâm (MSSV: 24110055 - Lớp: CLCO332779_05UTExMC)  
**Hình thức:** Cá nhân (1 sinh viên)  
**Môi trường triển khai:** AWS Academy Learner Lab (`us-east-1`, IAM Role: `LabRole`)

---

## I. KIỂM TRA & RÀ SOÁT CÔNG VIỆC TUẦN 1

### 1. Yêu cầu của GVHD trong Tuần 1:
* Tìm hiểu dịch vụ AWS (SageMaker, S3, IAM LabRole).
* Thiết kế kiến trúc, vẽ sơ đồ hệ thống.
* Dựng nền tảng trên AWS (S3, VPC, LabRole), cấu hình gắn tag `Project`/`Owner`.
* Chuẩn bị tập dữ liệu ảnh (lọc $\le$ 2.000 ảnh và đưa lên S3).

### 2. Thực tế Tuần 1 sinh viên đã hoàn thành:
* [x] Tìm hiểu, nghiên cứu về các công cụ và thuật toán chuẩn bị thực hiện.
* [x] Xác định chủ đề thực hiện (Phân loại bệnh trên lá cây trồng).
* [x] Phân chia công việc cụ thể các tuần.
* [x] Tạo cấu trúc thư mục đồ án và file `README.md`.

### 3. Các công việc Tuần 1 CÒN NỢ (CHƯA XONG) cần chuyển dồn sang Tuần 2:
* ⚠️ **Chưa thiết kế kiến trúc và chưa vẽ sơ đồ hệ thống (`architecture.png`).**
* ⚠️ **Chưa tạo S3 Bucket thực tế trên AWS và chưa cấu hình gắn thẻ Tag (`Project`, `Owner`).**
* ⚠️ **Chưa thu thập, tiền xử lý và chưa upload tập dữ liệu ($\le$ 2.000 ảnh) lên S3.**
* ⚠️ **Chưa có ảnh chụp minh chứng trên AWS Console (kèm Account ID, Region, thời gian).**

---

## II. PHƯƠNG ÁN PHÙ HỢP NHẤT ĐỂ THỰC HIỆN ĐỒ ÁN (OPTIMAL BLUEPRINT)

| Thành phần | Phương án lựa chọn tối ưu | Lý do kỹ thuật & Ràng buộc Learner Lab |
| :--- | :--- | :--- |
| **Chủ đề & Dữ liệu** | **PlantVillage Leaf Disease** (Lá ngô: Khỏe mạnh, Bệnh gỉ sắt, Đốm lá xám - quy mô **~1.500 ảnh**) | Đúng gợi ý đề bài; kích thước $\le$ 2.000 ảnh; dữ liệu sạch, cân bằng, dễ phân loại. |
| **Mô hình AI** | **MobileNetV2** (PyTorch Transfer Learning với ImageNet weights) | Trọng lượng siêu nhẹ (~3.5M params); huấn luyện trên CPU chỉ mất **6 - 10 phút** cho 8-10 epochs, đạt độ chính xác > 92%. |
| **Kiến trúc Huấn luyện** | **SageMaker Script Mode** (`PyTorch Estimator` chạy `src/train.py`) | Chuẩn MLOps AWS; máy huấn luyện chạy độc lập và **tự động giải phóng ngay sau khi hoàn thành**, không lo quên tắt máy gây hao hụt credit $100. |
| **Cấu hình Instance** | • Notebook Controller: **`ml.t3.medium`** (~$0.05/h)<br>• Máy train: **`ml.c5.xlarge`** (Compute-Optimized, 4 vCPU, ~$0.204/h) | Tối ưu hóa tính toán ma trận CPU (Intel AVX-512), nằm hoàn toàn trong hạn mức 32 vCPU của Learner Lab. |
| **Chiến lược Suy luận** | **Batch Transform (Chính)** kết hợp **Transient Endpoint (Phụ cho Demo)** | Batch Transform tự động tắt máy khi dự đoán xong (chi phí tối thiểu). Endpoint dựng tạm 15 phút để test & quay video, sau đó xóa ngay. |
| **Quyền & Bảo mật** | Dùng trực tiếp **`LabRole`** | Learner Lab không cho phép tạo role mới; cấu hình ARN `arn:aws:iam::<Account_ID>:role/LabRole` đảm bảo không bị `AccessDenied`. |
| **Tagging quy chuẩn** | `Project: PlantLeafClassification`<br>`Owner: LeBaTam` | Tuân thủ yêu cầu đề bài; gắn trên S3, SageMaker Notebook, Training Job. |

---

## III. KẾ HOẠCH ĐIỀU CHỈNH CHI TIẾT (TUẦN 2, 3, 4)

```
[Tuần 1: Xong]  --> Nghiên cứu công cụ, chọn chủ đề, lập kế hoạch, tạo cấu trúc repo.
       │
[Tuần 2: Bù + Lõi] --> Vẽ sơ đồ kiến trúc + Tạo S3/Tags + Chuẩn bị <= 2000 ảnh + Train MobileNetV2 trên ml.c5.xlarge.
       │
[Tuần 3: Tích hợp] --> Batch Transform (Confusion Matrix) + Test Endpoint + Giám sát CloudWatch + Script Teardown.
       │
[Tuần 4: Tổng kết] --> Kiểm thử E2E + Báo cáo chi phí Billing + Viết Báo cáo PDF + Quay Video Demo <= 10 phút.
```

### TUẦN 2: BÙ NỀN TẢNG TUẦN 1 & TRIỂN KHAI HUẤN LUYỆN LÕI TRÊN AWS
* **Công việc bù từ Tuần 1:**
  1. Thiết kế kiến trúc hệ thống và vẽ sơ đồ tổng thể `architecture/architecture.png`.
  2. Tạo S3 Bucket tại Region `us-east-1`, cấu hình gán tag `Project: PlantLeafClassification` và `Owner: LeBaTam`.
  3. Thu thập tập dữ liệu PlantVillage (~1.500 ảnh $\le$ 2.000 ảnh), tiền xử lý resize 224x224, chia tập `train/` (80%) và `val/` (20%), upload lên S3.
* **Công việc lõi của Tuần 2:**
  4. Khởi tạo SageMaker Notebook Instance (`ml.t3.medium`), gán quyền `LabRole`.
  5. Viết mã nguồn huấn luyện Transfer Learning `src/train.py` (MobileNetV2 Script Mode).
  6. Kích hoạt SageMaker Training Job trên instance CPU `ml.c5.xlarge`, theo dõi Loss/Accuracy qua log, xác nhận model artifact `model.tar.gz` lưu trên S3; dừng Notebook sau phiên làm việc.
* **Minh chứng bàn giao:**
  * Sơ đồ kiến trúc `architecture.png`.
  * Ảnh chụp S3 Console chứa dữ liệu (kèm Account ID, Region `us-east-1`, Tagging).
  * Ảnh chụp SageMaker Training Job trạng thái `Completed`.
  * Ảnh chụp file `model.tar.gz` trên S3.

---

### TUẦN 3: TRIỂN KHAI SUY LUẬN, ĐÁNH GIÁ, GIÁM SÁT & TỰ ĐỘNG HÓA
* **Công việc cụ thể:**
  1. Cấu hình Amazon CloudWatch Logs & Metrics theo dõi CPUUtilization, MemoryUtilization của job; thiết lập CloudWatch Billing Alarm cảnh báo chi phí.
  2. Triển khai SageMaker Batch Transform Job trên `ml.c5.xlarge` để dự đoán hàng loạt cho tập ảnh kiểm thử (Test set); tự động giải phóng máy khi xong.
  3. Thu thập kết quả dự đoán, viết script đánh giá tính Accuracy, Precision, Recall, F1-Score và xuất biểu đồ **Confusion Matrix Heatmap**.
  4. Viết custom handler `src/inference.py`, triển khai Real-time Endpoint thử nghiệm (`ml.m5.large`), test gửi ảnh dự đoán trực tiếp và **chạy lệnh xóa ngay Endpoint** để tránh phát sinh chi phí.
  5. Xây dựng script tự động hóa kiểm tra và dọn dẹp tài nguyên `scripts/teardown.py`.
* **Minh chứng bàn giao:**
  * Ảnh chụp CloudWatch Dashboard & Billing Alarm.
  * Ảnh chụp Batch Transform Job status `Completed`.
  * Biểu đồ Confusion Matrix Heatmap (`confusion_matrix.png`) và bảng chỉ số Accuracy.
  * Ảnh chụp Endpoint lúc `InService`, ảnh kết quả test và ảnh lệnh xóa Endpoint thành công.
  * Mã nguồn `scripts/teardown.py`.

---

### TUẦN 4: KIỂM THỬ TỔNG THỂ, TỐI ƯU CHI PHÍ, BÁO CÁO & VIDEO DEMO
* **Công việc cụ thể:**
  1. Kiểm thử toàn diện toàn bộ luồng hệ thống E2E; thực thi script `teardown.py` dọn dẹp sạch sẽ 100% tài nguyên tính phí (Endpoint Config, Model, S3 bucket/object, dừng Notebook).
  2. Truy cập AWS Billing & Cost Management / Cost Explorer, tổng hợp và phân tích chi phí từng dịch vụ đã sử dụng trong suốt 4 tuần (dự kiến < $3 / $100 credit).
  3. Soạn thảo Báo cáo tổng kết chính thức (Final Report PDF) đầy đủ các nội dung: đặt vấn đề, sơ đồ kiến trúc, cấu hình dịch vụ, giải thuật Transfer Learning, kết quả thực nghiệm (Accuracy, Confusion Matrix) và phần kê khai AI hỗ trợ.
  4. Quay và biên tập Video Demo sản phẩm (thời lượng $\le$ 10 phút): walkthrough kiến trúc, demo Training Job, demo suy luận/Batch Transform, show billing và chuẩn bị nội dung vấn đáp với GVHD Huỳnh Xuân Phụng.
* **Minh chứng bàn giao:**
  * Báo cáo chi phí AWS Billing (có Account ID, Region).
  * File Báo cáo cuối kỳ hoàn chỉnh (PDF).
  * Link Video Demo ($\le$ 10 phút) có đầy đủ các phân cảnh theo quy định.

---

## IV. BẢNG DỰ TOÁN CHI PHÍ TRÊN AWS LEARNER LAB ($100 CREDIT)

| Tài nguyên / Dịch vụ | Cấu hình & Thời gian sử dụng | Chi phí dự kiến |
| :--- | :--- | :--- |
| **SageMaker Notebook** | `ml.t3.medium` (~15 giờ bật/tắt trong 4 tuần) | ~$0.75 |
| **SageMaker Training Job** | `ml.c5.xlarge` (~6 lần train thử nghiệm = 1.0 giờ) | ~$0.20 |
| **Batch Transform Job** | `ml.c5.xlarge` (~2 lần chạy = 0.2 giờ) | ~$0.05 |
| **Real-time Endpoint** | `ml.m5.large` (Bật test & quay video 30 phút rồi xóa) | ~$0.06 |
| **S3 Storage & CloudWatch** | Dataset < 500 MB + CloudWatch Free Tier | ~$0.02 |
| **TỔNG CỘNG** | **Toàn bộ quá trình thực hiện 4 tuần** | **~$1.08 - $2.50** |

> **Cam kết an toàn:** Tổng chi phí sử dụng chỉ chiếm **< 3%** ngân sách $100 của Learner Lab. Không có nguy cơ cạn credit nếu tuân thủ việc tắt notebook và xóa endpoint.
