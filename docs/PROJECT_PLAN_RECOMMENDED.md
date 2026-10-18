# KẾ HOẠCH TỔNG THỂ ĐỒ ÁN (MASTER PROJECT PLAN)
**Đề tài:** Phân loại ảnh bằng Transfer Learning trên SageMaker (Nhóm: AI/ML)  
**Môn học:** Cloud - Đợt 1 - 2026-2027  
**Giảng viên hướng dẫn (GVHD):** Huỳnh Xuân Phụng  
**Sinh viên thực hiện:** Lê Bá Tâm (MSSV: 24110055 - Lớp: CLCO332779_05UTExMC)  
**Hình thức:** Cá nhân (1 sinh viên)  
**Môi trường triển khai:** AWS Academy Learner Lab (`us-east-1`, IAM Role: `LabRole`)

---

## I. PHƯƠNG ÁN KỸ THUẬT PHÙ HỢP NHẤT (OPTIMAL BLUEPRINT)

| Thành phần | Phương án lựa chọn tối ưu | Lý do kỹ thuật & Ràng buộc Learner Lab |
| :--- | :--- | :--- |
| **Chủ đề & Dữ liệu** | **PlantVillage Leaf Disease** (Lá ngô: Khỏe mạnh, Bệnh gỉ sắt, Đốm lá xám - quy mô **~1.500 ảnh**) | Đúng gợi ý đề bài; kích thước $\le$ 2.000 ảnh; dữ liệu sạch, cân bằng, dễ phân loại. |
| **Mô hình AI** | **MobileNetV2** (PyTorch Transfer Learning với ImageNet weights) | Trọng lượng siêu nhẹ (~3.5M params); huấn luyện trên CPU chỉ mất **6 - 10 phút** cho 8-10 epochs, đạt độ chính xác > 92%. |
| **Kiến trúc Huấn luyện** | **SageMaker Script Mode** (`PyTorch Estimator` chạy `src/train.py`) | Chuẩn MLOps AWS; máy huấn luyện chạy độc lập và **tự động giải phóng ngay sau khi hoàn thành**, không lo quên tắt máy gây vượt ngưỡng chi phí tối đa $50. |
| **Cấu hình Instance** | • Notebook Controller: **`ml.t3.medium`** (~$0.05/h)<br>• Máy train: **`ml.c5.xlarge`** (Compute-Optimized, 4 vCPU, ~$0.204/h) | Tối ưu hóa tính toán ma trận CPU (Intel AVX-512), nằm hoàn toàn trong hạn mức 32 vCPU của Learner Lab. |
| **Chiến lược Suy luận** | **Batch Transform (Chính)** kết hợp **Transient Endpoint (Phụ cho Demo)** | Batch Transform tự động tắt máy khi dự đoán xong (chi phí tối thiểu). Endpoint dựng tạm 15 phút để test & quay video, sau đó xóa ngay. |
| **Quyền & Bảo mật** | Dùng trực tiếp **`LabRole`** | Learner Lab không cho phép tạo role mới; cấu hình ARN `arn:aws:iam::<Account_ID>:role/LabRole` đảm bảo không bị `AccessDenied`. |
| **Tagging quy chuẩn** | `Project: PlantLeafClassification`<br>`Owner: LeBaTam` | Tuân thủ yêu cầu đề bài; gắn trên S3, SageMaker Notebook, Training Job. |

---

## II. KẾ HOẠCH TRIỂN KHAI 4 TUẦN

### TUẦN 1: TÌM HIỂU DỊCH VỤ, THIẾT KẾ KIẾN TRÚC & KHỞI TẠO DỰ ÁN
* Tìm hiểu các dịch vụ AWS: Amazon SageMaker, Amazon S3, IAM LabRole và các ràng buộc của AWS Learner Lab.
* Nghiên cứu giải thuật Transfer Learning và lựa chọn mô hình MobileNetV2 tối ưu cho CPU instance.
* Xác định chủ đề thực hiện: Phân loại bệnh trên lá cây trồng (PlantVillage Corn Leaf Disease).
* Lập kế hoạch triển khai 4 tuần và bảng dự toán chi phí.
* Khởi tạo cấu trúc thư mục đồ án chuẩn MLOps và file `README.md`.

---

### TUẦN 2: DỰNG NỀN TẢNG AWS & TRIỂN KHAI HUẤN LUYỆN LÕI
* Thiết kế kiến trúc hệ thống và vẽ sơ đồ tổng thể (`architecture/README.md`).
* Tạo Amazon S3 Bucket tại Region `us-east-1`, cấu hình gắn thẻ Tag quy chuẩn (`Project: PlantLeafClassification`, `Owner: LeBaTam`).
* Thu thập tập dữ liệu PlantVillage lá cây bệnh/khỏe, lọc chuẩn hóa quy mô ~1.500 ảnh (dưới 2.000 ảnh, resize 224x224), chia tập Train/Val (80/20) và upload lên S3.
* Khởi tạo SageMaker Notebook Instance (`ml.t3.medium`), gán quyền IAM `LabRole`, thiết lập môi trường PyTorch.
* Viết mã nguồn huấn luyện Transfer Learning `src/train.py` (Script Mode với MobileNetV2).
* Cấu hình PyTorch Estimator SDK, submit Training Job lên instance CPU `ml.c5.xlarge`, lưu model artifact `model.tar.gz` trên S3; dừng Notebook instance sau phiên làm việc.

---

### TUẦN 3: SUY LUẬN, ĐÁNH GIÁ, GIÁM SÁT & TỰ ĐỘNG HÓA
* Cấu hình Amazon CloudWatch Logs & Metrics theo dõi CPUUtilization, MemoryUtilization trong suốt quá trình huấn luyện; thiết lập CloudWatch Billing Alarm cảnh báo chi phí.
* Triển khai SageMaker Batch Transform Job trên máy ảo `ml.c5.xlarge` để dự đoán hàng loạt cho tập ảnh kiểm thử (Test set); tự động giải phóng máy khi hoàn thành.
* Thu thập kết quả dự đoán, viết script tính toán các chỉ số (Accuracy, Precision, Recall, F1-Score) và xuất biểu đồ ma trận nhầm lẫn (Confusion Matrix Heatmap).
* Viết custom handler `src/inference.py`, deploy Real-time Endpoint thử nghiệm (`ml.m5.large`), test gửi ảnh dự đoán trực tiếp và chạy lệnh xóa ngay Endpoint để tránh phát sinh chi phí.
* Xây dựng script tự động hóa kiểm tra và dọn dẹp tài nguyên `scripts/teardown.py` để phòng ngừa rò rỉ chi phí trên Learner Lab.

---

### TUẦN 4: KIỂM THỬ TỔNG THỂ, TỐI ƯU CHI PHÍ, BÁO CÁO & VIDEO DEMO
* Kiểm thử toàn diện toàn bộ luồng hệ thống E2E; thực thi script teardown dọn dẹp sạch sẽ 100% tài nguyên tính phí (Endpoint Config, Model, S3 bucket/object, dừng Notebook).
* Truy cập AWS Billing & Cost Management / Cost Explorer, tổng hợp và phân tích chi phí từng dịch vụ đã sử dụng trong suốt 4 tuần (dự kiến < $3, tối ưu vượt trội so với hạn mức tối đa $50).
* Soạn thảo Báo cáo tổng kết chính thức của đồ án (Final Report PDF): Đặt vấn đề, Sơ đồ kiến trúc, Cấu hình dịch vụ, Huấn luyện Transfer Learning, Đánh giá Confusion Matrix, Kê khai AI.
* Quay và biên tập Video Demo sản phẩm (thời lượng $\le$ 10 phút): walkthrough kiến trúc, train job, batch transform/endpoint, show billing và chuẩn bị nội dung vấn đáp với GVHD Huỳnh Xuân Phụng.

---

## III. BẢNG DỰ TOÁN CHI PHÍ TRÊN AWS (HẠN MỨC TỐI ĐA $50)

| Tài nguyên / Dịch vụ | Cấu hình & Thời gian sử dụng | Chi phí dự kiến |
| :--- | :--- | :--- |
| **SageMaker Notebook** | `ml.t3.medium` (~15 giờ bật/tắt trong 4 tuần) | ~$0.75 |
| **SageMaker Training Job** | `ml.c5.xlarge` (~6 lần train thử nghiệm = 1.0 giờ) | ~$0.20 |
| **Batch Transform Job** | `ml.c5.xlarge` (~2 lần chạy = 0.2 giờ) | ~$0.05 |
| **Real-time Endpoint** | `ml.m5.large` (Bật test & quay video 30 phút rồi xóa) | ~$0.06 |
| **S3 Storage & CloudWatch** | Dataset < 500 MB + CloudWatch Free Tier | ~$0.02 |
| **TỔNG CỘNG** | **Toàn bộ quá trình thực hiện 4 tuần** | **~$1.08 - $2.50** |
