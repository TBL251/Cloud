# BÁO CÁO TIẾN ĐỘ TUẦN 4
**Môn học:** Cloud - Đợt 1 - 2026-2027  
**Giảng viên hướng dẫn (GVHD):** Huỳnh Xuân Phụng  
**Đề tài:** Phân loại ảnh bằng Transfer Learning trên SageMaker (Nhóm: AI/ML)  

---

## 1. THÔNG TIN THỰC HIỆN, LINK GITHUB & LINK VIDEO DEMO
* **Sinh viên thực hiện:** Lê Bá Tâm - MSSV: 24110055 - Lớp: CLCO332779_05UTExMC  
* **Link GitHub Repository (Public):** https://github.com/TBL251/Cloud  
*(Repository ở chế độ Public để GVHD truy cập và kiểm chứng mã nguồn trực tiếp).*
* **Link Video Demo Đồ Án (Thời lượng ~9 phút):** [https://drive.google.com/file/d/1dhWreqLEgqtqwZTe-_w1gkN2HBoCfJsX/view?usp=sharing](https://drive.google.com/file/d/1dhWreqLEgqtqwZTe-_w1gkN2HBoCfJsX/view?usp=sharing)  
*(Video demo chi tiết toàn bộ quy trình thực nghiệm trên AWS SageMaker từ code, log huấn luyện, console đến dọn dẹp tài nguyên).*

---

## 2. BẢNG PHÂN BỔ CÔNG VIỆC VÀ THỜI LƯỢNG (≥ 8 GIỜ/TUẦN/SV)

| STT | Hạng mục công việc Tuần 4 | Thời lượng | Trạng thái |
| :---: | :--- | :---: | :---: |
| 1 | Đối soát và phân tích ngân sách thực tế trên Dashboard AWS Academy Learner Lab ($18.1 / $50) | 1.5 giờ | Hoàn thành |
| 2 | Xây dựng bảng đối chuẩn kỹ thuật (Benchmarking) so sánh toàn diện Batch Transform vs Real-time Endpoint | 2.0 giờ | Hoàn thành |
| 3 | Thực thi quy trình kiểm tra và dọn dẹp tài nguyên lần cuối (`scripts/teardown.py`), xác nhận trạng thái 0 tài nguyên tính phí | 1.0 giờ | Hoàn thành |
| 4 | Xây dựng kịch bản, quay và biên tập Video Demo thực nghiệm toàn bộ đồ án MLOps trên AWS SageMaker | 2.5 giờ | Hoàn thành |
| 5 | Tổng hợp toàn bộ mã nguồn, cấu trúc thư mục đồ án và hoàn thiện Báo cáo Tiến độ Tuần 4 | 2.0 giờ | Hoàn thành |

* **Tổng thời gian tuần 4:** **9.0 giờ**, em tập trung vào công tác nghiệm thu hệ thống, phân tích chi phí, đối chuẩn kỹ thuật, quay video demo thực nghiệm và đóng gói đồ án an toàn.

---

## 3. NỘI DUNG CÔNG VIỆC ĐÃ HOÀN THÀNH KÈM LINK FILE VÀ MINH CHỨNG TRÊN GITHUB

Toàn bộ công việc tuần 4 được thực hiện trực tiếp trên tài khoản AWS Academy Learner Lab (Account ID: `909005169225`, Region `us-east-1`):

### 1. Đối Soát Ngân Sách & Quản Lý Chi Phí Thực Tế Trên AWS Learner Lab
Sau 4 tuần triển khai toàn bộ vòng đời MLOps (chuẩn bị dữ liệu S3, huấn luyện 2 mô hình Transfer Learning, chạy Batch Transform 400 ảnh và dựng Endpoint thời gian thực), em tiến hành kiểm tra ngân sách trên Dashboard AWS Academy:
* **Hạn mức tài khoản:** AWS Academy cấp ngân sách ban đầu là **$50.00 USD**.
* **Ngân sách thực tế ghi nhận:** **`Used $18.1 of $50`** (Số dư còn lại: **$31.90 USD**).
* **Đánh giá hiệu quả chi phí:** Toàn bộ quá trình thực nghiệm 4 tuần được kiểm soát chặt chẽ, chi phí tiêu thụ nằm hoàn toàn trong hạn mức cho phép, bảo toàn ngân sách an toàn.
  * Link ảnh minh chứng ngân sách thực tế trên Learner Lab ($18.1 / $50): [images/billing/learner_lab_budget.png](https://github.com/TBL251/Cloud/blob/main/images/billing/learner_lab_budget.png)

---

### 2. Bảng Đối Chuẩn Kỹ Thuật (Benchmarking): Batch Transform vs. Real-time Endpoint
Dựa trên các số liệu đo lường thực tế từ Tuần 3 (Notebook 03 và Notebook 04), em lập bảng so sánh chi tiết giữa hai phương thức phục vụ suy luận của SageMaker:

| Tiêu chí So sánh | SageMaker Batch Transform (Notebook 03) | SageMaker Real-time Endpoint (Notebook 04) |
| :--- | :--- | :--- |
| **Máy ảo phục vụ** | `ml.c5.xlarge` (4 vCPU, 8 GiB RAM) | `ml.m5.large` (2 vCPU, 8 GiB RAM) |
| **Thời gian tồn tại máy ảo** | **Chỉ bật khi có job** (~4 phút cho 400 ảnh), tự động tắt máy ngay sau khi xong | **Bật liên tục 24/7** (hoặc dùng mô hình Transient bật vài phút để test) |
| **Độ trễ suy luận (Latency)** | Xử lý theo lô: ~0.6 giây/ảnh (tính trung bình trên 400 ảnh trong 240 giây) | Xử lý tức thì từng request: **`212.39 ms`** (End-to-End qua mạng HTTP) |
| **Độ chính xác / Output** | Accuracy **90.00%** trên 400 ảnh test, lưu 400 file JSON `.out` lên S3 | Trả về chuỗi JSON trực tiếp cho client với xác suất 4 lớp |
| **Chi phí đo được thực tế** | **`~$0.013 USD`** cho 1 lượt chấm điểm 400 ảnh | **`~$0.015 USD`** cho 5 phút test; nếu để quên 1 tháng sẽ tốn **`~$65.00 USD`** |
| **Khả năng mở rộng (Scaling)** | Tự động phân bổ dữ liệu song song trên các vCPU theo dung lượng payload | Hỗ trợ Auto-scaling dựa theo CloudWatch InvocationsPerInstance |
| **Kịch bản ứng dụng trong Nông nghiệp** | **Phù hợp quét định kỳ:** Hợp tác xã/nông trường dùng flycam hoặc vệ tinh chụp toàn bộ cánh đồng ngô vào cuối ngày, đẩy hàng nghìn ảnh lên S3 để hệ thống tự động quét và xuất báo cáo khoanh vùng dịch bệnh. | **Phù hợp ứng dụng di động cho nông dân:** Người làm vườn cầm điện thoại ra đồng, chụp 1 lá ngô bị đốm, bấm nút trên app để nhận kết quả chẩn đoán bệnh và hướng dẫn phun thuốc ngay tại chỗ. |

---

### 3. Video Demo Thực Nghiệm Toàn Diện Đồ Án Trên AWS SageMaker
Em đã thực hiện ghi hình video demo thực tế với thời lượng ~9 phút, trình bày chi tiết và rõ ràng toàn bộ các cấu phần của dự án:
1. **Giới thiệu đề tài & cấu trúc mã nguồn:** Trình bày mục tiêu phân loại 4 trạng thái lá bắp (Healthy, Common Rust, Gray Leaf Spot, Northern Leaf Blight) và cấu trúc thư mục trên GitHub repo.
2. **Dữ liệu & Quá trình huấn luyện Transfer Learning:** Minh chứng tập dữ liệu 1.600 ảnh trên S3; giải thích kiến trúc MobileNetV2 (~3.5M params); thuyết minh log huấn luyện 8 Epochs trên máy ảo `ml.c5.xlarge` (736 giây tính phí, Best Val Acc đạt 90.00%) và model artifact `model.tar.gz` lưu trên S3.
3. **Đánh giá hàng loạt (Batch Transform) & Ma trận nhầm lẫn:** Trình bày kết quả chạy Batch Transform trên 400 ảnh test độc lập (đạt Accuracy 90.00%, Recall lá khỏe đạt 100/100 tuyệt đối); phân tích biểu đồ nhiệt Confusion Matrix và giải thích lý do nhầm lẫn khách quan giữa vết hoại tử của Gray Leaf Spot và Northern Leaf Blight.
4. **Triển khai Transient Endpoint & Kiểm thử thời gian thực:** Trình bày cách thức hoạt động của bộ phục vụ suy luận `src/inference.py`, kết quả gửi ảnh test qua REST API và đo đạc độ trễ End-to-End thực tế là **`212.39 ms`**.
5. **Quy trình dọn dẹp tài nguyên (Teardown) & Quản lý ngân sách:** Trình bày thao tác hủy endpoint tự động, thực thi script `scripts/teardown.py` xác nhận 0 tài nguyên tính phí, và đối soát số dư an toàn trên AWS Learner Lab Dashboard ($18.1 / $50).

* Link Video Demo trực tiếp: [https://drive.google.com/file/d/1dhWreqLEgqtqwZTe-_w1gkN2HBoCfJsX/view?usp=sharing](https://drive.google.com/file/d/1dhWreqLEgqtqwZTe-_w1gkN2HBoCfJsX/view?usp=sharing)

---

### 4. Quy Trình Thu Dọn Tài Nguyên Toàn Diện (Final Teardown)
Để kết thúc đồ án trong trạng thái an toàn tuyệt đối, em chạy rà soát toàn bộ tài nguyên trên AWS:
1. **Chạy script dọn dẹp tự động:** Thực thi file [scripts/teardown.py](https://github.com/TBL251/Cloud/blob/main/scripts/teardown.py) để quét sạch:
   * 0 Endpoint nào đang chạy (Status: InService/Creating = 0).
   * 0 Endpoint Configuration nào còn tồn đọng.
   * 0 Model nào còn lưu trong SageMaker Model Registry.
2. **Dừng máy ảo điều khiển (Notebook Instance):** Chuyển trạng thái SageMaker Notebook `ml.t3.medium` về `Stopped` khi không thao tác để không bị trừ tiền giờ máy ảo.
3. **Bảo toàn dữ liệu S3:** Dữ liệu ảnh mẫu và model artifact `model.tar.gz` được giữ an toàn trên S3 Bucket `sagemaker-plant-leaf-909005169225` với chi phí lưu trữ cực nhỏ (~$0.02/tháng).
   * Link file script dọn dẹp: [scripts/teardown.py](https://github.com/TBL251/Cloud/blob/main/scripts/teardown.py)
   * Link ảnh minh chứng thực tế xóa sạch Endpoint và Model: [images/notebook04/Screenshot 2026-10-25 174627.png](https://github.com/TBL251/Cloud/blob/main/images/notebook04/Screenshot%202026-10-25%20174627.png)

---

## 4. BẢNG COMMIT GITHUB TRONG TUẦN (KÈM LINK COMMIT CỤ THỂ)

Tuần 4 em thực hiện commit hoàn thiện báo cáo và tài liệu tổng kết trên nhánh `main`:

| STT | Mã Commit (SHA) | Nội dung Commit | Link Commit trên GitHub |
| :---: | :---: | :--- | :---: |
| 1 | `5801dee` | Hoàn thiện báo cáo tiến độ tuần 4 tổng kết chi phí thực tế và đối chuẩn kỹ thuật | [Commit 5801dee](https://github.com/TBL251/Cloud/commit/5801deea78a671463c332d3b1380840c3600db03) |
| 2 | `35566bd` | Cập nhật README, worklog và tối ưu gitignore cho video demo đồ án | [Commit 35566bd](https://github.com/TBL251/Cloud/commit/35566bde68069580c42ff898db271eb003e3f11f) |

---

## 5. BÀI HỌC KINH NGHIỆM VÀ TỔNG KẾT ĐỒ ÁN

Qua 4 tuần tự tay triển khai toàn diện đồ án MLOps trên nền tảng AWS SageMaker, em rút ra được những bài học kỹ thuật quý giá:

1. **Ý thức kiểm soát và quản lý ngân sách Cloud:**
   * Trong suốt quá trình thực hành, việc bám sát số dư và chi tiêu thực tế trên Dashboard giúp em luôn chủ động kiểm soát tài nguyên. Tổng chi phí toàn bộ đồ án chỉ tiêu tốn $18.1 trên hạn mức $50 được cấp, bảo toàn an toàn tài khoản học tập.

2. **Bài học về tối ưu chi phí khi thiết kế hệ thống AI:**
   * Không nhất thiết phải dùng GPU đắt tiền (như `ml.g4dn` hay `ml.p3`) để giải quyết bài toán thị giác máy tính. Với việc chọn kiến trúc mạng nhẹ MobileNetV2 và áp dụng Transfer Learning, máy ảo CPU `ml.c5.xlarge` hoàn toàn đủ sức huấn luyện mô hình đạt độ chính xác 90% chỉ trong 12 phút với chi phí chưa tới $0.05.
   * Lựa chọn đúng giữa **Batch Transform** và **Real-time Endpoint** giúp tiết kiệm chi phí gấp hàng chục lần tùy theo bài toán nghiệp vụ thực tế.

3. **Kỷ luật vận hành và dọn dẹp tài nguyên (Teardown):**
   * Trong môi trường Cloud, "quên tắt tài nguyên" là nguyên nhân phổ biến nhất dẫn đến vượt ngân sách. Việc xây dựng kịch bản tự động xóa Endpoint ngay sau khi đo và viết script `teardown.py` bằng Boto3 giúp em hoàn toàn chủ động, giữ an toàn tài khoản qua suốt 4 tuần học.
