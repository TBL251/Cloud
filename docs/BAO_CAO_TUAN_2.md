# BÁO CÁO TIẾN ĐỘ TUẦN 2
**Môn học:** Cloud - Đợt 1 - 2026-2027  
**Giảng viên hướng dẫn (GVHD):** Huỳnh Xuân Phụng  
**Đề tài:** Phân loại ảnh bằng Transfer Learning trên SageMaker (Nhóm: AI/ML)  

---

## 1. THÔNG TIN THỰC HIỆN & LINK GITHUB PUBLIC
* **Sinh viên thực hiện:** Lê Bá Tâm - MSSV: 24110055 - Lớp: CLCO332779_05UTExMC  
* **Link GitHub Repository (Public):** https://github.com/TBL251/Cloud  
*(Repository ở chế độ Public để GVHD truy cập và kiểm chứng mã nguồn trực tiếp).*

---

## 2. BẢNG PHÂN BỔ CÔNG VIỆC VÀ THỜI LƯỢNG (≥ 8 GIỜ/TUẦN/SV)

| STT | Hạng mục công việc Tuần 2 | Thời lượng | Trạng thái |
| :---: | :--- | :---: | :---: |
| 1 | Khởi tạo SageMaker Notebook Instance (`ml.t3.medium`), cấu hình môi trường PyTorch và IAM LabRole | 1.5 giờ | Hoàn thành |
| 2 | Viết và thực thi script tiền xử lý ảnh, tạo S3 bucket, gắn thẻ Tag (`Project`, `Owner`) và tải 400 ảnh lên S3 | 2.5 giờ | Hoàn thành |
| 3 | Cấu hình SageMaker PyTorch Estimator (Script Mode) và huấn luyện Transfer Learning MobileNetV2 trên CPU `ml.c5.xlarge` | 2.5 giờ | Hoàn thành |
| 4 | Thu thập kết quả huấn luyện (Accuracy 1.0000, 224s), xác nhận `model.tar.gz` trên S3 và dừng tài nguyên (Stop Notebook) | 2.0 giờ | Hoàn thành |

* **Tổng thời gian thực hiện Tuần 2:** **8.5 giờ** (Đạt yêu cầu quy định $\ge$ 8 giờ/tuần).

---

## 3. NỘI DUNG CÔNG VIỆC ĐÃ HOÀN THÀNH KÈM LINK FILE VÀ MINH CHỨNG TRÊN GITHUB

Toàn bộ quá trình thực thi trên tài khoản AWS Academy Learner Lab thật (Account ID: `909005169225`, Region `us-east-1`) đã hoàn thành và được chụp ảnh minh chứng chi tiết:

### 1. Chuẩn bị Dữ liệu & Đồng bộ lên Amazon S3 (Notebook 01)
* Khởi tạo S3 Bucket `sagemaker-plant-leaf-909005169225` và gắn thẻ Tag định danh bắt buộc: `Project=PlantLeafClassification`, `Owner=LeBaTam`.
* Tiền xử lý, chuẩn hóa kích thước ảnh về chuẩn `224x224`, chia tập dữ liệu 80% Train và 20% Val cho 4 lớp bệnh lá bắp (`Corn___Healthy`, `Corn___Common_Rust`, `Corn___Gray_Leaf_Spot`, `Corn___Northern_Leaf_Blight`).
* Upload toàn bộ 400 ảnh (320 ảnh train, 80 ảnh val) lên S3, tuân thủ nghiêm ngặt giới hạn $\le$ 2.000 ảnh của đề bài.
  * Link file notebook: [notebooks/01_data_preparation.ipynb](https://github.com/TBL251/Cloud/blob/main/notebooks/01_data_preparation.ipynb)
  * Link file script: [src/prepare_data.py](https://github.com/TBL251/Cloud/blob/main/src/prepare_data.py)
  * Link ảnh minh chứng tạo S3 Bucket & gắn thẻ Tag: [images/notebook01/Screenshot 2026-10-18 221445.png](https://github.com/TBL251/Cloud/blob/main/images/notebook01/Screenshot%202026-10-18%20221445.png)
  * Link ảnh minh chứng tiền xử lý 4 lớp ảnh: [images/notebook01/Screenshot 2026-10-18 221836.png](https://github.com/TBL251/Cloud/blob/main/images/notebook01/Screenshot%202026-10-18%20221836.png)
  * Link ảnh minh chứng hoàn tất upload 400 ảnh lên S3: [images/notebook01/Screenshot 2026-10-18 221841.png](https://github.com/TBL251/Cloud/blob/main/images/notebook01/Screenshot%202026-10-18%20221841.png)

### 2. Huấn luyện Mô hình Transfer Learning trên SageMaker (Notebook 02)
* Sử dụng SageMaker Script Mode với PyTorch Estimator, nạp mã nguồn huấn luyện `train.py` từ repo.
* Cấu hình máy chủ huấn luyện độc lập trên CPU instance `ml.c5.xlarge` (4 vCPU, 8 GiB RAM), không sử dụng GPU nhằm tuân thủ quota của AWS Learner Lab.
* Huấn luyện mô hình Transfer Learning **MobileNetV2** qua 8 epochs. Mô hình đạt độ chính xác **Best Validation Accuracy: 1.0000 (100%)**.
* SageMaker Training Job tự động hoàn tất và đóng gói Model Artifact thành `model.tar.gz` lưu trữ an toàn trên S3.
  * Link file notebook: [notebooks/02_sagemaker_training.ipynb](https://github.com/TBL251/Cloud/blob/main/notebooks/02_sagemaker_training.ipynb)
  * Link file script train: [src/train.py](https://github.com/TBL251/Cloud/blob/main/src/train.py)
  * Link ảnh minh chứng cấu hình PyTorch Estimator: [images/notebook02/Screenshot 2026-10-18 221947.png](https://github.com/TBL251/Cloud/blob/main/images/notebook02/Screenshot%202026-10-18%20221947.png)
  * Link ảnh minh chứng tải container & khởi tạo ml.c5.xlarge: [images/notebook02/Screenshot 2026-10-18 222017.png](https://github.com/TBL251/Cloud/blob/main/images/notebook02/Screenshot%202026-10-18%20222017.png)
  * Link ảnh minh chứng tiến trình huấn luyện qua các Epoch: [images/notebook02/Screenshot 2026-10-18 222043.png](https://github.com/TBL251/Cloud/blob/main/images/notebook02/Screenshot%202026-10-18%20222043.png)
  * Link ảnh minh chứng huấn luyện thành công (224s, Acc 1.0000): [images/notebook02/Screenshot 2026-10-18 222104.png](https://github.com/TBL251/Cloud/blob/main/images/notebook02/Screenshot%202026-10-18%20222104.png)
  * Link ảnh minh chứng Model Artifact `model.tar.gz` trên S3: [images/notebook02/Screenshot 2026-10-18 222116.png](https://github.com/TBL251/Cloud/blob/main/images/notebook02/Screenshot%202026-10-18%20222116.png)

---

## 4. BÁO CÁO TÀI NGUYÊN & CHI PHÍ THỰC TẾ TRÊN AWS

Nhằm tuân thủ nghiêm ngặt yêu cầu ngân sách tối đa dưới 50 USD của đề bài:
* **Account ID thực tế:** `909005169225` (Region `us-east-1`).
* **Thời gian tính phí Training Job:** Đúng **224 giây** (~3.7 phút) trên máy `ml.c5.xlarge` $\rightarrow$ Chi phí: **~$0.012 USD**.
* **Lưu trữ S3:** 400 ảnh và file mô hình `model.tar.gz` (~28 MB) $\rightarrow$ Chi phí: **~$0.001 USD**.
* **SageMaker Notebook:** Chạy trên `ml.t3.medium` trong ~40 phút thực hành $\rightarrow$ Chi phí: **~$0.033 USD**.
* **Biện pháp an toàn chi phí:** Đã thực hiện lệnh **Stop** Notebook Instance ngay sau buổi thực hành để ngắt hoàn toàn chi phí giờ chạy máy ảo.
* **Tổng chi phí thực tế Tuần 2:** **~$0.046 USD** (tương đương ~1.150 VNĐ, bảo đảm cách rất xa giới hạn 50$ của đề bài).

---

## 5. KHAI BÁO SỬ DỤNG AI (AI DECLARATION) & MINH CHỨNG

Tuân thủ nghiêm ngặt quy chế đề bài về tính minh bạch khi có AI hỗ trợ:

**PROMPT**
> Tôi đang làm đồ án môn Điện toán đám mây triển khai huấn luyện mô hình Transfer Learning trên AWS SageMaker (Region us-east-1, tài khoản Learner Lab). Hãy gợi ý cú pháp khởi tạo PyTorch Estimator trong SageMaker Python SDK v2 chạy trên CPU ml.c5.xlarge và cách truyền đường dẫn channel S3 train và val.

**AI TẠO RA**
* Đoạn mã khởi tạo PyTorch Estimator mặc định với instance GPU đắt tiền (`ml.p3.2xlarge` hoặc `ml.g4dn.xlarge`).
* Sử dụng mạng lớn ResNet-50 hoặc VGG-16 cần tài nguyên GPU để huấn luyện.
* Đường dẫn `source_dir` được đặt mặc định là thư mục hiện tại không tương thích với cấu trúc repository phân cấp trên SageMaker.
* Thiếu phần cấu hình thẻ Tag định danh tài nguyên bắt buộc theo quy chế chấm bài.

**SV ĐÃ SỬA**
* **Chuyển đổi toàn bộ sang CPU instance:** Thay thế cấu hình GPU bằng `ml.c5.xlarge` và chuyển sang mạng nhẹ **MobileNetV2**, giúp giảm thời gian chạy xuống 224 giây và chi phí chỉ tốn $0.012 USD.
* **Gắn thẻ Tag định danh bắt buộc:** Bổ sung cấu hình thẻ Tag `Project=PlantLeafClassification` và `Owner=LeBaTam` trực tiếp vào Estimator và S3 Bucket.
* **Xử lý triệt để lỗi môi trường:** Điều chỉnh tham số `source_dir='Cloud/src'` và bổ sung lệnh gọi môi trường Conda Python (`sys.executable`) nhằm xử lý xung đột thư viện `Pillow` và cấu trúc thư mục trên JupyterLab của SageMaker.
* **Xác nhận kết quả thực tế:** Trực tiếp giám sát quá trình chạy trên AWS thật, thu thập đầy đủ 14 ảnh minh chứng có Account ID `909005169225`, ghi nhận thời gian chạy 224 giây và xác thực model artifact được lưu trữ an toàn trên S3.
