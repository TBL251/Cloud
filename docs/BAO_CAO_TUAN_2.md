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
| 2 | Chuẩn hóa tập dữ liệu 1.600 ảnh thật (PlantVillage), tạo S3 bucket, gắn thẻ Tag (`Project`, `Owner`) và upload lên S3 | 2.5 giờ | Hoàn thành |
| 3 | Cấu hình SageMaker PyTorch Estimator (Script Mode) và huấn luyện Transfer Learning MobileNetV2 trên CPU `ml.c5.xlarge` | 2.5 giờ | Hoàn thành |
| 4 | Thu thập kết quả huấn luyện (Validation Acc: 90%, 736s), xác nhận `model.tar.gz` trên S3 và dừng tài nguyên (Stop Notebook) | 2.0 giờ | Hoàn thành |

* **Tổng thời gian thực hiện Tuần 2:** **8.5 giờ** (Đạt yêu cầu quy định >= 8 giờ/tuần).

---

## 3. NỘI DUNG CÔNG VIỆC ĐÃ HOÀN THÀNH KÈM LINK FILE VÀ MINH CHỨNG TRÊN GITHUB

Toàn bộ quá trình thực thi trên tài khoản AWS Academy Learner Lab thật (Account ID: `909005169225`, Region `us-east-1`) đã hoàn thành xuất sắc với tập dữ liệu ảnh thật 100%:

### 1. Chuẩn bị Dữ liệu & Đồng bộ lên Amazon S3 (Notebook 01)
* Khởi tạo S3 Bucket `sagemaker-plant-leaf-909005169225` và gắn thẻ Tag định danh bắt buộc: `Project=PlantLeafClassification`, `Owner=LeBaTam`.
* Tiền xử lý, chuẩn hóa kích thước ảnh về chuẩn `224x224`, chia tập dữ liệu 80% Train (1.280 ảnh) và 20% Val (320 ảnh) cho 4 lớp bệnh lá ngô thật từ tập dữ liệu học thuật PlantVillage (`Corn___Northern_Leaf_Blight`, `Corn___Healthy`, `Corn___Gray_Leaf_Spot`, `Corn___Common_Rust`).
* Upload toàn bộ đúng 1.600 ảnh thật lên S3, tuân thủ nghiêm ngặt giới hạn <= 2.000 ảnh của đề bài.
  * Link file notebook: [notebooks/01_data_preparation.ipynb](https://github.com/TBL251/Cloud/blob/main/notebooks/01_data_preparation.ipynb)
  * Link file script: [src/prepare_data.py](https://github.com/TBL251/Cloud/blob/main/src/prepare_data.py)
  * Link ảnh minh chứng tạo S3 Bucket: [images/notebook01/Screenshot 2026-10-18 235104.png](https://github.com/TBL251/Cloud/blob/main/images/notebook01/Screenshot%202026-10-18%20235104.png)
  * Link ảnh minh chứng gắn thẻ Tag bắt buộc: [images/notebook01/Screenshot 2026-10-18 235111.png](https://github.com/TBL251/Cloud/blob/main/images/notebook01/Screenshot%202026-10-18%20235111.png)
  * Link ảnh minh chứng tiền xử lý 1.600 ảnh thật & upload S3: [images/notebook01/Screenshot 2026-10-18 235129.png](https://github.com/TBL251/Cloud/blob/main/images/notebook01/Screenshot%202026-10-18%20235129.png)
  * Link ảnh minh chứng danh sách đối tượng mẫu trên S3: [images/notebook01/Screenshot 2026-10-18 235137.png](https://github.com/TBL251/Cloud/blob/main/images/notebook01/Screenshot%202026-10-18%20235137.png)

### 2. Huấn luyện Mô hình Transfer Learning trên SageMaker (Notebook 02)
* Sử dụng SageMaker Script Mode với PyTorch Estimator, nạp mã nguồn huấn luyện `train.py` từ repo GitHub.
* Khởi tạo máy chủ huấn luyện độc lập trên CPU instance `ml.c5.xlarge` (4 vCPU, 8 GiB RAM), không dùng GPU nhằm tuân thủ hạn ngạch của AWS Learner Lab.
* Mã Training Job: `pytorch-training-2026-10-18-16-35-53-425`.
* Huấn luyện mô hình Transfer Learning **MobileNetV2** qua 8 epochs trên 1.600 ảnh thật. Tiến trình học diễn ra tự nhiên:
  * Epoch 1: Train Acc 65.44% | Val Acc 84.25%
  * Epoch 4: Train Acc 86.81% | Val Acc 88.00%
  * Epoch 8: Train Acc 88.06% | Val Acc 90.00%
  * **Best Validation Accuracy đạt 90.00% (0.9000)**.
* SageMaker Training Job tự động hoàn tất và đóng gói Model Artifact thành `model.tar.gz` lưu trữ an toàn trên S3.
  * Link file notebook: [notebooks/02_sagemaker_training.ipynb](https://github.com/TBL251/Cloud/blob/main/notebooks/02_sagemaker_training.ipynb)
  * Link file script train: [src/train.py](https://github.com/TBL251/Cloud/blob/main/src/train.py)
  * Link ảnh minh chứng cấu hình PyTorch Estimator: [images/notebook02/MobileNetV2/Screenshot 2026-10-18 235156.png](https://github.com/TBL251/Cloud/blob/main/images/notebook02/MobileNetV2/Screenshot%202026-10-18%20235156.png)
  * Link ảnh minh chứng khởi tạo máy ảo ml.c5.xlarge: [images/notebook02/MobileNetV2/Screenshot 2026-10-18 235222.png](https://github.com/TBL251/Cloud/blob/main/images/notebook02/MobileNetV2/Screenshot%202026-10-18%20235222.png)
  * Link ảnh minh chứng quá trình huấn luyện qua 8 Epoch (Val Acc 90%): [images/notebook02/MobileNetV2/Screenshot 2026-10-18 235244.png](https://github.com/TBL251/Cloud/blob/main/images/notebook02/MobileNetV2/Screenshot%202026-10-18%20235244.png)
  * Link ảnh minh chứng hoàn tất huấn luyện (736 giây tính phí): [images/notebook02/MobileNetV2/Screenshot 2026-10-18 235300.png](https://github.com/TBL251/Cloud/blob/main/images/notebook02/MobileNetV2/Screenshot%202026-10-18%20235300.png)
  * Link ảnh minh chứng Model Artifact `model.tar.gz` trên S3: [images/notebook02/MobileNetV2/Screenshot 2026-10-18 235306.png](https://github.com/TBL251/Cloud/blob/main/images/notebook02/MobileNetV2/Screenshot%202026-10-18%20235306.png)

### 3. Khảo sát & So sánh Thực nghiệm: MobileNetV2 vs ResNet-18 (Dữ liệu Huấn luyện Thật trên SageMaker)
Nhằm giải thích thuyết phục cho việc lựa chọn kiến trúc tối ưu nhất trên Cloud theo đúng gợi ý của GVHD, sinh viên đã kích hoạt cả 2 Training Job độc lập trên cùng cấu hình CPU instance `ml.c5.xlarge`, cùng tập 1.600 ảnh thật và 8 Epochs:
* **Mã Training Job MobileNetV2:** `pytorch-training-2026-10-18-16-35-53-425`
* **Mã Training Job ResNet-18:** `pytorch-training-2026-10-18-18-21-42-644`
* **Link ảnh minh chứng Training Job ResNet-18 (551 giây tính phí, Best Val Acc 87%):** [images/notebook02/ResNet-18/Screenshot 2026-10-19 013942.png](https://github.com/TBL251/Cloud/blob/main/images/notebook02/ResNet-18/Screenshot%202026-10-19%20013942.png)

| Tiêu chí so sánh | MobileNetV2 (Lựa chọn chính) | ResNet-18 (Đối sánh thực tế) | Ý nghĩa trên Điện toán đám mây |
| :--- | :---: | :---: | :--- |
| **Số lượng tham số (Parameters)** | **~3.5 Triệu** | ~11.7 Triệu | MobileNetV2 nhẹ hơn gấp **3.3 lần**, chiếm ít RAM máy ảo |
| **Dung lượng file Model Artifact** | **~14.2 MB** | **44.7 MB** | MobileNetV2 tiết kiệm **68% dung lượng lưu trữ S3** và truyền tải mạng |
| **Thời gian tính phí (Billable seconds)**| **736 giây** | **551 giây** | Cả 2 job đều hoàn tất nhanh chóng dưới 15 phút trên CPU |
| **Chi phí Training Job thực tế** | **~$0.035 USD** | **~$0.026 USD** | Cực kỳ tiết kiệm, bảo đảm an toàn ngân sách tuần < 3$ |
| **Độ ổn định huấn luyện qua 8 Epochs** | **Tăng đều đặn:** 84.25% -> 90.00% | **Có hiện tượng Overfit:** Val Acc đạt đỉnh 87.00% (Epoch 3) rồi giảm còn 78.50% (Epoch 8) do số tham số quá lớn |
| **Best Validation Accuracy (1.600 ảnh)** | **90.00% (0.9000)** | **87.00% (0.8700)** | **MobileNetV2 vượt trội hơn +3.00%** độ chính xác trên tập dữ liệu lá cây |

* **Kết luận lựa chọn kỹ thuật:** Kết quả thực nghiệm thật 100% trên AWS SageMaker chứng minh **MobileNetV2 là sự lựa chọn tối ưu toàn diện nhất**: vừa cho độ chính xác cao hơn (90% so với 87%), vừa chống hiện tượng quá khớp (overfitting) tốt hơn trên tập ảnh lá cây, đồng thời dung lượng mô hình chỉ bằng 1/3 (14 MB so với 44.7 MB), giúp giảm tối đa độ trễ nạp mô hình khi triển khai Endpoint thời gian thực ở Tuần 3.

---

## 4. LỊCH SỬ COMMIT TRÊN GITHUB PHẢN ÁNH TIẾN ĐỘ CÔNG VIỆC (YÊU CẦU 6)

Sinh viên thực hiện phân chia công việc rõ ràng, mỗi hạng mục hoàn thành đều được commit và đẩy lên GitHub Public:

* `them bo du lieu 1600 anh that plantvillage vao thu muc data`  
  Link commit: https://github.com/TBL251/Cloud/commit/31ca6ae0e7bd68dedea42fdfc1bf53d1e681cb1d
* `cap nhat cell 4 notebook 01 tu dong tai plantvillage va xu ly anh that`  
  Link commit: https://github.com/TBL251/Cloud/commit/42d892e1ccf88dcfd9f6d2f94774474264b2a414
* `bo sung cell thuc nghiem doi sanh resnet18 vao notebook 02`  
  Link commit: https://github.com/TBL251/Cloud/commit/23bc079c7e8df68320e2ecdb6d2032d53a460beb
* `bo sung so sanh mobilenetv2 vs resnet18 va ke hoach tuan 3 theo nhan xet cua gvhd`  
  Link commit: https://github.com/TBL251/Cloud/commit/87a1298b12c834a4ddf5c43b24ea0f0ee1755aa7

---

## 5. BÁO CÁO TÀI NGUYÊN & CHI PHÍ THỰC TẾ TRÊN AWS

Tuân thủ nghiêm ngặt yêu cầu của GVHD: **ngân sách tuần dưới 3$** và **tổng đồ án dưới 50$**:
* **Account ID thực tế:** `909005169225` (Region `us-east-1`).
* **Chi phí Training Job 1 (MobileNetV2 - 736 giây):** **~$0.035 USD**.
* **Chi phí Training Job 2 (ResNet-18 - 551 giây):** **~$0.026 USD**.
* **Lưu trữ S3:** 1.600 ảnh thật và 2 file model artifacts (`model.tar.gz`) -> Chi phí: **~$0.001 USD**.
* **SageMaker Notebook:** Chạy trên `ml.t3.medium` trong quá trình thực hành -> Chi phí: **~$0.040 USD**.
* **Biện pháp an toàn chi phí:** Đã thực hiện lệnh **Stop** Notebook Instance ngay sau buổi thực hành để ngắt hoàn toàn chi phí giờ chạy máy ảo.
* **Tổng chi phí thực tế Tuần 2 (cho cả 2 Training Jobs):** **~$0.102 USD** (~2.550 VNĐ, chỉ chiếm **3.4%** mức ngân sách 3$/tuần do GVHD quy định, cách rất xa ngưỡng trần 50$ của đề bài).
* **Kế hoạch đối chiếu Billing:** Sinh viên đã lưu vết số liệu và cam kết chụp screenshot toàn diện trang AWS Billing vào cuối Tuần 4 để đối chiếu theo đúng nhắc nhở của thầy.

---

## 6. KẾ HOẠCH CÔNG VIỆC TUẦN 3 TIẾP THEO

Toàn bộ 5 đầu mục công việc cụ thể của Tuần 3 đã được lập sẵn trong repository:
* Link file Worklog Tuần 3: [docs/worklogs/week_3_worklog.md](https://github.com/TBL251/Cloud/blob/main/docs/worklogs/week_3_worklog.md)

---

## 7. KHAI BÁO SỬ DỤNG AI (AI DECLARATION) & MINH CHỨNG

Tuân thủ nghiêm ngặt quy chế đề bài về tính minh bạch khi có AI hỗ trợ:

**PROMPT**
> Tôi đang làm đồ án môn Điện toán đám mây triển khai huấn luyện mô hình Transfer Learning phân loại bệnh lá cây trên AWS SageMaker (Region us-east-1, tài khoản Learner Lab). Hãy gợi ý cú pháp khởi tạo PyTorch Estimator trong SageMaker Python SDK v2 chạy trên CPU ml.c5.xlarge, cách đóng gói mã nguồn huấn luyện train.py (Script Mode) và truyền các kênh dữ liệu S3 train, val.

**AI TẠO RA**
* Đoạn mã mẫu khởi tạo PyTorch Estimator sử dụng instance GPU đắt tiền (`ml.p3.2xlarge` hoặc `ml.g4dn.xlarge`) không tương thích với hạn ngạch Service Quota của AWS Learner Lab.
* Đề xuất sử dụng các kiến trúc mạng nặng (ResNet-50, VGG-16) đòi hỏi GPU để huấn luyện trong thời gian dài.
* Cấu hình tham số source_dir mặc định trỏ về thư mục hiện tại ('.'), gây lỗi không tìm thấy train.py khi chạy trong môi trường Jupyter phân cấp của SageMaker.
* Thiếu cấu hình bắt buộc gắn thẻ Tag định danh tài nguyên (Project, Owner) và sử dụng IAM LabRole có sẵn.
* Đề xuất tải toàn bộ dữ liệu thô dung lượng lớn trực tiếp lên S3 mà không tiền xử lý kích thước ảnh chuẩn 224x224.

**SV ĐÃ SỬA**
* **Chuyển đổi và tinh chỉnh toàn bộ sang CPU instance:** Thay thế cấu hình GPU bằng CPU `ml.c5.xlarge` (4 vCPU, 8 GiB RAM), đổi sang kiến trúc mạng nhẹ **MobileNetV2** để hoàn tất huấn luyện 1.600 ảnh thật trong đúng 736 giây với độ chính xác Best Validation Accuracy đạt **90.00%**.
* **Tiết kiệm và khống chế chi phí tối đa:** Tổng chi phí buổi huấn luyện chỉ tốn ~$0.035 USD (cách xa ngưỡng trần 50$ của đề bài); tự viết quy trình Stop Notebook ngay sau khi hoàn thành để chống phát sinh chi phí ngoài ý muốn.
* **Bổ sung gắn thẻ Tag định danh bắt buộc:** Cấu hình trực tiếp TagSet (`Project=PlantLeafClassification`, `Owner=LeBaTam`) vào S3 Bucket và tham số tags của PyTorch Estimator, đồng thời liên kết chính xác IAM LabRole của Learner Lab (`arn:aws:iam::909005169225:role/LabRole`).
* **Xử lý triệt để lỗi môi trường thực tế:** Tinh chỉnh tham số `source_dir='Cloud/src'` để SageMaker nạp đúng mã nguồn từ GitHub repo; bổ sung cơ chế gọi môi trường Conda Python (`sys.executable`) để giải quyết dứt điểm xung đột thư viện Pillow/NumPy trên máy ảo.
* **Khảo sát và thực thi trực tiếp trên AWS thật:** Tự tay vận hành trên tài khoản Learner Lab thật, kiểm tra log huấn luyện qua từng Epoch và chụp đầy đủ các ảnh minh chứng có hiển thị Account ID `909005169225` và Region `us-east-1`.
