# BÁO CÁO TIẾN ĐỘ TUẦN 3
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

| STT | Hạng mục công việc Tuần 3 | Thời lượng | Trạng thái |
| :---: | :--- | :---: | :---: |
| 1 | Viết script phục vụ suy luận `inference.py` (model_fn, input_fn, predict_fn, output_fn) để SageMaker nạp model và nhận diện ảnh | 2.0 giờ | Hoàn thành |
| 2 | Cấu hình và chạy SageMaker Batch Transform Job trên CPU instance `ml.c5.xlarge` để dự đoán 400 ảnh kiểm thử (100 ảnh/lớp) | 2.5 giờ | Hoàn thành |
| 3 | Viết code đọc kết quả từ S3 (`evaluate.py`), xuất bảng Classification Report và vẽ biểu đồ Confusion Matrix Heatmap | 2.0 giờ | Hoàn thành |
| 4 | Triển khai thử nghiệm Real-time Endpoint trên máy ảo `ml.m5.large`, đo độ trễ suy luận (Latency ms/ảnh), nhận JSON kết quả và xóa ngay Endpoint | 1.5 giờ | Hoàn thành |
| 5 | Viết script dọn dẹp tài nguyên tự động `teardown.py` và kiểm tra lại chi phí thực tế trên AWS Learner Lab | 1.0 giờ | Hoàn thành |

* **Tổng thời gian tuần 3:** **9.0 giờ**, em tập trung viết code phục vụ suy luận, chạy đánh giá mô hình bằng Batch Transform và kiểm thử Endpoint rồi xóa dọn dẹp ngay.

---

## 3. NỘI DUNG CÔNG VIỆC ĐÃ HOÀN THÀNH KÈM LINK FILE VÀ MINH CHỨNG TRÊN GITHUB

Toàn bộ công việc tuần 3 được triển khai trực tiếp trên tài khoản AWS Academy Learner Lab (Account ID: `909005169225`, Region `us-east-1`):

### 1. Xây dựng Script Phục vụ Suy luận Độc lập (`inference.py`)
Em viết file `src/inference.py` làm custom handler để SageMaker PyTorch container nạp mô hình và xử lý ảnh theo luồng 4 bước:
* `model_fn(model_dir)`: Khởi tạo khung mạng MobileNetV2, đọc file `model_meta.json` để lấy danh sách 4 nhãn phân loại, sau đó nạp trọng số đã huấn luyện từ file `model.pth`.
* `input_fn(request_body, request_content_type)`: Nhận dữ liệu ảnh nhị phân dạng JPEG/PNG hoặc base64, chỉnh về kích thước `224x224` và chuẩn hóa theo phân phối chuẩn ImageNet thành Tensor kích thước `[1, 3, 224, 224]`.
* `predict_fn(input_data, model)`: Chạy forward pass trên CPU không tính gradient, trích xuất điểm Softmax xác suất của từng lớp bệnh.
* `output_fn(prediction, accept)`: Trả về kết quả dạng JSON gồm nhãn dự đoán cao nhất (`predicted_class`), độ tin cậy (`confidence`) và xác suất chi tiết của từng lớp.
  * Link mã nguồn: [src/inference.py](https://github.com/TBL251/Cloud/blob/main/src/inference.py)

---

### 2. Đánh giá Mô hình trên Tập Kiểm thử bằng SageMaker Batch Transform (Notebook 03)
Để đánh giá mô hình trên tập kiểm thử mà không cần giữ máy ảo chạy liên tục tốn tiền, em dùng tính năng **Batch Transform Job** của SageMaker:
* **Mô hình sử dụng:** Model Artifact MobileNetV2 (`s3://sagemaker-plant-leaf-909005169225/output/pytorch-training-2026-10-18-16-35-53-425/output/model.tar.gz`) đạt Validation Accuracy 90% ở Tuần 2.
* **Cấu hình máy ảo:** Dùng instance `ml.c5.xlarge` (4 vCPU, 8 GiB RAM). Máy ảo chỉ tự bật lên khi chạy job và tự tắt ngay khi xử lý xong toàn bộ ảnh.
* **Dữ liệu kiểm thử:** Toàn bộ 400 ảnh kiểm thử từ tập PlantVillage trên S3 (`s3://sagemaker-plant-leaf-909005169225/dataset/val/`), gồm 100 ảnh cho mỗi lớp.
* **Đầu ra:** 400 file `.out` chứa kết quả JSON được SageMaker lưu về thư mục `s3://sagemaker-plant-leaf-909005169225/batch_transform_output/`.

#### Bảng Báo cáo Đánh giá Mô hình (Classification Report trên 400 ảnh kiểm thử):
```
=================================================================
  BÁO CÁO ĐÁNH GIÁ MÔ HÌNH (CLASSIFICATION REPORT)
=================================================================
                             precision    recall  f1-score   support

         Corn___Common_Rust       0.96      0.92      0.94       100
      Corn___Gray_Leaf_Spot       0.85      0.81      0.83       100
             Corn___Healthy       0.98      1.00      0.99       100
Corn___Northern_Leaf_Blight       0.81      0.87      0.84       100

                   accuracy                           0.90       400
                  macro avg       0.90      0.90      0.90       400
               weighted avg       0.90      0.90      0.90       400
```

#### Phân tích Ma trận Nhầm lẫn (Confusion Matrix):
* **Độ chính xác tổng thể (Accuracy):** Đạt đúng **`90.00%`** (360/400 ảnh đoán đúng). Con số này khớp với kết quả kiểm tra ở Tuần 2.
* **Lớp lá ngô khỏe mạnh (`Corn___Healthy`):** Đạt Recall **`100% (100/100)`**, Precision đạt **`0.98`**, F1-score đạt **`0.99`**. Mô hình nhận diện rất chuẩn lá khỏe, không đoán nhầm bất kỳ lá bệnh nào sang lá khỏe.
* **Lớp bệnh rỉ sắt (`Corn___Common_Rust`):** Đạt Recall **`92% (92/100)`**, F1-score **`0.94`**. Các đốm rỉ màu nâu nổi rõ trên lá giúp mô hình dễ nhận diện chính xác.
* **Hiện tượng nhầm lẫn giữa Gray Leaf Spot và Northern Leaf Blight:**
  * Ma trận ghi nhận **16 ảnh** Gray Leaf Spot bị đoán nhầm thành Northern Leaf Blight, và **11 ảnh** Northern Leaf Blight bị đoán nhầm thành Gray Leaf Spot.
  * **Giải thích thực tế:** Cả hai loại bệnh này đều do nấm gây ra, biểu hiện thành các vết xám nâu chạy dọc theo gân lá ngô. Khi bệnh mới chớm hoặc chụp dưới ánh sáng ngoài ruộng, hình dạng tổn thương nhìn rất giống nhau, nên mạng MobileNetV2 khi nén đặc trưng dễ bị nhầm lẫn giữa hai lớp này.
  * Link file notebook: [notebooks/03_batch_transform.ipynb](https://github.com/TBL251/Cloud/blob/main/notebooks/03_batch_transform.ipynb)
  * Link file script đánh giá: [src/evaluate.py](https://github.com/TBL251/Cloud/blob/main/src/evaluate.py)
  * Link biểu đồ ma trận nhầm lẫn: [images/confusion_matrix.png](https://github.com/TBL251/Cloud/blob/main/images/confusion_matrix.png)
  * Link ảnh minh chứng thiết lập môi trường Notebook 03: [images/notebook03/Screenshot 2026-10-25 170259.png](https://github.com/TBL251/Cloud/blob/main/images/notebook03/Screenshot%202026-10-25%20170259.png)
  * Link ảnh minh chứng nạp PyTorchModel với inference.py: [images/notebook03/Screenshot 2026-10-25 170305.png](https://github.com/TBL251/Cloud/blob/main/images/notebook03/Screenshot%202026-10-25%20170305.png)
  * Link ảnh minh chứng đọc 400 kết quả dự đoán từ S3: [images/notebook03/Screenshot 2026-10-25 170922.png](https://github.com/TBL251/Cloud/blob/main/images/notebook03/Screenshot%202026-10-25%20170922.png)
  * Link ảnh minh chứng Classification Report & Heatmap: [images/notebook03/Screenshot 2026-10-25 170927.png](https://github.com/TBL251/Cloud/blob/main/images/notebook03/Screenshot%202026-10-25%20170927.png)

---

### 3. Triển khai Thử nghiệm Real-time Endpoint & Đo Độ Trễ Suy luận (Notebook 04)
Để phục vụ việc so sánh giữa cơ chế Batch và Real-time theo yêu cầu đề bài, em triển khai thử nghiệm một Endpoint tạm thời:
* **Máy ảo phục vụ:** `ml.m5.large` (2 vCPU, 8 GiB RAM), có gắn tag: `Project=PlantLeafClassification`, `Owner=LeBaTam`.
* **Gửi ảnh kiểm thử:** Lấy ảnh lá ngô bệnh `../data/Blight/Corn_Blight (145).JPG` và gửi HTTP request đến Endpoint qua `boto3.client('sagemaker-runtime')`.
* **Số liệu đo được:**
  * **Độ trễ suy luận (Latency):** **`212.39 ms`** (tính từ lúc client gửi request qua mạng, máy ảo tiền xử lý, chạy model trên CPU và trả kết quả về).
  * **Kết quả dự đoán:** Mô hình đoán đúng lớp **`Corn___Northern_Leaf_Blight`** (khớp với nhãn thư mục `Blight`).
  * **Dữ liệu JSON trả về thực tế:**
    ```json
    {
      "predicted_class": "Corn___Northern_Leaf_Blight",
      "confidence": 0.4409,
      "probabilities": {
        "Corn___Common_Rust": 0.3982,
        "Corn___Gray_Leaf_Spot": 0.1604,
        "Corn___Healthy": 0.0005,
        "Corn___Northern_Leaf_Blight": 0.4409
      }
    }
    ```
* **Chủ động tắt ngay Endpoint để tránh tốn tiền:** Ngay sau khi test xong kết quả, em chạy lệnh xóa ngay `predictor.delete_endpoint(delete_endpoint_config=True)` để tắt máy ảo và xóa cấu hình. Endpoint chỉ bật khoảng 5 phút để lấy số liệu rồi tắt, chi phí phát sinh ước tính dưới $0.015 USD, không để máy ảo chạy qua đêm gây tốn tiền tài khoản Learner Lab.
  * Link file notebook: [notebooks/04_endpoint_demo.ipynb](https://github.com/TBL251/Cloud/blob/main/notebooks/04_endpoint_demo.ipynb)
  * Link ảnh minh chứng gửi ảnh và đo Latency (212.39 ms): [images/notebook04/Screenshot 2026-10-25 174533.png](https://github.com/TBL251/Cloud/blob/main/images/notebook04/Screenshot%202026-10-25%20174533.png)
  * Link ảnh minh chứng xóa sạch Endpoint và Model: [images/notebook04/Screenshot 2026-10-25 174627.png](https://github.com/TBL251/Cloud/blob/main/images/notebook04/Screenshot%202026-10-25%20174627.png)

---

### 4. Xây dựng Script Dọn dẹp Tài nguyên Tự động (`scripts/teardown.py`)
Để tránh trường hợp quên tắt tài nguyên khi thao tác, em viết thêm script `scripts/teardown.py` dùng `boto3`:
* Tự động dò và xóa các Real-time Endpoints đang chạy hoặc đang khởi tạo dở.
* Quét và xóa các `EndpointConfig` cũ để không bị trùng tên khi tạo lại.
* Xóa các định nghĩa `Model` trên SageMaker registry.
* Nhờ script này, em kiểm tra được tài khoản đã tắt sạch tài nguyên sau mỗi buổi thực hành.
  * Link file script: [scripts/teardown.py](https://github.com/TBL251/Cloud/blob/main/scripts/teardown.py)

---

## 4. BẢNG COMMIT GITHUB TRONG TUẦN (KÈM LINK COMMIT CỤ THỂ)

Tuần 3 em thực hiện các commit cụ thể trên nhánh `main` để lưu lại toàn bộ tiến độ làm việc:

| STT | Mã Commit (SHA) | Nội dung Commit | Link Commit trên GitHub |
| :---: | :---: | :--- | :---: |
| 1 | `1607e51` | Triển khai mã nguồn tuần 3 gồm inference, evaluate, teardown và 2 notebook 03 và 04 | [Commit 1607e51](https://github.com/TBL251/Cloud/commit/1607e515a9f7f9d3560d9241e9137373367e3e5f) |
| 2 | `bf3085d` | Sửa lỗi TypeError tags trong PyTorchModel ở notebook 03 và 04 | [Commit bf3085d](https://github.com/TBL251/Cloud/commit/bf3085dc6c04183ede844177f8279d13ab61f190) |
| 3 | `af24ca9` | Sửa kiểu dữ liệu trả về của output_fn trong inference và cập nhật notebook 04 | [Commit af24ca9](https://github.com/TBL251/Cloud/commit/af24ca9d5a7f712210b08fb4a6437ab5d6be71c8) |
| 4 | `3c56073` | Dùng boto3 invoke_endpoint để đo độ trễ và nhận JSON trong notebook 04 | [Commit 3c56073](https://github.com/TBL251/Cloud/commit/3c56073f318a3052a806e70cfa696a0e3761b0f1) |
| 5 | `8dd9711` | Bổ sung ảnh chụp kết quả đo độ trễ latency và xóa endpoint ở notebook 04 | [Commit 8dd9711](https://github.com/TBL251/Cloud/commit/8dd9711984d1b011c15d7823e2c46320cc76f074) |

---

## 5. KHÓ KHĂN GẶP PHẢI, CÁCH GIẢI QUYẾT VÀ BÀI HỌC KINH NGHIỆM

Trong quá trình làm việc tuần 3, em gặp phải một số lỗi thực tế và đã xử lý như sau:

1. **Lỗi tham số `tags` khi tạo `PyTorchModel`:**
   * *Hiện tượng:* Khi khai báo `PyTorchModel(..., tags=[...])`, code báo lỗi `TypeError: Model.__init__() got an unexpected keyword argument 'tags'`.
   * *Nguyên nhân:* Trong thư viện SageMaker SDK v2, tham số `tags` không nằm trong hàm tạo `Model.__init__`, mà phải truyền vào lúc gọi lệnh `.deploy(tags=...)`.
   * *Cách sửa:* Em bỏ `tags` trong lúc tạo `PyTorchModel` và chuyển sang truyền ở phương thức `deploy()`.

2. **Thiếu thư viện vẽ biểu đồ trên kernel SageMaker:**
   * *Hiện tượng:* Khi chạy cell vẽ ma trận nhầm lẫn ở Notebook 03, cell báo lỗi `ModuleNotFoundError: No module named 'matplotlib'`.
   * *Nguyên nhân:* Môi trường `conda_python3` mặc định trên Notebook Instance chỉ có các gói cơ bản, chưa cài sẵn thư viện `matplotlib` và `scikit-learn`.
   * *Cách sửa:* Em thêm dòng `!pip install -q matplotlib scikit-learn` ở đầu cell để máy tự cài trước khi vẽ.

3. **Lỗi HTTP 500 khi gọi Endpoint do định dạng trả về của `output_fn`:**
   * *Hiện tượng:* Khi gửi ảnh qua Endpoint ở Notebook 04, lệnh `predict` báo lỗi `Received server error (500) from primary`.
   * *Nguyên nhân:* Trong file `inference.py`, hàm `output_fn` em trả về dạng tuple: `return json.dumps(response_payload), "application/json"`. Đối với Batch Transform job thì cơ chế cũ vẫn tự bóc tách được tuple này. Nhưng với Real-time Endpoint trên PyTorch 2.1, container dùng engine **TorchServe**, bắt buộc `output_fn` phải trả về chuỗi trực tiếp (`str`) hoặc mảng byte. Khi nhận tuple, TorchServe bị lỗi `AttributeError` khi đóng gói HTTP body và báo lỗi 500.
   * *Cách sửa:* Em sửa dòng cuối hàm `output_fn` thành `return json.dumps(response_payload, ensure_ascii=False)`. Sau khi sửa, container TorchServe phản hồi kết quả JSON bình thường với độ trễ 212 ms.

4. **Lỗi trùng tên Endpoint Configuration khi chạy lại lệnh deploy:**
   * *Hiện tượng:* Báo lỗi `ValidationException: Cannot create already existing endpoint configuration`.
   * *Nguyên nhân:* Khi chạy lại cell deploy mà chưa chạy lại cell sinh tên, biến `endpoint_name` vẫn mang timestamp cũ đã được tạo trên AWS ở lần trước.
   * *Cách sửa:* Chạy lại Cell 1 để tạo timestamp mới, hoặc chạy script `teardown.py` để xóa các config cũ bị kẹt.

5. **Bài học kinh nghiệm thực tế về chi phí:**
   * **Cách dùng Endpoint tạm thời (Transient):** Với các bài tập cần đo đạc tính năng thời gian thực, cách làm tiết kiệm nhất là chỉ bật Endpoint trong 5 - 10 phút để lấy số liệu rồi xóa ngay, chi phí chỉ tốn vài cent tiền máy ảo thay vì để máy chạy liên tục mất tiền oan.
   * **Điểm tiện lợi của Batch Transform:** Khi cần chấm điểm tập test lớn (400 ảnh), Batch Transform rất tiện và rẻ vì máy ảo tự bật lên xử lý xong là tự tắt, chạy 4 phút trên `ml.c5.xlarge` tốn chưa tới 0.01 USD.

---

## 6. KẾ HOẠCH TUẦN TIẾP THEO (TUẦN 4)
* **Tổng kết và hoàn thiện toàn bộ mã nguồn đồ án:** Rà soát lại code từ bước chuẩn bị dữ liệu, huấn luyện đến các chế độ suy luận.
* **Xây dựng bảng so sánh thực tế giữa Batch Transform và Real-time Endpoint:** Lập bảng số liệu so sánh rõ ràng về: Chi phí ($), Độ trễ (ms), Tốc độ xử lý (ảnh/phút) và Trường hợp nào nên dùng trong nông nghiệp.
* **Viết Báo cáo tổng kết đồ án cuối kỳ** theo đúng hướng dẫn và biểu mẫu của giảng viên.
