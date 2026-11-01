# Phân Loại Bệnh Lá Cây Sử Dụng Transfer Learning Trên AWS SageMaker

**Môn học:** Điện toán đám mây (Cloud) - Đợt 1 - 2026-2027  
**Giảng viên hướng dẫn (GVHD):** Huỳnh Xuân Phụng  
**Sinh viên thực hiện:** Lê Bá Tâm - MSSV: 24110055 - Lớp: CLCO332779_05UTExMC  
**Video Demo Đồ Án:** [Xem Video Demo trên Google Drive (~9 phút)](https://drive.google.com/file/d/1dhWreqLEgqtqwZTe-_w1gkN2HBoCfJsX/view?usp=sharing)  

---

## 📌 Báo Cáo Tiến Độ Hàng Tuần
* [Báo Cáo Tiến Độ Tuần 1 (docs/BAO_CAO_TUAN_1.md)](docs/BAO_CAO_TUAN_1.md) - Chuẩn bị môi trường & 1.600 ảnh dataset PlantVillage trên S3
* [Báo Cáo Tiến Độ Tuần 2 (docs/BAO_CAO_TUAN_2.md)](docs/BAO_CAO_TUAN_2.md) - Huấn luyện MobileNetV2 (Acc 90.00%) & Đối sánh ResNet-18 trên `ml.c5.xlarge`
* [Báo Cáo Tiến Độ Tuần 3 (docs/BAO_CAO_TUAN_3.md)](docs/BAO_CAO_TUAN_3.md) - Batch Transform 400 ảnh (Acc 90%), Confusion Matrix & Transient Endpoint (212.39 ms)
* [Báo Cáo Tiến Độ Tuần 4 (docs/BAO_CAO_TUAN_4.md)](docs/BAO_CAO_TUAN_4.md) - Đối chuẩn kỹ thuật, Nghiệm thu ngân sách ($18.1 / $50) & Video Demo

---

## 📂 Cấu Trúc Thư Mục Dự Án

```text
e:/Cloud/
├── architecture/
│   └── README.md               # Sơ đồ và tài liệu kiến trúc hệ thống
├── data/
│   ├── sample_images/          # Ảnh mẫu test suy luận
│   │   └── README.md
│   └── class_labels.json       # Tên nhãn lớp bệnh lá cây
├── src/
│   ├── prepare_data.py         # Lọc, tiền xử lý và upload <= 2000 ảnh lên S3
│   ├── train.py                # Mã nguồn PyTorch Transfer Learning chạy trong SageMaker
│   ├── inference.py            # Custom handler phục vụ endpoint (model_fn, predict_fn)
│   └── evaluate.py             # Script vẽ Confusion Matrix, tính Accuracy, F1-Score
├── notebooks/
│   ├── 01_data_preparation.ipynb     # Chuẩn bị dữ liệu và khám phá dataset
│   ├── 02_sagemaker_training.ipynb   # Kích hoạt PyTorch Estimator Script Mode
│   ├── 03_batch_transform.ipynb     # Chạy Batch Transform Job đánh giá
│   └── 04_endpoint_demo.ipynb       # Deploy endpoint tạm thời, test & xóa endpoint
├── scripts/
│   ├── teardown.py             # Script tự động scan & dọn dẹp tài nguyên tránh tốn tiền
│   └── check_quota.py          # Kiểm tra quota và tagging trên AWS
├── docs/
│   ├── BAO_CAO_TUAN_1.md       # Báo cáo tiến độ tuần 1
│   ├── BAO_CAO_TUAN_2.md       # Báo cáo tiến độ tuần 2
│   ├── BAO_CAO_TUAN_3.md       # Báo cáo tiến độ tuần 3
│   ├── BAO_CAO_TUAN_4.md       # Báo cáo tiến độ tuần 4
│   ├── PROJECT_PLAN_RECOMMENDED.md  # Kế hoạch chi tiết 4 tuần theo phương án tối ưu
│   └── worklogs/
│       ├── week_1_worklog.md   # Nhật ký công việc Tuần 1
│       ├── week_2_worklog.md   # Nhật ký công việc Tuần 2
│       ├── week_3_worklog.md   # Nhật ký công việc Tuần 3
│       └── week_4_worklog.md   # Nhật ký công việc Tuần 4
└── README.md
```
