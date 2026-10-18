## Cấu trúc Thư mục Dự án

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
│   ├── PROJECT_PLAN_RECOMMENDED.md  # Kế hoạch chi tiết 4 tuần theo phương án tối ưu
│   └── worklogs/
│       ├── week_1_worklog.md   # Nhật ký công việc Tuần 1
│       ├── week_2_worklog.md   # Nhật ký công việc Tuần 2
│       ├── week_3_worklog.md   # Nhật ký công việc Tuần 3
│       └── week_4_worklog.md   # Nhật ký công việc Tuần 4
└── README.md
```
