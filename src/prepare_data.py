"""
prepare_data.py
Mục đích: Lọc, resize 224x224, chia train/val (80/20) và upload dữ liệu lên Amazon S3.
Đề tài: Phân loại bệnh lá cây bằng Transfer Learning trên SageMaker
Ràng buộc: Bộ dữ liệu <= 2.000 ảnh, gắn thẻ Tag Project & Owner.
"""

import os
import sys
import shutil
import random
import argparse
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

try:
    from PIL import Image
    import boto3
    import numpy as np
except ImportError as e:
    print(f"[LỖI] Thiếu thư viện: {e}")
    print("Vui lòng chạy: pip install pillow boto3 numpy")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Tiền xử lý và upload dataset lên S3")
    parser.add_argument("--source_dir", type=str, default="./raw_dataset", help="Thư mục ảnh gốc")
    parser.add_argument("--output_dir", type=str, default="./processed_dataset", help="Thư mục sau tiền xử lý")
    parser.add_argument("--s3_bucket", type=str, default="", help="Tên S3 Bucket đích")
    parser.add_argument("--max_total_images", type=int, default=1600, help="Giới hạn tối đa ảnh (<= 2000)")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Tỉ lệ tập huấn luyện (80%%)")
    parser.add_argument("--image_size", type=int, default=224, help="Kích thước ảnh sau resize (224x224)")
    parser.add_argument("--project_tag", type=str, default="PlantLeafClassification", help="Giá trị thẻ tag Project")
    parser.add_argument("--owner_tag", type=str, default="LeBaTam", help="Giá trị thẻ tag Owner")
    return parser.parse_args()


def process_and_split_dataset(source_dir, output_dir, max_total_images=1600, train_ratio=0.8, image_size=224):
    """Lọc dữ liệu đảm bảo <= 2000 ảnh, resize và chia thư mục train / val."""
    source_path = Path(source_dir)
    output_path = Path(output_dir)

    train_path = output_path / "train"
    val_path = output_path / "val"

    if output_path.exists():
        shutil.rmtree(output_path)
    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)

    classes = [d.name for d in source_path.iterdir() if d.is_dir()]
    if not classes:
        print(f"[CẢNH BÁO] Không tìm thấy thư mục class nào trong: {source_dir}")
        return 0

    max_per_class = max_total_images // len(classes)
    total_processed = 0

    print("=" * 60)
    print("  BẮT ĐẦU TIỀN XỬ LÝ DỮ LIỆU ẢNH LÁ CÂY")
    print("=" * 60)
    print(f"Số lượng lớp nhãn: {len(classes)} ({classes})")
    print(f"Giới hạn tối đa mỗi lớp: {max_per_class} ảnh (Tổng <= {max_total_images} ảnh)")

    for cls in classes:
        cls_source = source_path / cls
        (train_path / cls).mkdir(parents=True, exist_ok=True)
        (val_path / cls).mkdir(parents=True, exist_ok=True)

        images = [f for f in cls_source.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        random.seed(42)
        random.shuffle(images)

        selected_images = images[:max_per_class]
        split_idx = int(len(selected_images) * train_ratio)
        train_imgs = selected_images[:split_idx]
        val_imgs = selected_images[split_idx:]

        for img_path in train_imgs:
            try:
                with Image.open(img_path) as img:
                    img_resized = img.convert("RGB").resize((image_size, image_size))
                    img_resized.save(train_path / cls / img_path.name, "JPEG", quality=90)
                    total_processed += 1
            except Exception as e:
                print(f"Lỗi đọc ảnh {img_path}: {e}")

        for img_path in val_imgs:
            try:
                with Image.open(img_path) as img:
                    img_resized = img.convert("RGB").resize((image_size, image_size))
                    img_resized.save(val_path / cls / img_path.name, "JPEG", quality=90)
                    total_processed += 1
            except Exception as e:
                print(f"Lỗi đọc ảnh {img_path}: {e}")

        print(f"  + Lớp '{cls}': {len(train_imgs)} ảnh train, {len(val_imgs)} ảnh val")

    print(f"\n=> TỔNG SỐ ẢNH ĐÃ XỬ LÝ: {total_processed} ảnh (Thỏa mãn ràng buộc <= 2.000 ảnh).")
    return total_processed


def upload_to_s3(local_dir, bucket_name, project_tag="PlantLeafClassification", owner_tag="LeBaTam"):
    """Tải thư mục train/val lên S3 với các Tag quy chuẩn."""
    if not bucket_name:
        print("[INFO] Chưa cung cấp --s3_bucket, bỏ qua bước upload.")
        return

    s3_client = boto3.client("s3")
    print(f"\n=== ĐANG UPLOAD DỮ LIỆU LÊN S3: s3://{bucket_name}/dataset/ ===")

    uploaded_count = 0
    for root, _, files in os.walk(local_dir):
        for file in files:
            local_file_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_file_path, local_dir)
            s3_key = f"dataset/{relative_path}".replace("\\", "/")

            s3_client.upload_file(
                local_file_path,
                bucket_name,
                s3_key,
                ExtraArgs={"Tagging": f"Project={project_tag}&Owner={owner_tag}"}
            )
            uploaded_count += 1
            if uploaded_count % 100 == 0:
                print(f"  -> Đã upload {uploaded_count} ảnh...")

    print(f"=> Hoàn tất upload {uploaded_count} ảnh lên S3 Bucket: {bucket_name}")


def generate_sample_dataset(source_dir, num_samples_per_class=100):
    """Tự động sinh dataset mẫu cho 4 lớp nếu sinh viên chưa kịp tải PlantVillage gốc."""
    source_path = Path(source_dir)
    classes = [
        "Corn___Healthy",
        "Corn___Common_Rust",
        "Corn___Gray_Leaf_Spot",
        "Corn___Northern_Leaf_Blight"
    ]
    color_map = {
        "Corn___Healthy": (34, 139, 34),          # Xanh lá tươi
        "Corn___Common_Rust": (178, 34, 34),       # Đỏ gỉ sắt
        "Corn___Gray_Leaf_Spot": (128, 128, 128),  # Xám đốm lá
        "Corn___Northern_Leaf_Blight": (160, 82, 45) # Nâu cháy lá
    }

    print(f"[INFO] Tự động sinh dữ liệu lá cây mẫu tại '{source_dir}' ({num_samples_per_class} ảnh/lớp)...")
    for cls in classes:
        cls_dir = source_path / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        base_color = color_map.get(cls, (50, 150, 50))

        for i in range(num_samples_per_class):
            # Tạo ảnh có biến thiên màu sắc và họa tiết mô phỏng lá
            import numpy as np
            arr = np.full((224, 224, 3), base_color, dtype=np.uint8)
            noise = np.random.randint(-25, 25, (224, 224, 3), dtype=np.int16)
            arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            img = Image.fromarray(arr)
            img.save(cls_dir / f"leaf_sample_{i+1:04d}.jpg", "JPEG", quality=90)

    print(f"=> Đã tạo dữ liệu mẫu cho 4 lớp thành công!")


if __name__ == "__main__":
    args = parse_args()
    source_p = Path(args.source_dir)
    images_found = list(source_p.rglob("*.jpg")) + list(source_p.rglob("*.jpeg")) + list(source_p.rglob("*.png")) if source_p.exists() else []
    if not source_p.exists() or len(images_found) == 0:
        print(f"[THÔNG BÁO] Thư mục '{args.source_dir}' chưa có ảnh hợp lệ nào.")
        generate_sample_dataset(args.source_dir, num_samples_per_class=100)

    process_and_split_dataset(args.source_dir, args.output_dir, args.max_total_images, args.train_ratio, args.image_size)
    if args.s3_bucket:
        upload_to_s3(args.output_dir, args.s3_bucket, args.project_tag, args.owner_tag)
