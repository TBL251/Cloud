"""
evaluate.py
Mục đích: Đánh giá mô hình phân loại bệnh lá cây và xuất ma trận nhầm lẫn (Confusion Matrix).
Đầu ra:
  - Bảng Classification Report (Precision, Recall, F1-Score cho từng lớp).
  - Biểu đồ nhiệt ma trận nhầm lẫn (Confusion Matrix Heatmap) lưu vào thư mục images/.
  - File JSON lưu các chỉ số đánh giá kỹ thuật.
"""

import os
import sys
import json
import logging
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np

# Thư viện vẽ biểu đồ
import matplotlib
matplotlib.use("Agg")  # Chạy headless không cần màn hình hiển thị
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def load_evaluation_model(model_path, num_classes=4, model_name="mobilenet_v2", device="cpu"):
    """Nạp trọng số mô hình đã huấn luyện từ file .pth."""
    logger.info(f"Đang nạp mô hình {model_name} từ: {model_path}")
    
    if model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )
    elif model_name == "resnet18":
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Kiến trúc không hỗ trợ: {model_name}")

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def get_val_dataloader(val_dir, batch_size=32):
    """Khởi tạo DataLoader cho tập kiểm thử (Validation set)."""
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return val_loader, val_dataset.classes


def plot_confusion_matrix(cm, class_names, output_image_path):
    """Vẽ và lưu biểu đồ Confusion Matrix Heatmap trực quan chuẩn báo cáo."""
    plt.figure(figsize=(8, 6), dpi=150)
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - MobileNetV2 (PlantVillage)", fontsize=14, fontweight="bold", pad=15)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=30, ha="right", fontsize=10)
    plt.yticks(tick_marks, class_names, fontsize=10)

    # Hiển thị số lượng dự đoán trong từng ô
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > thresh else "black"
            plt.text(j, i, f"{val:d}", ha="center", va="center", color=color, fontsize=12, fontweight="bold")

    plt.ylabel("Nhãn thực tế (True Label)", fontsize=11, fontweight="bold")
    plt.xlabel("Nhãn dự đoán (Predicted Label)", fontsize=11, fontweight="bold")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    plt.savefig(output_image_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Đã lưu biểu đồ Confusion Matrix tại: {output_image_path}")


def evaluate(val_dir="data_split/val", model_path="model.pth", output_dir="images"):
    """Thực thi quy trình đánh giá toàn diện."""
    device = torch.device("cpu")

    # 1. Nạp dữ liệu
    val_loader, class_names = get_val_dataloader(val_dir)
    logger.info(f"Tập kiểm thử có {len(val_loader.dataset)} ảnh thuộc {len(class_names)} lớp: {class_names}")

    # 2. Nạp mô hình
    model = load_evaluation_model(model_path, num_classes=len(class_names), device=device)

    # 3. Chạy suy luận trên toàn bộ tập kiểm thử
    all_preds = []
    all_targets = []

    logger.info("Đang thực hiện suy luận trên tập kiểm thử...")
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    # 4. Tính toán Confusion Matrix và Report
    cm = confusion_matrix(all_targets, all_preds)
    report = classification_report(all_targets, all_preds, target_names=class_names, output_dict=True)

    print("\n" + "=" * 65)
    print("  BÁO CÁO ĐÁNH GIÁ MÔ HÌNH (CLASSIFICATION REPORT)")
    print("=" * 65)
    print(classification_report(all_targets, all_preds, target_names=class_names))
    print("=" * 65)

    # 5. Lưu biểu đồ và file metrics
    cm_image_path = os.path.join(output_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, cm_image_path)

    metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "confusion_matrix": cm.tolist(),
            "classes": class_names,
            "classification_report": report
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"Đã lưu các chỉ số chi tiết tại: {metrics_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Leaf Disease Classification Model")
    parser.add_argument("--val_dir", type=str, default="data_split/val", help="Thư mục dữ liệu validation")
    parser.add_argument("--model_path", type=str, default="model.pth", help="Đường dẫn file trọng số model.pth")
    parser.add_argument("--output_dir", type=str, default="images", help="Thư mục lưu ảnh Confusion Matrix")
    args = parser.parse_args()

    if os.path.exists(args.val_dir) and os.path.exists(args.model_path):
        evaluate(args.val_dir, args.model_path, args.output_dir)
    else:
        logger.info("Chạy ở chế độ script đánh giá. Sử dụng trong Notebook 03 hoặc chạy độc lập khi có file model.")
