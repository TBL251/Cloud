"""
train.py
Mục đích: Script huấn luyện Transfer Learning (MobileNetV2) chạy trong SageMaker PyTorch Container.
Đề tài: Phân loại bệnh lá cây bằng Transfer Learning trên SageMaker
Ràng buộc: Huấn luyện trên CPU instance (ml.c5.xlarge / ml.m5.xlarge) trong AWS Learner Lab.
"""

import os
import sys
import json
import time
import argparse
import logging

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# Logger chuẩn SageMaker
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def parse_args():
    parser = argparse.ArgumentParser(description="SageMaker PyTorch Transfer Learning Training")

    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=8, help="Số epochs huấn luyện (8-10 epochs tối ưu cho CPU)")
    parser.add_argument("--batch_size", type=int, default=32, help="Kích thước batch")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Tốc độ học (learning rate)")
    parser.add_argument("--model_name", type=str, default="mobilenet_v2", help="Mô hình backbone: mobilenet_v2 hoặc resnet18")

    # Đường dẫn môi trường chuẩn do SageMaker cung cấp
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    parser.add_argument("--val", type=str, default=os.environ.get("SM_CHANNEL_VAL", "/opt/ml/input/data/val"))

    return parser.parse_args()


def get_data_loaders(train_dir, val_dir, batch_size):
    """Khởi tạo DataLoader với Data Augmentation và Normalization ImageNet chuẩn."""
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, train_dataset.classes


def build_model(model_name, num_classes):
    """Khởi tạo mô hình Transfer Learning và đóng băng các tầng trích xuất đặc trưng."""
    logger.info(f"Đang khởi tạo kiến trúc: {model_name} với {num_classes} nhãn phân loại")

    if model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        # Đóng băng feature extractor
        for param in model.features.parameters():
            param.requires_grad = False
        # Thay thế classifier cuối
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )
    elif model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Mô hình không được hỗ trợ: {model_name}")

    return model


def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device):
    """Vòng lặp huấn luyện và đánh giá trên CPU."""
    best_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        corrects = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data).item()
            total += labels.size(0)

        epoch_train_loss = running_loss / total
        epoch_train_acc = corrects / total

        # Đánh giá tập Validation
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data).item()
                val_total += labels.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_corrects / val_total
        elapsed = time.time() - start_time

        logger.info(
            f"Epoch [{epoch}/{epochs}] ({elapsed:.1f}s) - "
            f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}"
        )

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc

    logger.info(f"Huấn luyện hoàn tất! Best Validation Accuracy: {best_acc:.4f}")
    return model, history


def main():
    args = parse_args()
    logger.info(f"Khởi chạy huấn luyện với tham số: {args}")

    # Chạy trên CPU trong môi trường AWS Learner Lab
    device = torch.device("cpu")
    logger.info(f"Thiết bị tính toán: {device}")

    # 1. Nạp dữ liệu
    train_loader, val_loader, class_names = get_data_loaders(args.train, args.val, args.batch_size)
    num_classes = len(class_names)
    logger.info(f"Danh sách các lớp nhận dạng: {class_names}")

    # 2. Xây dựng mô hình
    model = build_model(args.model_name, num_classes).to(device)

    # 3. Định nghĩa Loss và Optimizer
    criterion = nn.CrossEntropyLoss()
    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params_to_update, lr=args.learning_rate)

    # 4. Huấn luyện
    model, history = train_model(model, train_loader, val_loader, criterion, optimizer, args.epochs, device)

    # 5. Lưu Model Artifacts vào /opt/ml/model
    os.makedirs(args.model_dir, exist_ok=True)
    model_save_path = os.path.join(args.model_dir, "model.pth")
    torch.save(model.state_dict(), model_save_path)

    meta_save_path = os.path.join(args.model_dir, "model_meta.json")
    with open(meta_save_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": args.model_name,
            "classes": class_names,
            "history": history
        }, f, indent=2)

    logger.info(f"Đã lưu model artifact thành công vào: {args.model_dir}")


if __name__ == "__main__":
    main()
