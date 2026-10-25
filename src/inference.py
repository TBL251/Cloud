"""
inference.py
Mục đích: Script phục vụ suy luận (Serving Handler) trên AWS SageMaker (PyTorch Serving).
Tương thích: Real-time Endpoint và SageMaker Batch Transform Job.
Kiến trúc: MobileNetV2 (Hỗ trợ nạp model_meta.json và class labels tự động).
"""

import os
import io
import json
import logging
import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

DEFAULT_CLASSES = ["Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"]


def get_inference_transforms():
    """Chuẩn hóa ảnh đầu vào chuẩn ImageNet 224x224 cho suy luận."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def model_fn(model_dir):
    """
    Nạp mô hình từ model artifact (model.pth) được SageMaker giải nén vào model_dir.
    Được gọi 1 lần duy nhất khi container khởi động.
    """
    logger.info(f"[model_fn] Bắt đầu nạp mô hình từ thư mục: {model_dir}")
    device = torch.device("cpu")

    # 1. Đọc metadata nếu có
    meta_path = os.path.join(model_dir, "model_meta.json")
    model_name = "mobilenet_v2"
    classes = DEFAULT_CLASSES

    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                model_name = meta.get("model_name", "mobilenet_v2")
                classes = meta.get("classes", DEFAULT_CLASSES)
                logger.info(f"[model_fn] Đã nạp metadata: model={model_name}, classes={classes}")
        except Exception as e:
            logger.warning(f"[model_fn] Không thể đọc metadata: {e}. Sử dụng cấu hình mặc định.")

    num_classes = len(classes)

    # 2. Xây dựng khung kiến trúc mô hình
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
        raise ValueError(f"Kiến trúc không được hỗ trợ: {model_name}")

    # 3. Nạp trọng số từ model.pth
    model_path = os.path.join(model_dir, "model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy file trọng số model.pth tại: {model_dir}")

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Gán danh sách lớp vào đối tượng model để predict_fn sử dụng
    model.classes = classes
    logger.info(f"[model_fn] Nạp mô hình {model_name} thành công với {num_classes} lớp nhận diện.")
    return model


def input_fn(request_body, request_content_type):
    """
    Tiền xử lý dữ liệu đầu vào từ HTTP Request (Endpoint) hoặc file ảnh (Batch Transform).
    Hỗ trợ định dạng ảnh JPEG, PNG, x-image, octet-stream hoặc JSON base64.
    """
    logger.info(f"[input_fn] Nhận request với Content-Type: {request_content_type}")
    transform = get_inference_transforms()

    if request_content_type in ["image/jpeg", "image/png", "image/jpg", "application/x-image", "application/octet-stream"]:
        image = Image.open(io.BytesIO(request_body)).convert("RGB")
        tensor = transform(image).unsqueeze(0)  # Shape: [1, 3, 224, 224]
        return tensor

    elif request_content_type == "application/json":
        data = json.loads(request_body)
        if "image_base64" in data:
            import base64
            img_bytes = base64.b64decode(data["image_base64"])
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            tensor = transform(image).unsqueeze(0)
            return tensor
        elif "instances" in data:
            # Dạng mảng tensor đã chuẩn hóa
            return torch.tensor(data["instances"], dtype=torch.float32)
        else:
            raise ValueError("JSON request cần chứa key 'image_base64' hoặc 'instances'")

    else:
        raise ValueError(f"Content-Type không được hỗ trợ: {request_content_type}")


def predict_fn(input_data, model):
    """
    Thực hiện suy luận dự đoán nhãn bệnh trên CPU mà không tính gradient.
    """
    logger.info(f"[predict_fn] Thực hiện dự đoán trên tensor kích thước: {input_data.shape}")
    with torch.no_grad():
        outputs = model(input_data)
        probabilities = torch.softmax(outputs, dim=1)
        confidences, predicted_indices = torch.max(probabilities, dim=1)

    return {
        "probabilities": probabilities.tolist(),
        "confidences": confidences.tolist(),
        "predicted_indices": predicted_indices.tolist(),
        "classes": getattr(model, "classes", DEFAULT_CLASSES)
    }


def output_fn(prediction, accept):
    """
    Đóng gói kết quả dự đoán trả về cho client theo định dạng yêu cầu (JSON).
    """
    logger.info(f"[output_fn] Đóng gói kết quả trả về với Accept: {accept}")

    classes = prediction["classes"]
    results = []

    for idx, conf, probs in zip(prediction["predicted_indices"], prediction["confidences"], prediction["probabilities"]):
        class_name = classes[idx]
        class_probs = {classes[i]: round(float(probs[i]), 4) for i in range(len(classes))}
        results.append({
            "predicted_class": class_name,
            "confidence": round(float(conf), 4),
            "probabilities": class_probs
        })

    # Nếu chỉ có 1 ảnh thì trả về object, nếu nhiều ảnh thì trả về list
    response_payload = results[0] if len(results) == 1 else results
    return json.dumps(response_payload, ensure_ascii=False), "application/json"
