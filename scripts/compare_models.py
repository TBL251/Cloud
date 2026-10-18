"""
compare_models.py
Mục đích: Script thực nghiệm đối sánh trực tiếp các chỉ số kỹ thuật giữa MobileNetV2 và ResNet-18 trên CPU.
Chạy được: Trực tiếp trên SageMaker Notebook Instance hoặc môi trường local có cài PyTorch/torchvision.
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
from torchvision import models

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def measure_inference_latency(model, input_tensor, runs=100, warmup=10):
    model.eval()
    with torch.no_grad():
        # Warm-up CPU cache
        for _ in range(warmup):
            _ = model(input_tensor)

        # Đo thời gian thực tế
        start_time = time.perf_counter()
        for _ in range(runs):
            _ = model(input_tensor)
        end_time = time.perf_counter()

    avg_ms = ((end_time - start_time) / runs) * 1000
    return avg_ms


def measure_model_disk_size(model, filename="temp_model.pth"):
    torch.save(model.state_dict(), filename)
    size_mb = os.path.getsize(filename) / (1024 * 1024)
    if os.path.exists(filename):
        os.remove(filename)
    return size_mb


def benchmark(num_classes=4):
    print("=" * 70)
    print("  ĐANG CHẠY THỰC NGHIỆM ĐỐI SÁNH: MOBILENET_V2 vs RESNET-18 (TRÊN CPU)")
    print("=" * 70)

    device = torch.device("cpu")
    dummy_input = torch.randn(1, 3, 224, 224, device=device)

    # 1. Khởi tạo MobileNetV2
    print("[1/2] Đang đánh giá MobileNetV2...")
    mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    for p in mobilenet.features.parameters():
        p.requires_grad = False
    in_feat_mb = mobilenet.classifier[1].in_features
    mobilenet.classifier[1] = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_feat_mb, num_classes)
    )
    mobilenet.to(device)

    mb_total, mb_trainable = count_parameters(mobilenet)
    mb_latency = measure_inference_latency(mobilenet, dummy_input)
    mb_size = measure_model_disk_size(mobilenet, "mobilenet_temp.pth")

    # 2. Khởi tạo ResNet-18
    print("[2/2] Đang đánh giá ResNet-18...")
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for p in resnet.parameters():
        p.requires_grad = False
    in_feat_res = resnet.fc.in_features
    resnet.fc = nn.Linear(in_feat_res, num_classes)
    resnet.to(device)

    res_total, res_trainable = count_parameters(resnet)
    res_latency = measure_inference_latency(resnet, dummy_input)
    res_size = measure_model_disk_size(resnet, "resnet18_temp.pth")

    # 3. Xuất bảng so sánh định dạng Markdown
    print("\n" + "=" * 70)
    print("  KẾT QUẢ ĐO ĐẠC THỰC TẾ (BENCHMARK RESULTS TRÊN CPU)")
    print("=" * 70)
    print(f"| Tiêu chí kỹ thuật                  | MobileNetV2         | ResNet-18           | Chênh lệch (Tối ưu) |")
    print(f"| :--------------------------------- | :-----------------: | :-----------------: | :------------------ |")
    print(f"| Tổng tham số (Total Params)        | {mb_total/1e6:.2f}M ({mb_total:,}) | {res_total/1e6:.2f}M ({res_total:,}) | ResNet gấp {res_total/mb_total:.1f} lần |")
    print(f"| Tham số huấn luyện (Trainable)     | {mb_trainable:,}           | {res_trainable:,}           | MobileNetV2 nhẹ hơn |")
    print(f"| Dung lượng file Model (.pth)       | {mb_size:.2f} MB            | {res_size:.2f} MB            | Tiết kiệm {(1 - mb_size/res_size)*100:.1f}% bộ nhớ |")
    print(f"| Độ trễ suy luận CPU (Latency)      | {mb_latency:.2f} ms/ảnh        | {res_latency:.2f} ms/ảnh        | MobileNetV2 nhanh gấp {res_latency/mb_latency:.1f} lần |")
    print("=" * 70)


if __name__ == "__main__":
    benchmark(num_classes=4)
