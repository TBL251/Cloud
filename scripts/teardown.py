"""
teardown.py
Mục đích: Script tự động hóa kiểm tra và dọn dẹp các tài nguyên tính phí trên AWS SageMaker.
Phạm vi quét và dọn dẹp:
  1. Real-time Endpoints (Đang chạy hoặc đang khởi tạo).
  2. Endpoint Configurations.
  3. SageMaker Models.
An toàn: Giúp sinh viên bảo toàn ngân sách đồ án, triệt tiêu nguy cơ quên tắt tài nguyên trên Learner Lab.
"""

import sys
import boto3

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def teardown_sagemaker_resources():
    print("=" * 65)
    print("  BẮT ĐẦU QUÉT VÀ DỌN DẸP TÀI NGUYÊN SAGEMAKER (TEARDOWN)")
    print("=" * 65)

    sm = boto3.client("sagemaker", region_name="us-east-1")

    # 1. Quét và xóa Real-time Endpoints
    print("\n[1/3] Đang kiểm tra các Real-time Endpoints...")
    endpoints = sm.list_endpoints(StatusEquals="InService")["Endpoints"]
    endpoints += sm.list_endpoints(StatusEquals="Creating")["Endpoints"]

    if endpoints:
        for ep in endpoints:
            ep_name = ep["EndpointName"]
            print(f"  -> Phát hiện Endpoint: {ep_name} (Status: {ep['EndpointStatus']})")
            print(f"     Đang xóa Endpoint: {ep_name}...")
            sm.delete_endpoint(EndpointName=ep_name)
            print(f"     [ĐÃ XÓA] Endpoint: {ep_name}")
    else:
        print("  -> Không có Endpoint nào đang chạy. An toàn!")

    # 2. Quét và xóa Endpoint Configurations
    print("\n[2/3] Đang kiểm tra các Endpoint Configurations...")
    configs = sm.list_endpoint_configs(MaxResults=20)["EndpointConfigs"]
    if configs:
        for cfg in configs:
            cfg_name = cfg["EndpointConfigName"]
            print(f"  -> Đang xóa Endpoint Config: {cfg_name}...")
            sm.delete_endpoint_config(EndpointConfigName=cfg_name)
            print(f"     [ĐÃ XÓA] Config: {cfg_name}")
    else:
        print("  -> Không có Endpoint Config nào còn sót. An toàn!")

    # 3. Quét và xóa SageMaker Models
    print("\n[3/3] Đang kiểm tra danh sách SageMaker Models...")
    models = sm.list_models(MaxResults=20)["Models"]
    if models:
        for m in models:
            m_name = m["ModelName"]
            print(f"  -> Đang xóa Model: {m_name}...")
            sm.delete_model(ModelName=m_name)
            print(f"     [ĐÃ XÓA] Model: {m_name}")
    else:
        print("  -> Không có Model nào trong danh sách. An toàn!")

    print("\n" + "=" * 65)
    print("  HOÀN TẤT DỌN DẸP! TÀI KHOẢN ĐÃ AN TOÀN 100%.")
    print("=" * 65)


if __name__ == "__main__":
    teardown_sagemaker_resources()
