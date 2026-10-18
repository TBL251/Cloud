"""
check_quota.py
Mục đích: Kiểm tra môi trường kết nối AWS Learner Lab:
- Xác nhận Account ID & Region (us-east-1 / us-west-2)
- Xác nhận quyền LabRole
- Kiểm tra các S3 Bucket hiện có
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

try:
    import boto3
except ImportError:
    print("[THONG BAO] Chua cai dat boto3 tren moi truong Python local.")
    print("Ban co the chay: pip install boto3")
    print("Hoac mo file nay chay truc tiep ben trong SageMaker Notebook (noi boto3 da co san).")
    sys.exit(0)

def check_aws_env():
    print("=" * 65)
    print("  AWS LEARNER LAB PRE-FLIGHT VERIFICATION (TUẦN 2)")
    print("=" * 65)

    # 1. Kiểm tra STS Identity
    try:
        sts = boto3.client("sts")
        identity = sts.get_caller_identity()
        account_id = identity["Account"]
        arn = identity["Arn"]
        print(f"[1] Account ID : {account_id}")
        print(f"    Caller ARN : {arn}")
    except Exception as e:
        print(f"[LỖI] Không thể kết nối AWS CLI / Credentials: {e}")
        print("Vui lòng cấu hình file ~/.aws/credentials từ nút 'AWS Details' trong Learner Lab.")
        return None

    # 2. Kiểm tra Region
    session = boto3.session.Session()
    region = session.region_name or "us-east-1"
    print(f"\n[2] Current Region : {region}")
    if region in ["us-east-1", "us-west-2"]:
        print(f"    -> Region HỢP LỆ theo quy định của môn học ({region}).")
    else:
        print(f"    -> [CẢNH BÁO]: Đề bài yêu cầu chỉ dùng us-east-1 hoặc us-west-2!")

    # 3. Kiểm tra LabRole
    lab_role_arn = f"arn:aws:iam::{account_id}:role/LabRole"
    print(f"\n[3] Target LabRole : {lab_role_arn}")
    try:
        iam = boto3.client("iam")
        role_info = iam.get_role(RoleName="LabRole")
        print("    -> LabRole TỒN TẠI và sẵn sàng gán cho SageMaker.")
    except Exception as e:
        print(f"    -> Lưu ý: Không thể query IAM trực tiếp ({e}), nhưng LabRole ARN vẫn dùng được cho SageMaker.")

    # 4. Kiểm tra S3 Buckets
    try:
        s3 = boto3.client("s3", region_name=region)
        buckets = s3.list_buckets().get("Buckets", [])
        print(f"\n[4] Danh sách S3 Buckets hiện có ({len(buckets)}):")
        for b in buckets:
            print(f"    - {b['Name']}")
    except Exception as e:
        print(f"    -> Lỗi liệt kê S3: {e}")

    print("\n" + "=" * 65)
    print("  HOÀN TẤT KIỂM TRA MÔI TRƯỜNG AWS")
    print("=" * 65)
    return account_id, region

if __name__ == "__main__":
    check_aws_env()
