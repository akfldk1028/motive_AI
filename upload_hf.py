from huggingface_hub import HfApi
import os

# Hugging Face 토큰 설정
os.environ["HUGGINGFACE_TOKEN"] = "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"

# API 객체 생성
api = HfApi()

# 로컬 모델 경로와 Hugging Face 저장소 이름 설정
local_model_path = "./models"
repo_name = "Ammunity/motive"

# 저장소의 현재 파일 목록 가져오기
existing_files = set(api.list_repo_files(repo_id=repo_name, repo_type="model"))

# models 폴더와 그 내용 업로드
for root, dirs, files in os.walk(local_model_path):
    for file in files:
        local_path = os.path.join(root, file)
        # models 폴더를 포함한 상대 경로 계산
        relative_path = os.path.relpath(local_path, os.path.dirname(local_model_path))

        # 이미 업로드된 파일 건너뛰기
        if relative_path in existing_files:
            print(f"Skipping existing file: {relative_path}")
            continue

        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=relative_path,
                repo_id=repo_name,
                repo_type="model"
            )
            print(f"Uploaded: {relative_path}")
        except Exception as e:
            print(f"Error uploading {relative_path}: {str(e)}")

print("업로드 프로세스가 완료되었습니다.")