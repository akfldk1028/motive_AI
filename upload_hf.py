from huggingface_hub import HfApi
import os
import hashlib

# Hugging Face 토큰 설정
os.environ["HUGGINGFACE_TOKEN"] = "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"

# API 객체 생성
api = HfApi()

# 로컬 모델 경로와 Hugging Face 저장소 이름 설정
local_model_path = "./models"
repo_name = "Ammunity/motive_v1"

# 저장소의 현재 파일 목록 가져오기
try:
    existing_files = set(api.list_repo_files(repo_id=repo_name, repo_type="model"))
except Exception as e:
    print(f"Error fetching existing files: {str(e)}")
    existing_files = set()

# 로그 파일 설정
log_file = "upload_log.txt"

# models 폴더와 그 내용 업로드
with open(log_file, "a") as log:
    for root, dirs, files in os.walk(local_model_path):
        for file in files:
            local_path = os.path.join(root, file)
            # models 폴더를 포함한 상대 경로 계산
            relative_path = os.path.relpath(local_path, local_model_path)

            # 이미 업로드된 파일 건너뛰기
            if relative_path in existing_files:
                print(f"Skipping existing file: {relative_path}")
                log.write(f"Skipped: {relative_path}\n")
                continue

            try:
                with open(local_path, "rb") as file_obj:
                    api.upload_file(
                        path_or_fileobj=file_obj,
                        path_in_repo=relative_path,
                        repo_id=repo_name,
                        repo_type="model"
                    )
                print(f"Uploaded: {relative_path}")
                log.write(f"Uploaded: {relative_path}\n")
            except Exception as e:
                print(f"Error uploading {relative_path}: {str(e)}")
                log.write(f"Error uploading {relative_path}: {str(e)}\n")

print("업로드 프로세스가 완료되었습니다.")



# git remote -v
# git remote set-url origin <새로운_URL>
# git remote set-url origin https://github.com/username/new-repository.git
# git remote -v
