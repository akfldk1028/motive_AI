from huggingface_hub import snapshot_download


def download_models():
    # Hugging Face의 모델 저장소 이름
    repo_id = "Ammunity/motive"

    # 모델 다운로드
    snapshot_download(repo_id, local_dir="./models")


if __name__ == "__main__":
    download_models()


# from huggingface_hub import HfApi
#
# api = HfApi()
# repo_id = "Amminity/motive"
# try:
#     repo_info = api.repo_info(repo_id=repo_id)
#     print(repo_info)
# except Exception as e:
#     print(e)