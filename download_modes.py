from huggingface_hub import snapshot_download


def download_models():
    # Hugging Face의 모델 저장소 이름
    repo_id = "Amminity/motive"

    # 모델 다운로드
    snapshot_download(repo_id, local_dir="./models")


if __name__ == "__main__":
    download_models()