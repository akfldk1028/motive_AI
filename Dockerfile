FROM python:3.11.6

WORKDIR /app

ENV BASE_URL="https://backend.motive.beauty/api/v1"
# Git 설치
RUN apt-get update && apt-get install -y git

# 필요한 Python 패키지 설치
RUN pip install huggingface_hub gitpython

# 코드 복제
RUN git clone "https://github.com/akfldk1028/motive_AI.git" .

# requirements.txt 파일이 있다면 의존성 설치
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Hugging Face 모델 다운로드 스크립트 복사
COPY download_models.py .

# 실행 스크립트 복사
COPY run.sh .

RUN chmod +x run.sh

CMD ["./run.sh"]

#docker build -t motive-ai-service .
#docker run -p 5000:5000 motive-ai-service
#docker run -e BASE_URL="http://127.0.0.1:8000/api/v1" -p 5000:5000 motive-ai-service
