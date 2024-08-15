#!/bin/bash

# Hugging Face 모델 다운로드
python download_models.py

# Git 저장소 최신 상태로 업데이트
git pull

# 메인 애플리케이션 실행
python app_v2.py