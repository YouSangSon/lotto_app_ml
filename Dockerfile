FROM python:3.11-slim

LABEL authors="yousang"
LABEL version="1.0.1"

# 작업 디렉토리 설정
WORKDIR /app

# 의존성을 설치하기 위한 requirements.txt 파일을 컨테이너에 복사
COPY requirements.txt /app

# 파이썬 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt

# 파이썬 스크립트와 필요한 모든 파일들을 컨테이너에 복사
COPY . /app

# 컨테이너를 실행할 때 파이썬 애플리케이션을 실행하도록 명령 설정
ENTRYPOINT ["python3", "/app/main.py"]

