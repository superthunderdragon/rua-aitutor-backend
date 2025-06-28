# ▶ 1. 베이스 이미지
FROM python:3.10-slim

# ▶ 2. 환경 변수
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VIRTUALENVS_CREATE=false

# ▶ 3. 작업 디렉터리
WORKDIR /app

# ▶ 4. 의존성 설치
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# ▶ 5. 애플리케이션 소스 복사
COPY . .

# ▶ 6. 네트워크
EXPOSE 5100

# ▶ 7. 엔트리포인트 (Gunicorn + Uvicorn worker)
CMD ["python", "app.py"]
