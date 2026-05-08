# PoshanEnv — Dockerfile

FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn==0.29.0 \
    pydantic==2.7.1 \
    python-multipart==0.0.9

COPY . .

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
