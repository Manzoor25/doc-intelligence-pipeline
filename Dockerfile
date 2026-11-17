FROM python:3.12.7-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-deu \
    fonts-dejavu-core \
    fonts-liberation \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY pyproject.toml /app/pyproject.toml
COPY src /app/src
COPY scripts /app/scripts

RUN mkdir -p /app/data/synthetic/documents \
             /app/data/synthetic/ground_truth \
             /app/data/eval_feedback \
             /app/data/inbox/processed \
             /app/data/inbox/failed \
             /app/data/exports

EXPOSE 8000

CMD ["uvicorn", "doc_pipeline.main:app", "--host", "0.0.0.0", "--port", "8000"]
