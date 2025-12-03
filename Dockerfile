FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TARGET_GGUF_PATH=/workspace/models/gguf/mistral.gguf
ENV PDF_ARCHIVE_PATH=/workspace/data/archive

RUN apt-get update && apt-get install -y \
    git wget curl python3 python3-pip python3-dev build-essential \
    libssl-dev libffi-dev libxml2-dev libxslt1-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir \
    transformers==4.43.3 \
    peft==0.12.0 \
    accelerate==0.33.0 \
    bitsandbytes \
    trl==0.9.6 \
    datasets \
    pandas \
    pypdf \
    sentencepiece \
    torch

COPY app/ /app/

RUN chmod +x /app/main.py

ENTRYPOINT ["python3", "/app/main.py"]
