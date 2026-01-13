FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV MODEL_PATH=/workspace/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf
ENV PDF_ARCHIVE_PATH=/workspace/data/archive

RUN apt-get update && apt-get install -y \
    git wget curl nano unzip python3 python3-pip python3-dev build-essential cmake \
    libssl-dev libffi-dev libxml2-dev libxslt1-dev zlib1g-dev libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/models /workspace/peft /workspace/data /workspace/llama.cpp /app/scripts

WORKDIR /workspace

# Build llama.cpp with tooling (llama-quantize, llama-cli, etc.)
RUN git -c http.version=HTTP/1.1 clone --depth 1 --single-branch https://github.com/ggerganov/llama.cpp /workspace/llama.cpp && \
    cd /workspace/llama.cpp && mkdir -p build && cd build && \
    cmake .. \
        -DGGML_CUDA=ON \
        -DLLAMA_BUILD_TOOLS=ON \
        -DGGML_AVX2=OFF \
        -DGGML_FMA=OFF \
        -DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs && \
    make -j"$(nproc)"

# Patch convert_hf_to_gguf.py to alias missing torch uint types
RUN python3 - <<'PY'
from pathlib import Path
path = Path('/workspace/llama.cpp/convert_hf_to_gguf.py')
txt = path.read_text()
marker = 'import torch\n'
inject = """import torch
if not hasattr(torch, 'uint64'): torch.uint64 = torch.int64
if not hasattr(torch, 'uint32'): torch.uint32 = torch.int32
if not hasattr(torch, 'uint16'): torch.uint16 = torch.int16
if not hasattr(torch, 'uint8'):  torch.uint8  = torch.int8
"""
if inject not in txt:
    txt = txt.replace(marker, inject, 1)
    path.write_text(txt)
PY

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir \
    transformers==4.43.3 \
    peft==0.12.0 \
    accelerate==0.33.0 \
    trl==0.9.6 \
    datasets \
    pandas \
    pypdf \
    cryptography \
    sentencepiece \
    safetensors \
    einops \
    rich

# CUDA-enabled PyTorch stack (cu121) for GPU + bitsandbytes/4-bit (compatible versions)
RUN pip install --no-cache-dir \
    torch==2.1.2+cu121 \
    torchvision==0.16.2+cu121 \
    torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir bitsandbytes==0.43.1

COPY app/ /app/

RUN chmod +x /app/main.py

# Keep the container alive by default so RunPod terminal stays available.
# Override the default CMD to run training, e.g.:
#   docker run awais2512/martin-model-tune:latest "python /app/main.py train_all"
ENTRYPOINT ["/bin/bash"]
CMD ["-c", "sleep infinity"]
