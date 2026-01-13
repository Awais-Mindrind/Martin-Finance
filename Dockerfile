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

# Build llama.cpp (Optimized for speed and GitHub Actions memory limits)
RUN git -c http.version=HTTP/1.1 clone --depth 1 --single-branch https://github.com/ggerganov/llama.cpp /workspace/llama.cpp && \
    cd /workspace/llama.cpp && mkdir -p build && cd build && \
    # Fix for libcuda.so.1 not found in non-GPU environments (like GitHub Actions)
    ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    export LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LIBRARY_PATH && \
    cmake .. \
        -DGGML_CUDA=ON \
        -DGGML_AVX2=OFF \
        -DGGML_FMA=OFF \
        -DCMAKE_CUDA_ARCHITECTURES="80;86" \
        -DCMAKE_EXE_LINKER_FLAGS="-Wl,--allow-shlib-undefined" && \
    # Build only necessary tools with limited parallelism to prevent OOM
    make -j2 llama-cli llama-quantize llama-export-lora

# Patch convert_hf_to_gguf.py to alias missing torch uint types
RUN python3 - <<'PY'
from pathlib import Path
path = Path('/workspace/llama.cpp/convert_hf_to_gguf.py')
if path.exists():
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

# CUDA-enabled PyTorch stack (cu121) for GPU + bitsandbytes/4-bit
RUN pip install --no-cache-dir \
    torch==2.1.2+cu121 \
    torchvision==0.16.2+cu121 \
    torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir bitsandbytes==0.43.1

COPY app/ /app/

RUN chmod +x /app/main.py

ENTRYPOINT ["/bin/bash"]
CMD ["-c", "python3 /app/main.py && sleep infinity"]
