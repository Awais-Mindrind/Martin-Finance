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
RUN git clone https://github.com/ggerganov/llama.cpp /workspace/llama.cpp && \
    cd /workspace/llama.cpp && mkdir -p build && cd build && \
    cmake .. -DLLAMA_BUILD_TOOL=ON && \
    make -j"$(nproc)"

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
    cryptography \
    sentencepiece \
    safetensors \
    einops

# CUDA-enabled PyTorch stack (cu121) for GPU + bitsandbytes/4-bit
RUN pip install --no-cache-dir \
    torch==2.2.1+cu121 \
    torchvision==0.17.1+cu121 \
    torchaudio==2.2.1 \
    --index-url https://download.pytorch.org/whl/cu121

COPY app/ /app/

RUN chmod +x /app/main.py

# Keep the container alive by default so RunPod terminal stays available.
# Override the default CMD to run training, e.g.:
#   docker run awais2512/martin-model-tune:latest "python /app/main.py train_all"
ENTRYPOINT ["/bin/bash"]
CMD ["-c", "sleep infinity"]
