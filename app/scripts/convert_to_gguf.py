import os
import subprocess

HF_MERGED = "/workspace/peft/merged"
LLAMA_CPP = "/workspace/llama.cpp"
QUANT_BIN = f"{LLAMA_CPP}/build/bin/llama-quantize"

def run(cmd):
    print("Running:", cmd)
    subprocess.run(cmd, shell=True, check=True)

def main():
    target_path = os.environ.get("MODEL_PATH", "/workspace/models/mistral.gguf")
    parent = os.path.dirname(target_path)
    os.makedirs(parent, exist_ok=True)

    print(f"Output GGUF Path: {target_path}")

    # Clone llama.cpp if missing
    if not os.path.exists(LLAMA_CPP):
        run(f"git clone https://github.com/ggerganov/llama.cpp {LLAMA_CPP}")

    # Build llama.cpp if llama-quantize is missing
    if not os.path.exists(QUANT_BIN):
        run(
            f"cd {LLAMA_CPP} && rm -rf build && mkdir build && cd build && "
            f"cmake -DLLAMA_CURL=OFF -DLLAMA_BUILD_TOOL=ON .. && make -j"
        )

    f16_path = target_path.replace(".gguf", ".f16.gguf")

    # Convert HF model → GGUF FP16
    run(
        f"python3 {LLAMA_CPP}/convert_hf_to_gguf.py "
        f"{HF_MERGED} "
        f"--outfile {f16_path}"
    )

    # Quantize to Q4_K_M
    run(f"{QUANT_BIN} {f16_path} {target_path} Q4_K_M")

    print("GGUF created:", target_path)

if __name__ == "__main__":
    main()
