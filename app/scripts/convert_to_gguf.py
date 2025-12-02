import os
import subprocess

HF_MERGED = "/workspace/peft/merged"

def run(cmd):
    print("Running:", cmd)
    subprocess.run(cmd, shell=True, check=True)

def main():
    # 1️⃣ Read environment variable for final GGUF output
    # Default falls back to mistral.gguf inside the gguf folder
    target_path = os.environ.get(
        "MODEL_PATH",
        "/workspace/models/gguf/mistral.gguf"
    )
    
    parent_dir = os.path.dirname(target_path)
    os.makedirs(parent_dir, exist_ok=True)

    print(f"Final target GGUF model path: {target_path}")

    # 2️⃣ Clone llama.cpp if missing
    if not os.path.exists("/workspace/llama.cpp"):
        run("git clone https://github.com/ggerganov/llama.cpp /workspace/llama.cpp")
        run("cd /workspace/llama.cpp && mkdir -p build && cd build && cmake .. && make -j")

    # 3️⃣ Create intermediate F16 GGUF before quantization
    f16_path = target_path.replace(".gguf", ".f16.gguf")

    run(
        "python3 /workspace/llama.cpp/convert-hf-to-gguf.py "
        f"--model-dir {HF_MERGED} "
        f"--outfile {f16_path}"
    )

    # 4️⃣ Quantize to Q4_K_M
    run(
        f"/workspace/llama.cpp/build/bin/quantize "
        f"{f16_path} {target_path} Q4_K_M"
    )

    print("===================================================")
    print(f"Fine-tuned GGUF model saved to:\n{target_path}")
    print("===================================================")

if __name__ == "__main__":
    main()
