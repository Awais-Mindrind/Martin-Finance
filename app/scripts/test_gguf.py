import argparse
import subprocess
import os
import shlex

DEFAULT_PROMPT = "Explain the importance of liquidity risk management in banking."

def find_binary():
    paths = [
        "/workspace/llama.cpp/build/bin/llama-cli",
        "/workspace/llama.cpp/build/bin/main",
        "/workspace/llama.cpp/llama-cli",
        "/workspace/llama.cpp/main"
    ]
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def has_gpu():
    if os.path.exists("/dev/nvidia0"):
        return True
    try:
        subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False

def list_adapters(adapters_dir):
    if not adapters_dir or not os.path.isdir(adapters_dir):
        return []
    out = []
    for name in sorted(os.listdir(adapters_dir)):
        if name.lower().endswith(".gguf"):
            out.append(os.path.join(adapters_dir, name))
    return out

def run_inference(binary, model, prompt, max_tokens, temp, ngl, adapter=None):
    cmd = [
        binary,
        "-m", model,
        "-p", prompt,
        "-n", str(max_tokens),
        "--temp", str(temp),
    ]
    if ngl and ngl > 0:
        cmd += ["-ngl", str(ngl)]
    if adapter:
        cmd += ["--lora", adapter]

    print(f"Running inference with model: {model}")
    if adapter:
        print(f"Using adapter: {adapter}")
    print(f"Command: {' '.join(shlex.quote(c) for c in cmd)}")
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="Test a GGUF model.")
    parser.add_argument("model", help="Path to the .gguf model file")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Test prompt")
    parser.add_argument("--adapter", help="Path to a .gguf adapter file (optional)")
    parser.add_argument("--adapters-dir", help="Directory of .gguf adapters to test (optional)")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate")
    parser.add_argument("--temp", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--ngl", type=int, default=35, help="GPU offload layers (set 0 for CPU)")
    args = parser.parse_args()

    binary = find_binary()
    if not binary:
        print("Error: llama-cli or main binary not found in /workspace/llama.cpp/build/bin/")
        return

    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        return

    if args.adapter and args.adapters_dir:
        print("Error: use either --adapter or --adapters-dir, not both")
        return

    if args.ngl and args.ngl > 0 and not has_gpu():
        print("GPU not detected; forcing -ngl 0 for CPU.")
        args.ngl = 0

    try:
        if args.adapters_dir:
            adapters = list_adapters(args.adapters_dir)
            if not adapters:
                print(f"No .gguf adapters found in {args.adapters_dir}")
                return
            for adapter in adapters:
                run_inference(
                    binary, args.model, args.prompt, args.max_tokens,
                    args.temp, args.ngl, adapter=adapter
                )
        else:
            run_inference(
                binary, args.model, args.prompt, args.max_tokens,
                args.temp, args.ngl, adapter=args.adapter
            )
    except subprocess.CalledProcessError as e:
        print(f"Inference failed: {e}")

if __name__ == "__main__":
    main()
