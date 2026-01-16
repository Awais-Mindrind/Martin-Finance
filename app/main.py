import argparse
import subprocess
import os
import shlex

def run(cmd):
    print(f"\n===== Running: {cmd} =====")
    subprocess.run(cmd, shell=True, check=True)

def print_welcome_message():
    print("""
==================================================
   MARTIN FINANCE - LLM ADAPTER CONTROL CENTER
==================================================

Welcome, Martin. Use the following commands to test your models:

STEP 1: VERIFY YOUR SETUP
   python3 /app/main.py verify_adapters

STEP 2: TEST YOUR MODELS
   # Interactive adapter switcher (recommended)
   python3 /app/main.py switch_adapter

   # Or use direct commands:
   # Base model only:
   python3 /app/main.py test_gguf --model /workspace/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf

   # Test B2 adapter:
   python3 /app/main.py switch_adapter --adapter B2

   # Test all adapters:
   python3 /app/main.py switch_adapter --all

STEP 3: EXPORT ADAPTERS (if needed)
   python3 /app/main.py export_adapters

Need help? Contact Mian.
==================================================
""")

def main():
    parser = argparse.ArgumentParser(description="Martin Finance LLM Control Center")
    parser.add_argument("mode", nargs='?', choices=[
        "pdf_pretest",
        "build_dataset",
        "train_level1",
        "train_level2",
        "train_level3",
        "eval_all",
        "merge_level",
        "convert_to_gguf",
        "archive_pdfs",
        "train_all",
        "test_gguf",
        "export_adapters",
        "verify_adapters",
        "switch_adapter"
    ], help="Action to perform")
    
    parser.add_argument("--model", help="Path to GGUF model for test_gguf mode", default="/workspace/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf")
    parser.add_argument("--base-gguf", help="Path to base GGUF model for adapter export", default="/workspace/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf")
    parser.add_argument("--adapter", help="Path to GGUF adapter for test_gguf mode")
    parser.add_argument("--adapters-dir", help="Directory of GGUF adapters to test")
    parser.add_argument("--prompt", help="Prompt for test_gguf mode")
    parser.add_argument("--max-tokens", type=int, help="Max tokens for test_gguf mode")
    parser.add_argument("--temp", type=float, help="Temperature for test_gguf mode")
    parser.add_argument("--ngl", type=int, help="GPU offload layers for test_gguf mode")
    
    args = parser.parse_args()

    if not args.mode:
        print_welcome_message()
        return

    if args.mode == "pdf_pretest":
        run("python3 /app/scripts/pdf_pretest.py")

    elif args.mode == "build_dataset":
        run("python3 /app/scripts/build_dataset.py")

    elif args.mode == "train_level1":
        run("python3 /app/scripts/train_lora.py --lora_name level1")

    elif args.mode == "train_level2":
        run("python3 /app/scripts/train_lora.py --lora_name level2")

    elif args.mode == "train_level3":
        run("python3 /app/scripts/train_lora.py --lora_name level3")

    elif args.mode == "eval_all":
        run("python3 /app/scripts/eval_layers.py")

    elif args.mode == "merge_level":
        run("python3 /app/scripts/merge_lora.py")

    elif args.mode == "archive_pdfs":
        run("python3 /app/scripts/archive_used_pdfs.py")

    elif args.mode == "convert_to_gguf":
        run("python3 /app/scripts/convert_to_gguf.py")

    elif args.mode == "test_gguf":
        cmd = ["python3", "/app/scripts/test_gguf.py", args.model]
        if args.adapter:
            cmd += ["--adapter", args.adapter]
        if args.adapters_dir:
            cmd += ["--adapters-dir", args.adapters_dir]
        if args.prompt:
            cmd += ["--prompt", args.prompt]
        if args.max_tokens is not None:
            cmd += ["--max-tokens", str(args.max_tokens)]
        if args.temp is not None:
            cmd += ["--temp", str(args.temp)]
        if args.ngl is not None:
            cmd += ["--ngl", str(args.ngl)]
        run(" ".join(shlex.quote(c) for c in cmd))

    elif args.mode == "export_adapters":
        cmd = ["python3", "/app/scripts/export_lora.py", "--base_model", args.base_gguf]
        if args.adapters_dir:
            cmd += ["--adapters_dir", args.adapters_dir]
        run(" ".join(shlex.quote(c) for c in cmd))

    elif args.mode == "verify_adapters":
        run("python3 /app/scripts/verify_adapters.py")

    elif args.mode == "switch_adapter":
        cmd = ["python3", "/app/scripts/switch_adapter.py"]
        if args.model:
            cmd += ["--model", args.model]
        if args.adapter:
            cmd += ["--adapter", args.adapter]
        if args.adapters_dir:
            cmd += ["--adapters-dir", args.adapters_dir]
        if args.prompt:
            cmd += ["--prompt", args.prompt]
        if args.max_tokens is not None:
            cmd += ["--max-tokens", str(args.max_tokens)]
        if args.temp is not None:
            cmd += ["--temp", str(args.temp)]
        if args.ngl is not None:
            cmd += ["--ngl", str(args.ngl)]
        run(" ".join(shlex.quote(c) for c in cmd))

    # 🚀 Full pipeline (new PDFs → dataset → LoRA → merge → GGUF → archive PDFs)
    elif args.mode == "train_all":
        run("python3 /app/scripts/pdf_pretest.py")
        run("python3 /app/scripts/build_dataset.py")
        run("python3 /app/scripts/train_lora.py --lora_name level1")
        run("python3 /app/scripts/train_lora.py --lora_name level2")
        run("python3 /app/scripts/train_lora.py --lora_name level3")
        run("python3 /app/scripts/merge_lora.py")
        run("python3 /app/scripts/convert_to_gguf.py")
        run("python3 /app/scripts/archive_used_pdfs.py")

        print("\n================ DONE ================")
        print("Model updated & PDFs archived.")
        print("========================================")

if __name__ == "__main__":
    main()
