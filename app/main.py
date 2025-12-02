import argparse
import subprocess
import os

def run(cmd):
    print(f"\n===== Running: {cmd} =====")
    subprocess.run(cmd, shell=True, check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=[
        "pdf_pretest",
        "build_dataset",
        "train_level1",
        "train_level2",
        "train_level3",
        "eval_all",
        "merge_level",
        "convert_to_gguf",
        "archive_pdfs",
        "train_all"
    ])
    args = parser.parse_args()

    if args.mode == "pdf_pretest":
        run("python3 /app/scripts/pdf_pretest.py")

    elif args.mode == "build_dataset":
        run("python3 /app/scripts/build_dataset.py")

    elif args.mode == "train_level1":
        run("python3 /app/scripts/train_lora.py --lora level1")

    elif args.mode == "train_level2":
        run("python3 /app/scripts/train_lora.py --lora level2")

    elif args.mode == "train_level3":
        run("python3 /app/scripts/train_lora.py --lora level3")

    elif args.mode == "eval_all":
        run("python3 /app/scripts/eval_layers.py")

    elif args.mode == "merge_level":
        run("python3 /app/scripts/merge_lora.py")

    elif args.mode == "archive_pdfs":
        run("python3 /app/scripts/archive_used_pdfs.py")

    elif args.mode == "convert_to_gguf":
        run("python3 /app/scripts/convert_to_gguf.py")

    # 🚀 Full pipeline (new PDFs → dataset → LoRA → merge → GGUF → archive PDFs)
    elif args.mode == "train_all":
        run("python3 /app/scripts/pdf_pretest.py")
        run("python3 /app/scripts/build_dataset.py")
        run("python3 /app/scripts/train_lora.py --lora level1")
        run("python3 /app/scripts/train_lora.py --lora level2")
        run("python3 /app/scripts/train_lora.py --lora level3")
        run("python3 /app/scripts/merge_lora.py")
        run("python3 /app/scripts/convert_to_gguf.py")
        run("python3 /app/scripts/archive_used_pdfs.py")

        print("\n================ DONE ================")
        print("Model updated & PDFs archived.")
        print("========================================")

if __name__ == "__main__":
    main()
