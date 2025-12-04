import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "/workspace/models/hf_mistral"

def merge_adapter(model, adapter_path):
    print(f"Merging adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter",
        default=None,
        help="Path to adapter folder (if none, merges all levels sequentially)",
    )
    args = parser.parse_args()

    out_dir = "/workspace/peft/merged"
    os.makedirs(out_dir, exist_ok=True)

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(BASE)

    if args.adapter:
        model = merge_adapter(model, args.adapter)
    else:
        for level in ["level1", "level2", "level3"]:
            path = f"/workspace/peft/{level}"
            if not os.path.isdir(path):
                print(f"Skipping {path}: not found")
                continue
            try:
                model = merge_adapter(model, path)
            except Exception as exc:
                print(f"Skipping {path}: {exc}")

    print(f"Saving merged model to: {out_dir}")
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)


if __name__ == "__main__":
    main()
