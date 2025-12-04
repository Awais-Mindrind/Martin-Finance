import argparse
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from lora_layer_config import load_lora_config

HF_MODEL_DIR = "/workspace/models/hf_mistral"
DATA_PATH = "/workspace/data/processed/train.jsonl"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora_name", required=True)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--max_seq_length", type=int, default=512)
    return ap.parse_args()


def main():
    args = parse_args()

    cfg = load_lora_config(args.lora_name)
    out_dir = f"/workspace/peft/{args.lora_name}"
    os.makedirs(out_dir, exist_ok=True)

    print("Loading dataset...")
    ds = load_dataset("json", data_files=DATA_PATH)["train"]

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Loading model (4-bit)...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_DIR,
        quantization_config=bnb,
        device_map="auto",
    )

    print("Applying LoRA config...")
    lora_cfg = LoraConfig(
        r=cfg.r,
        lora_alpha=cfg.alpha,
        target_modules=cfg.target_modules,
        lora_dropout=cfg.dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_cfg)

    print("Trainer config...")
    train_cfg = SFTConfig(
        output_dir=out_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=cfg.learning_rate,
        max_steps=args.max_steps,
        warmup_steps=20,
        logging_steps=10,
        fp16=True,
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        packing=False,
    )

    print("Starting trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        args=train_cfg,
    )

    trainer.train()

    print("Saving adapter...")
    trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    print(f"Training complete → {out_dir}")


if __name__ == "__main__":
    main()
