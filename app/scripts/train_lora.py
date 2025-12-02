import argparse, os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

HF = "/workspace/models/hf_mistral"

def get_cfg(level):
    if level=="level1":
        return dict(r=8,alpha=16,modules=["q_proj","v_proj"],lr=2e-4)
    if level=="level2":
        return dict(r=16,alpha=32,modules=["q_proj","v_proj","k_proj","o_proj"],lr=1e-4)
    if level=="level3":
        return dict(r=16,alpha=32,modules=["q_proj","v_proj","k_proj","o_proj"],lr=5e-5)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--lora",required=True)
    args=parser.parse_args()

    cfg=get_cfg(args.lora)
    out = f"/workspace/peft/{args.lora}"
    os.makedirs(out,exist_ok=True)

    ds=load_dataset("json",data_files="/workspace/data/processed/train.jsonl")["train"]

    tokenizer=AutoTokenizer.from_pretrained(HF)
    tokenizer.pad_token = tokenizer.eos_token

    bnb=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4")

    model=AutoModelForCausalLM.from_pretrained(HF,quantization_config=bnb,device_map="auto")

    lora=LoraConfig(
        r=cfg["r"],
        lora_alpha=cfg["alpha"],
        target_modules=cfg["modules"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model=get_peft_model(model,lora)

    trainer=SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        dataset_text_field="text",
        max_seq_length=512,
        packing=False,
        args=dict(
            output_dir=out,
            max_steps=200,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=cfg["lr"],
            save_strategy="no"
        )
    )
    trainer.train()
    trainer.model.save_pretrained(out)

if __name__=="__main__":
    main()
