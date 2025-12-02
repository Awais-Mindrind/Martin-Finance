import argparse, os
from transformers import AutoModelForCausalLM
from peft import PeftModel
import torch

HF="/workspace/models/hf_mistral"

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--adapter",required=True)
    args=parser.parse_args()

    out="/workspace/peft/merged"
    os.makedirs(out,exist_ok=True)

    base=AutoModelForCausalLM.from_pretrained(HF,torch_dtype=torch.float16)
    model=PeftModel.from_pretrained(base,args.adapter)
    merged=model.merge_and_unload()
    merged.save_pretrained(out)

if __name__=="__main__":
    main()
