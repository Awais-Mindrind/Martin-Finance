import json, os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

HF="/workspace/models/hf_mistral"
PROMPTS=[
    "Summarize a quarterly financial report.",
    "Explain net interest income vs non-interest income.",
    "Estimate liquidity risk from a balance sheet."
]

def run_scenario(name, adapters):
    tokenizer=AutoTokenizer.from_pretrained(HF)
    m=AutoModelForCausalLM.from_pretrained(HF,device_map="auto")

    for ad in adapters:
        m=PeftModel.from_pretrained(m,ad)

    results=[]
    for p in PROMPTS:
        out=m.generate(**tokenizer(p,return_tensors="pt").to(m.device),
                       max_new_tokens=256)
        txt=tokenizer.decode(out[0],skip_special_tokens=True)
        results.append({"scenario":name,"prompt":p,"response":txt})
    return results

def main():
    scenarios={
        "base":[],
        "level1":["/workspace/peft/level1"],
        "level2":["/workspace/peft/level2"],
        "level3":["/workspace/peft/level3"],
    }
    out="/workspace/eval/eval.jsonl"
    os.makedirs("/workspace/eval",exist_ok=True)
    with open(out,"w") as w:
        for s,ad in scenarios.items():
            for r in run_scenario(s,ad):
                w.write(json.dumps(r)+"\n")

if __name__=="__main__":
    main()
