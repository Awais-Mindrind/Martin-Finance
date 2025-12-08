# How to fine‑tune on a PDF in /workspace

1) Copy your PDF into the raw input folder:
```bash
mkdir -p /workspace/data/raw_pdfs
cp /workspace/inputs/v1/Investments/Investments.pdf /workspace/data/raw_pdfs/
```
(Replace the source path with any file under `/workspace`.)

2) Run the pipeline steps individually (in order). Open a terminal in the pod and execute:

- Pretest PDFs (scores suitability, writes `/workspace/data/processed/pdf_pretest.json`):
```bash
python3 /app/scripts/pdf_pretest.py
```

- Build dataset from recommended PDFs (writes `/workspace/data/processed/train.jsonl`):
```bash
python3 /app/scripts/build_dataset.py
```

- Train LoRA adapters (saves to `/workspace/peft/level1|2|3`):
```bash
python3 /app/scripts/train_lora.py --lora_name level1
python3 /app/scripts/train_lora.py --lora_name level2
python3 /app/scripts/train_lora.py --lora_name level3
```

- Evaluate base vs adapters (writes `/workspace/eval/eval.jsonl`):
```bash
python3 /app/scripts/eval_layers.py
```

- Merge adapter into base HF model (saves to `/workspace/peft/merged`):
```bash
python3 /app/scripts/merge_lora.py --adapter /workspace/peft/level3
```

- Convert merged HF → GGUF (F16 + Q4_K_M) at `MODEL_PATH` (default `/workspace/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf`):
```bash
python3 /app/scripts/convert_to_gguf.py
```

- Archive processed PDFs/TXTs to `PDF_ARCHIVE_PATH` (default `/workspace/data/archive`):
```bash
python3 /app/scripts/archive_used_pdfs.py
```

That’s the full sequence for a single PDF or a batch placed in `/workspace/data/raw_pdfs`.
