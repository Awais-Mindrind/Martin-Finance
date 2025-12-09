# How to fine‑tune on a PDF in /workspace
0) Install non-conflicting dependencies
```bash
pip uninstall -y bitsandbytes torch torchvision torchaudio
pip install --no-cache-dir \
  torch==2.1.2+cu121 \
  torchvision==0.16.2+cu121 \
  torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu121
pip install --no-cache-dir bitsandbytes==0.43.1
```

```bash
python3 - <<'PY'
from pathlib import Path
path = Path("/workspace/llama.cpp/convert_hf_to_gguf.py")
txt = path.read_text()
marker = "import torch\n"
inject = """import torch
if not hasattr(torch, 'uint64'): torch.uint64 = torch.int64
if not hasattr(torch, 'uint32'): torch.uint32 = torch.int32
if not hasattr(torch, 'uint16'): torch.uint16 = torch.int16
if not hasattr(torch, 'uint8'):  torch.uint8  = torch.int8
"""
if inject not in txt:
    txt = txt.replace(marker, inject, 1)
    path.write_text(txt)
    print("Patched convert_hf_to_gguf.py with uint aliases")
else:
    print("Patch already present")
PY
```

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
