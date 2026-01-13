## End-to-End Layering & Testing Guide

This guide covers PEFT training on a GPU pod, serving static GGUF+LoRA on a CPU pod, and hot-swapping LoRA scales at runtime via `/lora-adapters` (llama.cpp supports `--lora`, `--lora-init-without-apply`, and POST `/lora-adapters`).

### Pods and Roles
- **GPU pod (training/build):** Uses this repo’s CUDA image to run the PEFT pipeline, produce LoRA adapters and merged GGUFs.
- **CPU pod (serving/test):** Uses the serving image (llama.cpp server) to load base GGUF + LoRA and expose `/lora-adapters` for dynamic layering.

### 1) GPU Pod: Train and Export Layers
Pipeline (driven by `app/main.py`):
- `pdf_pretest` → score PDFs in `/workspace/data/raw_pdfs`, write `/workspace/data/processed/pdf_pretest.json`.
- `build_dataset` → chunk recommended PDFs, write `/workspace/data/processed/train.jsonl`.
- `train_level1/2/3` → train LoRA adapters with presets from `app/scripts/lora_layer_config.py`, save to `/workspace/peft/level{1,2,3}`.
- `merge_level` → merge adapters into the base HF model (`/workspace/models/hf_mistral` by default), save merged HF to `/workspace/peft/merged`.
- `convert_to_gguf` → convert merged HF → GGUF FP16 → quantize Q4_K_M; output at `MODEL_PATH` (default `/workspace/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf`).
- `archive_pdfs` → move used PDFs from `/workspace/data/raw_pdfs` to `PDF_ARCHIVE_PATH` (default `/workspace/data/archive`).
- `train_all` runs the full chain above.

Move PDFs into `raw_pdfs` (example):
```
cp /workspace/inputs/v1/Investments/Investments.pdf /workspace/data/raw_pdfs/
```

Key commands (inside GPU pod):
- Full pipeline: `python /app/main.py train_all`
- Individual steps: `python /app/main.py pdf_pretest` … `build_dataset` … `train_level1` … `train_level2` … `train_level3` … `merge_level` … `convert_to_gguf` … `archive_pdfs`

Outputs to carry to CPU pod:
- LoRA adapters: `/workspace/output/peft/<layer>` (e.g., your ASC_* folders). For convenience, create a one-time symlink: `ln -s /workspace/output/peft /workspace/peft`.
- Merged HF model: use the path where you save it (default in scripts is `/workspace/peft/merged`; if you keep everything in `/workspace/output`, save or move it there and reference that path).
- GGUF (quantized): `/workspace/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf` (or whatever `MODEL_PATH` you set).
- If you prefer a single directory, copy/symlink adapters into `/workspace/models` and adjust paths below accordingly.

### 2) CPU Pod: Static GGUF + LoRA (llama.cpp server)
- Image: your serving image (e.g., `docker.io/awais2512/martin-finance-grand:v1`)
- Env: `MODEL_PATH` → base GGUF to load (e.g., `/workspace/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf`), plus SSH/TTYD ports as needed.
- Ports: publish 5000 (LLM), 22 (SSH), 7681 (web terminal); add others if required.
- Startup: `/start.sh` launches sshd, ttyd, and `python -m llama_cpp.server --model $MODEL_PATH ...` on port 5000.

Static load with a LoRA at start (inside CPU pod):
```
pkill -f "llama_cpp.server" || true
python -m llama_cpp.server \
  --model /workspace/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf \
  --lora /workspace/output/peft/ASC_Financial_Accounting \
  --host 0.0.0.0 --port 5000 \
  --n_ctx ${N_CTX:-2048} --n_threads ${N_THREADS:-8} \
  --chat_format mistral-instruct \
  --flash_attn False &
```
- This loads the base GGUF and a LoRA adapter into memory. Files on disk are not modified. Other services stay up.

### 3) CPU Pod: Dynamic Layering via `/lora-adapters`
Hot-swap adapters and scales without restart:
- Apply one adapter:
```
curl -s -X POST http://127.0.0.1:5000/lora-adapters \
  -H "Content-Type: application/json" \
  -d '{"adapters":[{"path":"/workspace/output/peft/ASC_Financial_Accounting","scale":1.0}]}'
```
- Switch to another:
```
curl -s -X POST http://127.0.0.1:5000/lora-adapters \
  -H "Content-Type: application/json" \
  -d '{"adapters":[{"path":"/workspace/output/peft/ASC_Financial_Services","scale":1.0}]}'
```
- Blend multiple:
```
curl -s -X POST http://127.0.0.1:5000/lora-adapters \
  -H "Content-Type: application/json" \
  -d '{"adapters":[{"path":"/workspace/output/peft/ASC_Financial_Accounting","scale":0.6},{"path":"/workspace/output/peft/ASC_Long_Lived_Assets_Intangibles","scale":0.4}]}'
```
- Clear adapters: `{"adapters":[]}`
- Under the hood: base GGUF stays loaded; LoRA deltas/scales are applied in memory. No disk edits.

### 4) QA/Test Loop for the 25 Layers
1) Pick a base GGUF (`MODEL_PATH` or restart command).  
2) Apply adapters/scales via `/lora-adapters` (or start with `--lora`).  
3) Run prompts (accounting/ASC scenarios, edge cases) and record quality/latency.  
4) Repeat for each of the 25 layers (and blends, if needed).  
5) Compare outputs to assess training impact.

Quick LLM sanity check:
```
curl -s http://127.0.0.1:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"local","messages":[{"role":"user","content":"Give me two facts about Mars"}],"max_tokens":128}'
```

### 5) PEFT → Serving Flow (what’s happening)
- Training (GPU): base HF model at `/workspace/models/hf_mistral` → LoRA adapters (`/workspace/peft/levelN`) → optional merge (`/workspace/peft/merged`) → GGUF (`/workspace/models/...gguf`) via `convert_to_gguf.py` and `llama-quantize`.
- Serving (CPU): llama.cpp server loads the base GGUF; `--lora` and `/lora-adapters` apply adapters/scales in memory.
- Files remain unchanged during serving; only the in-memory model state updates.

### 6) Health Checks & Troubleshooting
- ttyd: `curl -I http://127.0.0.1:${TTYD_PORT:-7681}` → expect 200.
- LLM head check: `curl -I http://127.0.0.1:5000/v1/chat/completions` → 200/404 is fine.
- If adapter load fails: verify the exact path under `/workspace/models` or `/workspace/peft`, and permissions.
- If port unreachable externally: ensure the port is published in RunPod.
