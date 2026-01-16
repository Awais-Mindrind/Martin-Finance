# Adapter Testing Guide

Quick reference for testing your 22 adapters and base model.

## Step 1: Verify Your Setup

First, check that everything is ready:

```bash
python3 /app/main.py verify_adapters
```

This will show you:
- ✓ Base model location and size
- ✓ B2.gguf adapter status
- ✓ All adapters found in `/workspace/output/adapters_gguf/v3/`
- ✓ Test binary availability

## Step 2: Test Your Models

### Option A: Interactive Menu (Recommended)

Easiest way to switch between adapters:

```bash
python3 /app/main.py switch_adapter
```

This gives you a menu:
- `[0]` Test base model (no adapter)
- `[1]` Test B2.gguf specifically
- `[2-N]` Test individual adapters
- `[N+1]` Test all adapters sequentially
- `[q]` Quit

### Option B: Direct Commands

**Test base model only:**
```bash
python3 /app/main.py test_gguf --model /workspace/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

**Test B2 adapter:**
```bash
python3 /app/main.py switch_adapter --adapter B2
```

**Test specific adapter by name:**
```bash
python3 /app/main.py switch_adapter --adapter ASC_Financial_Accounting
```

**Test all 22 adapters sequentially:**
```bash
python3 /app/main.py switch_adapter --all
```

**Custom prompt:**
```bash
python3 /app/main.py switch_adapter --adapter B2 --prompt "What is financial risk management?"
```

## Step 3: Understanding the Output

Each test will:
1. Load the base model
2. Apply the adapter (if specified)
3. Run inference with your prompt
4. Display the generated text

Compare outputs between:
- Base model (no adapter)
- B2.gguf
- Other adapters

## Troubleshooting

### "Adapter not found"
- Check that adapters are in `/workspace/output/adapters_gguf/v3/`
- Verify they're `.gguf` files (not PEFT folders)
- Run `verify_adapters` to see what's available

### "Test binary not found"
- The `llama-cli` binary should be at `/workspace/llama.cpp/build/bin/llama-cli`
- May need to rebuild llama.cpp if missing

### "No .gguf adapters found"
- Your adapters might be PEFT format (folders) instead of GGUF files
- Run `python3 /app/main.py export_adapters` to convert them

## Quick Reference

| Task | Command |
|------|---------|
| Verify setup | `python3 /app/main.py verify_adapters` |
| Interactive menu | `python3 /app/main.py switch_adapter` |
| Test base | `python3 /app/main.py test_gguf --model /workspace/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf` |
| Test B2 | `python3 /app/main.py switch_adapter --adapter B2` |
| Test all | `python3 /app/main.py switch_adapter --all` |
| Export adapters | `python3 /app/main.py export_adapters` |

## Notes

- **Sequential testing**: The current setup tests adapters one at a time (restarts inference for each)
- **Hot-swap**: Runtime adapter switching via `/lora-adapters` endpoint is not available in the current build
- **GPU/CPU**: The scripts auto-detect GPU and adjust settings accordingly
