import argparse
import subprocess
import os
import glob
import shlex

LLAMA_CPP_DIR = "/workspace/llama.cpp"
EXPORT_BIN = f"{LLAMA_CPP_DIR}/build/bin/llama-export-lora"
CONVERT_LORA_PY = f"{LLAMA_CPP_DIR}/convert_lora_to_gguf.py"

def run(cmd):
    print(f"Executing: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        return False

def export_single_adapter(lora_path, output_path, base_model_path=None):
    """
    Attempts to export a PEFT adapter to GGUF format.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Strategy 1: Use llama-export-lora binary (if available and base GGUF provided)
    if os.path.exists(EXPORT_BIN) and base_model_path and base_model_path.endswith(".gguf"):
        print(f"Attempting binary export for {lora_path}...")
        cmd = f"{shlex.quote(EXPORT_BIN)} -m {shlex.quote(base_model_path)} -o {shlex.quote(output_path)} {shlex.quote(lora_path)}"
        if run(cmd):
            return True

    # Strategy 2: Use python conversion script
    if os.path.exists(CONVERT_LORA_PY):
        print(f"Attempting python script conversion for {lora_path}...")
        cmd = f"python3 {shlex.quote(CONVERT_LORA_PY)} {shlex.quote(lora_path)} --outfile {shlex.quote(output_path)}"
        if run(cmd):
            return True

    print(f"Failed to export adapter: {lora_path}")
    return False

def main():
    parser = argparse.ArgumentParser(description="Export PEFT adapters to GGUF format.")
    parser.add_argument("--adapters_dir", default="/workspace/output/peft", help="Directory containing PEFT adapter folders")
    parser.add_argument("--output_dir", default="/workspace/output/adapters_gguf/v3", help="Directory to save GGUF adapters")
    parser.add_argument("--base_model", help="Path to base GGUF model (optional, for binary export)")
    parser.add_argument("--single_adapter", help="Path to a single PEFT adapter folder")
    
    args = parser.parse_args()

    if args.single_adapter:
        name = os.path.basename(args.single_adapter.rstrip("/"))
        output_path = os.path.join(args.output_dir, f"{name}.gguf")
        export_single_adapter(args.single_adapter, output_path, args.base_model)
    else:
        # Batch mode
        adapter_paths = [d for d in glob.glob(os.path.join(args.adapters_dir, "*")) if os.path.isdir(d)]
        if not adapter_paths:
            print(f"No adapters found in {args.adapters_dir}")
            return

        print(f"Found {len(adapter_paths)} adapters. Starting batch export...")
        for ap in adapter_paths:
            name = os.path.basename(ap.rstrip("/"))
            # Skip 'merged' or other non-adapter dirs if necessary
            if name == "merged": continue
            
            output_path = os.path.join(args.output_dir, f"{name}.gguf")
            print(f"\n>>> Exporting {name}...")
            export_single_adapter(ap, output_path, args.base_model)

if __name__ == "__main__":
    main()

