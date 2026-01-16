#!/usr/bin/env python3
"""
Quick verification script to check adapter files and test setup.
"""
import os
import sys

def check_file(path, description):
    """Check if a file/directory exists and report its status."""
    if os.path.exists(path):
        if os.path.isfile(path):
            size = os.path.getsize(path) / (1024 * 1024)  # MB
            print(f"✓ {description}: {path} ({size:.2f} MB)")
            return True
        elif os.path.isdir(path):
            count = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
            print(f"✓ {description}: {path} (directory, {count} files)")
            return True
    else:
        print(f"✗ {description}: {path} (NOT FOUND)")
        return False

def list_gguf_adapters(adapters_dir):
    """List all .gguf files in the adapters directory."""
    if not os.path.isdir(adapters_dir):
        print(f"\n✗ Adapters directory not found: {adapters_dir}")
        return []
    
    adapters = []
    for name in sorted(os.listdir(adapters_dir)):
        if name.lower().endswith(".gguf"):
            full_path = os.path.join(adapters_dir, name)
            size = os.path.getsize(full_path) / (1024 * 1024)  # MB
            adapters.append((name, full_path, size))
    
    return adapters

def main():
    print("=" * 60)
    print("ADAPTER VERIFICATION REPORT")
    print("=" * 60)
    
    # Check base model
    base_model = "/workspace/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    base_exists = check_file(base_model, "Base model")
    
    # Check B2 adapter specifically
    b2_adapter = "/workspace/output/adapters_gguf/v3/B2.gguf"
    b2_exists = check_file(b2_adapter, "B2 adapter")
    
    # Check adapters directory
    adapters_dir = "/workspace/output/adapters_gguf/v3"
    adapters_dir_exists = check_file(adapters_dir, "Adapters directory")
    
    # List all GGUF adapters
    print("\n" + "=" * 60)
    print("ALL GGUF ADAPTERS FOUND:")
    print("=" * 60)
    
    if adapters_dir_exists:
        adapters = list_gguf_adapters(adapters_dir)
        if adapters:
            print(f"\nFound {len(adapters)} adapter(s):\n")
            for name, path, size in adapters:
                print(f"  • {name:30s} ({size:7.2f} MB)")
        else:
            print("\n✗ No .gguf files found in adapters directory")
            print("  (Directory exists but is empty or contains non-GGUF files)")
    else:
        print("\n✗ Cannot list adapters - directory not found")
    
    # Check test binary
    print("\n" + "=" * 60)
    print("TEST INFRASTRUCTURE:")
    print("=" * 60)
    
    binary_paths = [
        "/workspace/llama.cpp/build/bin/llama-cli",
        "/workspace/llama.cpp/build/bin/main",
    ]
    binary_found = False
    for bp in binary_paths:
        if os.path.exists(bp):
            print(f"✓ Test binary found: {bp}")
            binary_found = True
            break
    
    if not binary_found:
        print("✗ Test binary (llama-cli) not found")
        print("  Expected locations:")
        for bp in binary_paths:
            print(f"    - {bp}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("=" * 60)
    
    all_ready = base_exists and b2_exists and adapters_dir_exists and binary_found
    
    if all_ready:
        print("✓ All components ready for testing!")
        print("\nYou can now run:")
        print("  1. Base model: python /app/main.py test_gguf --model", base_model)
        print("  2. B2 adapter: python /app/main.py test_gguf --model", base_model, "--adapter", b2_adapter)
        if adapters:
            print(f"  3. All adapters: python /app/main.py test_gguf --model {base_model} --adapters-dir {adapters_dir}")
    else:
        print("✗ Some components are missing. Please check the errors above.")
        if not base_exists:
            print("  → Base model not found. Check MODEL_PATH.")
        if not b2_exists:
            print("  → B2.gguf not found. May need to export from PEFT.")
        if not adapters_dir_exists:
            print("  → Adapters directory not found. Check path.")
        if not binary_found:
            print("  → Test binary not found. May need to rebuild llama.cpp.")
    
    print("=" * 60)
    
    return 0 if all_ready else 1

if __name__ == "__main__":
    sys.exit(main())
