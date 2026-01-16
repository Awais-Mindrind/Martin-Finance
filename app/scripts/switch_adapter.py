#!/usr/bin/env python3
"""
Interactive adapter switcher for testing different adapters.
Provides an easy way to test base model and switch between adapters.
"""
import argparse
import os
import subprocess
import sys

DEFAULT_BASE_MODEL = "/workspace/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
DEFAULT_ADAPTERS_DIR = "/workspace/output/adapters_gguf/v3"
DEFAULT_PROMPT = "Explain the importance of liquidity risk management in banking."

def list_adapters(adapters_dir):
    """List all .gguf adapters in the directory."""
    if not os.path.isdir(adapters_dir):
        return []
    
    adapters = []
    for name in sorted(os.listdir(adapters_dir)):
        if name.lower().endswith(".gguf"):
            adapters.append((name, os.path.join(adapters_dir, name)))
    
    return adapters

def run_test(model, adapter=None, prompt=DEFAULT_PROMPT, max_tokens=256, temp=0.7, ngl=None):
    """Run a single test with the given model and optional adapter."""
    cmd = ["python3", "/app/scripts/test_gguf.py", model]
    
    if adapter:
        cmd.extend(["--adapter", adapter])
    
    cmd.extend([
        "--prompt", prompt,
        "--max-tokens", str(max_tokens),
        "--temp", str(temp)
    ])
    
    if ngl is not None:
        cmd.extend(["--ngl", str(ngl)])
    
    print("\n" + "=" * 70)
    print(f"Testing: {'Base model only' if not adapter else os.path.basename(adapter)}")
    print("=" * 70)
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Test failed: {e}")
        return False
    
    return True

def interactive_menu(base_model, adapters_dir, prompt, max_tokens, temp, ngl):
    """Interactive menu to select and test adapters."""
    adapters = list_adapters(adapters_dir)
    
    print("\n" + "=" * 70)
    print("ADAPTER TESTING MENU")
    print("=" * 70)
    print(f"\nBase model: {base_model}")
    print(f"Adapters directory: {adapters_dir}")
    print(f"Found {len(adapters)} adapter(s)\n")
    
    while True:
        print("\nOptions:")
        print("  [0] Test base model (no adapter)")
        print("  [1] Test B2.gguf specifically")
        
        if adapters:
            print(f"\n  [2-{len(adapters)+1}] Test individual adapters:")
            for idx, (name, path) in enumerate(adapters, start=2):
                print(f"      [{idx}] {name}")
            
            print(f"\n  [{len(adapters)+2}] Test ALL adapters sequentially")
        
        print("  [q] Quit")
        
        choice = input("\nSelect option: ").strip().lower()
        
        if choice == 'q':
            print("\nExiting...")
            break
        elif choice == '0':
            run_test(base_model, None, prompt, max_tokens, temp, ngl)
        elif choice == '1':
            b2_path = os.path.join(adapters_dir, "B2.gguf")
            if os.path.exists(b2_path):
                run_test(base_model, b2_path, prompt, max_tokens, temp, ngl)
            else:
                print(f"\n✗ B2.gguf not found at {b2_path}")
        elif adapters and choice.isdigit():
            idx = int(choice)
            if 2 <= idx <= len(adapters) + 1:
                name, path = adapters[idx - 2]
                run_test(base_model, path, prompt, max_tokens, temp, ngl)
            elif idx == len(adapters) + 2:
                print(f"\nTesting all {len(adapters)} adapters sequentially...")
                for name, path in adapters:
                    run_test(base_model, path, prompt, max_tokens, temp, ngl)
            else:
                print("\n✗ Invalid option")
        else:
            print("\n✗ Invalid option")

def main():
    parser = argparse.ArgumentParser(
        description="Interactive adapter switcher for testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive menu
  python /app/scripts/switch_adapter.py

  # Test base model only
  python /app/scripts/switch_adapter.py --base-only

  # Test B2 adapter
  python /app/scripts/switch_adapter.py --adapter B2

  # Test specific adapter by name
  python /app/scripts/switch_adapter.py --adapter ASC_Financial_Accounting

  # Test all adapters
  python /app/scripts/switch_adapter.py --all

  # Custom prompt
  python /app/scripts/switch_adapter.py --adapter B2 --prompt "What is financial risk?"
        """
    )
    
    parser.add_argument("--model", default=DEFAULT_BASE_MODEL, help="Path to base GGUF model")
    parser.add_argument("--adapters-dir", default=DEFAULT_ADAPTERS_DIR, help="Directory containing GGUF adapters")
    parser.add_argument("--adapter", help="Test specific adapter by name (e.g., B2 or ASC_Financial_Accounting)")
    parser.add_argument("--base-only", action="store_true", help="Test base model only (no adapter)")
    parser.add_argument("--all", action="store_true", help="Test all adapters sequentially")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Test prompt")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate")
    parser.add_argument("--temp", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--ngl", type=int, help="GPU offload layers (set 0 for CPU)")
    
    args = parser.parse_args()
    
    # Validate base model exists
    if not os.path.exists(args.model):
        print(f"✗ Base model not found: {args.model}")
        return 1
    
    # Handle different modes
    if args.base_only:
        return 0 if run_test(args.model, None, args.prompt, args.max_tokens, args.temp, args.ngl) else 1
    
    elif args.adapter:
        # Find adapter by name (with or without .gguf extension)
        adapter_name = args.adapter
        if not adapter_name.endswith(".gguf"):
            adapter_name += ".gguf"
        
        adapter_path = os.path.join(args.adapters_dir, adapter_name)
        
        if not os.path.exists(adapter_path):
            print(f"✗ Adapter not found: {adapter_path}")
            print(f"\nAvailable adapters:")
            adapters = list_adapters(args.adapters_dir)
            for name, path in adapters:
                print(f"  • {name}")
            return 1
        
        return 0 if run_test(args.model, adapter_path, args.prompt, args.max_tokens, args.temp, args.ngl) else 1
    
    elif args.all:
        adapters = list_adapters(args.adapters_dir)
        if not adapters:
            print(f"✗ No adapters found in {args.adapters_dir}")
            return 1
        
        print(f"\nTesting all {len(adapters)} adapters sequentially...\n")
        success = True
        for name, path in adapters:
            if not run_test(args.model, path, args.prompt, args.max_tokens, args.temp, args.ngl):
                success = False
        
        return 0 if success else 1
    
    else:
        # Interactive mode
        interactive_menu(args.model, args.adapters_dir, args.prompt, args.max_tokens, args.temp, args.ngl)
        return 0

if __name__ == "__main__":
    sys.exit(main())
