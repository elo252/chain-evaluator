#!/usr/bin/env python3
"""
Quick start script - Run this to verify your package structure
and get next steps.
"""

import os
import sys

def check_file(path, description):
    """Check if a file exists and print status."""
    exists = os.path.exists(path)
    symbol = "✅" if exists else "❌"
    print(f"{symbol} {description}: {path}")
    return exists

def main():
    print("🔍 Checking package structure...\n")
    
    # Check all important files
    files_to_check = [
        ("pyproject.toml", "Package configuration"),
        ("README.md", "Documentation"),
        ("LICENSE", "License file"),
        ("src/reasoning_evaluator/__init__.py", "Package init"),
        ("src/reasoning_evaluator/models.py", "Data models"),
        ("src/reasoning_evaluator/utils.py", "Utility functions"),
        ("src/reasoning_evaluator/evaluator.py", "Main evaluator"),
        ("src/reasoning_evaluator/cli.py", "CLI interface"),
    ]
    
    all_exist = all(check_file(f, desc) for f, desc in files_to_check)
    
    print("\n" + "="*60)
    
    if all_exist:
        print("✅ All files present!")
    else:
        print("⚠️  Some files are missing")
    
    print("\n📋 Next Steps:\n")
    
    # Check if evaluator.py has real code or just template
    evaluator_path = "src/reasoning_evaluator/evaluator.py"
    if os.path.exists(evaluator_path):
        with open(evaluator_path, 'r') as f:
            content = f.read()
            has_real_code = "NotImplementedError" not in content
    else:
        has_real_code = False
    
    if not has_real_code:
        print("1. ⚠️  COMPLETE evaluator.py:")
        print("   - Open: src/reasoning_evaluator/evaluator.py")
        print("   - Copy your FullEvaluator and ChainComparator classes from notebook")
        print("   - See HOW_TO_CONVERT.md for detailed instructions")
        print()
    else:
        print("1. ✅ evaluator.py is complete!")
        print()
    
    print("2. 📦 Install the package:")
    print("   cd reasoning-evaluator")
    print("   pip install -e .[semantic]")
    print()
    
    print("3. 🧪 Test it works:")
    print("   python -c 'import reasoning_evaluator; print(reasoning_evaluator.__version__)'")
    print()
    
    print("4. 🚀 Use in your code:")
    print("   from reasoning_evaluator import FullEvaluator, load_dataset")
    print()
    
    print("5. 📖 Read the documentation:")
    print("   - SUMMARY.md - Quick overview")
    print("   - HOW_TO_CONVERT.md - Detailed instructions")
    print("   - README.md - Full package documentation")
    print("   - INSTALL.md - Installation guide")
    
    print("\n" + "="*60)
    print("\n💡 Tip: Run 'pip install -e .[semantic]' to install with all features")
    print()

if __name__ == "__main__":
    # Change to package directory if script is run from parent
    if os.path.exists("reasoning-evaluator"):
        os.chdir("reasoning-evaluator")
    
    main()
