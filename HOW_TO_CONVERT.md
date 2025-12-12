# Converting Your Notebook to a PIP Package

Your reasoning evaluator notebook has been set up as a pip-installable package! Here's how to complete the conversion and use it:

## 📦 Package Structure Created

```
reasoning-evaluator/
├── src/
│   └── reasoning_evaluator/
│       ├── __init__.py          # Package initialization
│       ├── models.py             # Data classes  
│       ├── utils.py              # Helper functions
│       └── evaluator.py          # Main evaluator (YOU NEED TO ADD THIS)
├── pyproject.toml                # Modern Python packaging config
├── setup.py                      # Legacy setup file
├── README.md                     # Documentation
├── LICENSE                       # MIT License
└── INSTALL.md                    # Installation guide
```

## 🚀 Quick Setup Steps

### Step 1: Complete the evaluator.py file

The main evaluator code from your notebook needs to be added to:
```
src/reasoning_evaluator/evaluator.py
```

**Option A - Manual Copy** (Recommended):
1. Open your `Untitled19.ipynb` notebook
2. Copy ALL the Python code (everything except the example at the end)
3. Paste it into `src/reasoning_evaluator/evaluator.py`
4. Add these imports at the top:
```python
import os
import re
import json
import time
import math
from typing import List, Dict, Tuple, Optional
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from reasoning_evaluator.models import ReasoningQuestion, ReasoningChain, EvaluationReport

try:
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except Exception:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
```

**Option B - Automated** (if you're comfortable with Python):
```bash
cd "/Users/tanishpatwa/Library/Containers/net.whatsapp.WhatsApp/Data/tmp/documents/E9DC2552-A84F-48A3-A021-9A6F9853501F"
python3 convert_notebook_to_package.py
```

### Step 2: Install the package locally

```bash
cd reasoning-evaluator

# Basic installation
pip install -e .

# OR with semantic features (recommended)
pip install -e .[semantic]
```

### Step 3: Test it works

```bash
python3 -c "import reasoning_evaluator; print('✓ Package installed successfully!')"
```

## 📝 Using Your Package

### In Python scripts:

```python
from reasoning_evaluator import (
    FullEvaluator,
    ReasoningQuestion,
    ReasoningChain,
    ChainComparator,
    load_dataset
)

# Load your dataset
good_chains, weak_chains = load_dataset("your_dataset.json")

# Create evaluator
evaluator = FullEvaluator(use_semantic_model=True)

# Evaluate chains
for chain in good_chains:
    report = evaluator.evaluate_chain(chain, save_dir="results")
    print(f"Chain {chain.problem_id}: {report.chain_score:.3f}")

# Compare chains
ChainComparator.compare_chains(good_chains, weak_chains, save_dir="comparisons")
```

### From command line (after adding CLI):

```bash
reasoning-eval evaluate dataset.json --output results/
reasoning-eval compare dataset.json --output comparisons/
```

## 🌐 Publishing to PyPI (Optional)

To make it available for `pip install reasoning-evaluator`:

### 1. Update package metadata

Edit `pyproject.toml`:
- Change `name`, `authors`, `email`
- Update `homepage` and `repository` URLs

### 2. Build the package

```bash
pip install build twine
python -m build
```

### 3. Upload to PyPI

```bash
# Test on TestPyPI first
twine upload --repository testpypi dist/*

# Then real PyPI
twine upload dist/*
```

Now anyone can install with:
```bash
pip install reasoning-evaluator
```

## 🔧 Troubleshooting

### "Module not found" errors

Make sure you're in the `reasoning-evaluator` directory and run:
```bash
pip install -e .[semantic]
```

### Dependency issues

Install dependencies manually:
```bash
pip install numpy scipy matplotlib seaborn pandas sentence-transformers transformers torch
```

### Can't find evaluator.py

You need to copy your notebook code into `src/reasoning_evaluator/evaluator.py`. See Step 1 above.

## 📚 What's Next?

1. ✅ Complete `evaluator.py` (Step 1 above)
2. ✅ Install locally (`pip install -e .`)
3. ✅ Test with your datasets
4. 🎯 Add CLI interface (optional)
5. 🌐 Publish to PyPI (optional)
6. 📖 Add more documentation
7. ✨ Add unit tests

## 💡 Tips

- Use `pip install -e .` for development (changes reflect immediately)
- Add example scripts in `examples/` directory
- Write tests in `tests/` directory  
- Use semantic versioning for releases (0.1.0, 0.2.0, 1.0.0, etc.)

## 🆘 Need Help?

Check these files:
- `README.md` - Full documentation
- `INSTALL.md` - Detailed installation guide
- `pyproject.toml` - Package configuration

---

**Made with your Jupyter notebook!** 🎉
