# Installation Guide

## Quick Start

### 1. Install the package locally

From the `reasoning-evaluator` directory:

```bash
# Basic installation (without semantic features)
pip install -e .

# With semantic analysis (recommended)
pip install -e .[semantic]

# For development
pip install -e .[dev,semantic]
```

### 2. Verify installation

```bash
python -c "import reasoning_evaluator; print(reasoning_evaluator.__version__)"
```

### 3. Run example

```python
from reasoning_evaluator import FullEvaluator, ReasoningQuestion, ReasoningChain

# Create a question
q = ReasoningQuestion(
    text="What is the total cost?",
    step_number=1,
    context="Problem context here"
)

# Create a chain
chain = ReasoningChain(
    problem="Calculate costs",
    questions=[q],
    final_answer="$100",
    problem_id="test_1",
    chain_type="good"
)

# Evaluate
evaluator = FullEvaluator(use_semantic_model=False)  # Set to True if you installed [semantic]
report = evaluator.evaluate_chain(chain)
print(f"Chain score: {report.chain_score:.3f}")
```

## Publishing to PyPI (Optional)

To make your package available via `pip install reasoning-evaluator`:

### 1. Build the package

```bash
pip install build twine
python -m build
```

### 2. Upload to Test PyPI (recommended first)

```bash
twine upload --repository testpypi dist/*
```

### 3. Test installation from Test PyPI

```bash
pip install --index-url https://test.pypi.org/simple/ reasoning-evaluator
```

### 4. Upload to PyPI

```bash
twine upload dist/*
```

Now anyone can install with:
```bash
pip install reasoning-evaluator
```

## Troubleshooting

###Dependency issues

If you get errors about missing dependencies:

```bash
# Install all dependencies manually
pip install numpy scipy matplotlib seaborn pandas

# For semantic features
pip install sentence-transformers transformers torch
```

### ImportError

If you get import errors after installation:

```bash
# Reinstall in editable mode
pip uninstall reasoning-evaluator
pip install -e .
```

### Permission errors

```bash
# Use --user flag
pip install --user -e .
```

## Next Steps

1. See `README.md` for usage examples
2. Check `examples/` directory for sample scripts
3. Read the API documentation (coming soon)
