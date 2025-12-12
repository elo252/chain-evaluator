# Reasoning Evaluator

A comprehensive 12-dimension reasoning question evaluator with semantic analysis capabilities.

## Features

- **12-Dimension Evaluation**: Analyzes questions across clarity, specificity, contextfulness, necessity, progression, non-repetition, answerability, structural coherence, semantic coherence, diagnostic quality, neutrality, and semantic relevance
- **Hybrid Scoring**: Combines heuristic and scientific (MLM/semantic) approaches using agreement-based weighting
- **Visualizations**: Generates heatmaps, radar plots, and progression charts
- **Chain Comparison**: Compare good vs weak reasoning chains
- **Extensible**: Easy to integrate into your own workflows

## Installation

### Basic Installation

```bash
pip install reasoning-evaluator
```

### With Semantic Analysis (Recommended)

For full semantic and MLM-based features:

```bash
pip install reasoning-evaluator[semantic]
```

### Development Installation

```bash
git clone https://github.com/yourusername/reasoning-evaluator.git
cd reasoning-evaluator
pip install -e .[dev,semantic]
```

## Quick Start

### Basic Usage

```python
from reasoning_evaluator import ReasoningQuestion, ReasoningChain, FullEvaluator

# Create a reasoning question
question = ReasoningQuestion(
    text="What is the total cost of the series movies?",
    step_number=1,
    context="There are 600 movies, with 1/3 in series..."
)

# Create a chain
chain = ReasoningChain(
    problem="Calculate total movie costs",
    questions=[question],
    final_answer="$4400",
    problem_id="example_1",
    chain_type="good"
)

# Evaluate
evaluator = FullEvaluator(use_semantic_model=True)
report = evaluator.evaluate_chain(chain, save_dir="results")
```

### Using the CLI

```bash
# Evaluate a dataset
reasoning-eval evaluate dataset.json --output results/

# Compare chains
reasoning-eval compare dataset.json --output comparisons/
```

### Dataset Format

Your JSON dataset should follow this structure:

```json
[
  {
    "problem_id": "problem_0",
    "ground_truth": "Solution description...",
    "good_question_chain": [
      "Question 1?",
      "Question 2?"
    ],
    "weak_question_chain": [
      "Weak question 1?",
      "Weak question 2?"
    ]
  }
]
```

## 12 Evaluation Dimensions

1. **Clarity**: Wording precision and absence of ambiguity
2. **Specificity**: Presence of concrete entities and details
3. **Contextfulness**: Relevance to previous context
4. **Necessity**: Whether the question advances reasoning
5. **Progression**: Smooth complexity/difficulty flow
6. **Non-Repetition**: Novelty compared to prior questions
7. **Answerability**: Presence of actionable prompts
8. **Structural Coherence**: Logical connection to previous steps
9. **Semantic Coherence**: Semantic similarity to chain context
10. **Diagnostic**: Verification/checking capability
11. **Neutrality**: Absence of leading language
12. **Semantic Relevance**: Alignment with problem context

## Advanced Features

### Custom Weights

```python
evaluator = FullEvaluator()
evaluator.final_weights = {
    'clarity': 0.15,
    'necessity': 0.15,
    # ... customize other weights
}
```

### Chain Comparison

```python
from reasoning_evaluator import ChainComparator

comparator = ChainComparator()
comparator.compare_chains(good_chains, weak_chains, save_dir="comparison_results")
```

### Visualization Only

```python
evaluator._plot_chain_heatmap(chain, "heatmap.png")
evaluator._plot_chain_radar(chain, "radar.png")
evaluator._plot_progression_chart(chain, "progression.png")
```

## Output Files

When you run evaluation with `save_dir`, you'll get:

- `evaluation_report.json`: Detailed scores and diagnostics
- `dimension_heatmap.png`: Visual heatmap of all dimensions
- `dimension_radar.png`: Radar plot of average scores
- `progression.png`: Line chart showing metric progression
- `comparisons/`: Cross-chain comparison visualizations

## Requirements

### Core Dependencies
- numpy >= 1.20.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- pandas >= 1.3.0

### Optional (for semantic features)
- sentence-transformers >= 2.0.0
- transformers >= 4.20.0
- torch >= 1.9.0

## Citation

If you use this in your research, please cite:

```bibtex
@software{reasoning_evaluator,
  title={Reasoning Evaluator: 12-Dimension Question Quality Assessment},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/reasoning-evaluator}
}
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions welcome! Please open an issue or PR on GitHub.

## Support

For questions or issues, please open a GitHub issue.
