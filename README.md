Reasoning Evaluator

A comprehensive 12-dimension reasoning question evaluator with semantic,
structural, and ground-truth–aligned analysis.
Designed to evaluate, score, and compare multi-step reasoning chains with
expert-level rigor.

Developed by: Dhruv Gada
Email: dhruv_gada@brown.edu
GitHub: https://github.com/elo252

-------------------------------------------------------------------------
FEATURES
-------------------------------------------------------------------------

12-Dimension Evaluation:
Each question is analyzed across:
- Clarity
- Specificity
- Contextfulness
- Necessity
- Progression
- Non-Repetition
- Answerability
- Structural Coherence
- Semantic Coherence
- Diagnostic Quality
- Neutrality
- Semantic Relevance

Expert Evaluation Architecture:
- Hybrid heuristic + semantic + NLI scoring
- Ground-truth anchored necessity and progression analysis
- Fact extraction + redundancy detection
- Role-based structural flow validation

Visualizations:
- Heatmaps
- Radar plots
- Progression charts
- Cross-chain comparison visuals

Chain Comparison Mode:
- Compare Chain A vs Chain B
- Good vs Weak chain analysis
- Full dimension breakdown
- Statistical significance testing
- Automatic result visualizations

-------------------------------------------------------------------------
INSTALLATION
-------------------------------------------------------------------------

!git clone https://github.com/elo252/chain-evaluator.git
%cd chain-evaluator
!pip install -e .[dev,semantic]


-------------------------------------------------------------------------
QUICK START
-------------------------------------------------------------------------

Example:

from reasoning_evaluator import ReasoningQuestion, ReasoningChain, FullEvaluator

question = ReasoningQuestion(
    text="What is the total cost of the series movies?",
    step_number=1,
    context="There are 600 movies, with 1/3 in series..."
)

chain = ReasoningChain(
    problem="Calculate total movie costs",
    questions=[question],
    final_answer="$4400",
    problem_id="example_1",
    chain_type="good"
)

evaluator = FullEvaluator(use_semantic_model=True)
report = evaluator.evaluate_chain(chain, save_dir="results")

-------------------------------------------------------------------------
CLI USAGE
-------------------------------------------------------------------------

Evaluate a dataset:
  reasoning-eval evaluate dataset.json --output results/

Compare chains:
  reasoning-eval compare dataset.json --output comparisons/

-------------------------------------------------------------------------
DATASET FORMAT
-------------------------------------------------------------------------

[
  {
    "problem_id": "problem_0",
    "ground_truth": "Step-by-step ground truth reasoning...",
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

-------------------------------------------------------------------------
ADVANCED USAGE
-------------------------------------------------------------------------

Custom Weights:

evaluator.final_weights["necessity"] = 0.20

Chain Comparison:

from reasoning_evaluator import ChainComparator
ChainComparator.compare_chains(good_chains, weak_chains, save_dir="comparison_results")

Visualizations:

evaluator._plot_chain_heatmap(chain, "heatmap.png")
evaluator._plot_chain_radar(chain, "radar.png")
evaluator._plot_progression_chart(chain, "progression.png")

-------------------------------------------------------------------------
OUTPUT FILES
-------------------------------------------------------------------------

- evaluation_report.json
- dimension_heatmap.png
- dimension_radar.png
- progression.png
- comparisons/ folder with cross-chain visualizations

-------------------------------------------------------------------------
REQUIREMENTS
-------------------------------------------------------------------------

Core:
- numpy
- scipy
- pandas
- matplotlib
- seaborn

Semantic (optional):
- torch
- transformers
- sentence-transformers
- spacy

-------------------------------------------------------------------------
CITATION
-------------------------------------------------------------------------

@software{reasoning_evaluator,
  title={Reasoning Evaluator: 12-Dimension Question Quality Assessment},
  author={Gada, Dhruv},
  year={2025},
  url={https://github.com/elo252/reasoning-evaluator}
}

-------------------------------------------------------------------------
CONTRIBUTING
-------------------------------------------------------------------------

Pull requests and issues welcome at:
https://github.com/elo252/reasoning-evaluator/issues

-------------------------------------------------------------------------
SUPPORT
-------------------------------------------------------------------------

Email: dhruv_gada@brown.edu

