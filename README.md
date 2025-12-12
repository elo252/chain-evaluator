Chain Evaluator
A comprehensive 12-dimension reasoning chain evaluator with semantic, structural, and ground-truth–aligned analysis. Designed to evaluate, score, and compare multi-step reasoning chains with expert-level rigor.
Developed by: Dhruv Gada
Email: dhruv_gada@brown.edu
GitHub: https://github.com/elo252/chain-evaluator

Features
12-Dimension Question Evaluation
Each question is analyzed across:

Clarity – How understandable is the question?
Specificity – Is the question precise and focused?
Contextfulness – Does it leverage available context?
Necessity – Is it essential to solving the problem?
Progression – Does it advance the reasoning chain?
Non-Repetition – Avoids redundancy with previous questions
Answerability – Can it be answered with given information?
Structural Coherence – Logical flow within the chain
Semantic Coherence – Meaningful connections between questions
Diagnostic Quality – Ability to identify critical information
Neutrality – Unbiased and objective formulation
Semantic Relevance – Direct relevance to the problem

Expert Evaluation Architecture

Hybrid heuristic + semantic + NLI scoring
Ground-truth anchored necessity & progression scoring
Fact extraction + redundancy detection
Chain-of-thought structural flow analysis

Visualizations

Heatmaps
Radar plots
Progression charts
Chain-vs-Chain comparison visualizations

Chain Comparison Mode

Compare Chain A vs Chain B
Good vs Weak chain analysis
Dimension breakdown
Coverage & necessity analysis
Statistical significance testing
Automatic visual results


Installation
Colab / Local Development
bashgit clone https://github.com/elo252/chain-evaluator.git
cd chain-evaluator
pip install -e .[dev,semantic]
If extras fail (Colab sometimes breaks extras):
bashpip install -e .
pip install sentence-transformers transformers torch matplotlib seaborn pandas scipy

Quick Start
pythonfrom chain_evaluator import ReasoningQuestion, ReasoningChain, FullEvaluator

# Create a reasoning question
question = ReasoningQuestion(
    text="What is the total cost of the series movies?",
    step_number=1,
    context="There are 600 movies, with 1/3 in series..."
)

# Create a reasoning chain
chain = ReasoningChain(
    problem="Calculate total movie costs",
    questions=[question],
    final_answer="$4400",
    problem_id="example_1",
    chain_type="good"
)

# Evaluate the chain
evaluator = FullEvaluator(use_semantic_model=True)
report = evaluator.evaluate_chain(chain, save_dir="results")

CLI Usage
Evaluate a dataset:
bashreasoning-eval evaluate dataset.json --output results/
Compare chains:
bashreasoning-eval compare dataset.json --output comparisons/
(If CLI does not work, please open an issue — easy fix.)

Dataset Format
json[
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

Advanced Usage
Custom Weights
pythonevaluator.final_weights["necessity"] = 0.20
Chain Comparison
pythonfrom chain_evaluator import ChainComparator

ChainComparator.compare_chains(
    good_chains, 
    weak_chains, 
    save_dir="comparison_results"
)
Generate Visualizations
pythonevaluator._plot_chain_heatmap(chain, "heatmap.png")
evaluator._plot_chain_radar(chain, "radar.png")
evaluator._plot_progression_chart(chain, "progression.png")

Output Files

evaluation_report.json
dimension_heatmap.png
dimension_radar.png
progression.png
comparisons/ directory for chain comparisons


Requirements
Core Dependencies

numpy
scipy
pandas
matplotlib
seaborn

Semantic (Optional)

torch
transformers
sentence-transformers
spacy


Citation
bibtex@software{chain_evaluator,
  title={Chain Evaluator: 12-Dimension Reasoning Chain Quality Assessment},
  author={Gada, Dhruv},
  year={2025},
  url={https://github.com/elo252/chain-evaluator}
}

Contributing
Pull requests and issues welcome at:
https://github.com/elo252/chain-evaluator/issues

Support
Email: dhruv_gada@brown.edu

License
MIT License (or specify your license)
