#!/usr/bin/env python3
"""
Test script to evaluate the example dataset
"""
import json
from reasoning_evaluator import FullEvaluator, load_dataset

# Create example dataset
example_data = [
    {
        "problem_id": "problem_0",
        "ground_truth": "There are 600/3=200 movies in series. So he pays 200*6=$1200. He has 600-200=400 movies not in series. 400*0.4=160 movies are old. So those cost 160*5=$800. 400-160=240 movies are neither old nor in series. They cost 240*10=$2400. So the total cost is 1200+800+2400=$4400",
        "good_question_chain": [
            "From the total of 600 movies, how did you compute the number of movies that belong to series, given that a third of the movies are in series?",
            "Once you determined how many movies are in series, how did you translate the statement about series costing 60% of the normal price into a per-movie cost and then into a total cost for all series movies?",
            "After removing the series movies from the total, how many movies remain, and how did you apply the 40% figure to find how many of these remaining movies are older $5 movies?",
            "How did you determine how many movies were neither in series nor older, and what unit price did you apply to compute their total cost?",
            "When you combined the three partial costs (series, older, and remaining normal-price movies), what total did you get, and how did you verify that the total number of movies accounted for matches the original 600?"
        ],
        "weak_question_chain": [
            "Did you separate the movies into series, older, and regular categories before calculating the cost?",
            "Did you multiply the number of movies in each category by the correct price for that category?"
        ]
    }
]

# Save to file
print("Creating example_dataset.json...")
with open("example_dataset.json", "w") as f:
    json.dump(example_data, f, indent=2)
print("✓ Dataset created\n")

# Load dataset
print("Loading dataset...")
good_chains, weak_chains = load_dataset("example_dataset.json")
print(f"✓ Loaded {len(good_chains)} good chains and {len(weak_chains)} weak chains\n")

# Create evaluator (without semantic model for faster testing)
print("Creating evaluator (without semantic models for faster testing)...")
evaluator = FullEvaluator(use_semantic_model=True)
print("✓ Evaluator created\n")

# Evaluate good chain
print("="*80)
print("EVALUATING GOOD CHAIN")
print("="*80)
good_report = evaluator.evaluate_chain(good_chains[0], save_dir="test_results")

# Evaluate weak chain
print("\n" + "="*80)
print("EVALUATING WEAK CHAIN")
print("="*80)
weak_report = evaluator.evaluate_chain(weak_chains[0], save_dir="test_results")

# Compare
print("\n" + "="*80)
print("COMPARISON")
print("="*80)
from reasoning_evaluator import ChainComparator
ChainComparator.compare_chains(good_chains, weak_chains, save_dir="test_results/comparisons")

print("\n" + "="*80)
print("✅ EVALUATION COMPLETE!")
print("="*80)
print(f"Good chain average quality: {good_report.chain_score:.3f}")
print(f"Weak chain average quality: {weak_report.chain_score:.3f}")
print(f"Difference: {good_report.chain_score - weak_report.chain_score:.3f}")
print(f"\nResults saved to: test_results/")
print("="*80)
