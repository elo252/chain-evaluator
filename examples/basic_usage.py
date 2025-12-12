"""
Example: Basic usage of the reasoning evaluator package
"""

from reasoning_evaluator import (
    FullEvaluator,
    ReasoningQuestion,
    ReasoningChain,
    ChainComparator,
    load_dataset
)


def example_1_single_question():
    """Evaluate a single question."""
    print("Example 1: Evaluating a single question")
    print("=" * 60)
    
    # Create a question
    question = ReasoningQuestion(
        text="How many movies are in series if there are 600 movies total and 1/3 are in series?",
        step_number=1,
        context="There are 600 movies in total. 1/3 are in series."
    )
    
    # Create evaluator (without semantic model for speed)
    evaluator = FullEvaluator(use_semantic_model=False)
    
    # Evaluate clarity
    clarity_score, diag = evaluator.score_clarity(question)
    print(f"\nClarity Score: {clarity_score:.3f}")
    print(f"Diagnostics: {diag}")
    print()


def example_2_chain_evaluation():
    """Evaluate a complete reasoning chain."""
    print("\nExample 2: Evaluating a complete chain")
    print("=" * 60)
    
    # Create a chain of questions
    questions = [
        ReasoningQuestion(
            text="How many movies are in series?",
            step_number=1,
            context="There are 600 movies. 1/3 are in series."
        ),
        ReasoningQuestion(
            text="What is the cost per series movie if series cost 60% of $10?",
            step_number=2,
            context="Series movies cost 60% of normal price ($10)."
        ),
        ReasoningQuestion(
            text="What is the total cost for all series movies?",
            step_number=3,
            context="200 series movies at $6 each."
        ),
    ]
    
    chain = ReasoningChain(
        problem="Calculate the total cost of all movies",
        questions=questions,
        final_answer="$4400",
        problem_id="example_1",
        chain_type="good"
    )
    
    # Evaluate the chain
    evaluator = FullEvaluator(use_semantic_model=False)
    report = evaluator.evaluate_chain(chain, save_dir="example_output")
    
    print(f"\nChain Score: {report.chain_score:.3f}")
    print(f"Recommendations: {report.recommendations}")
    print()


def example_3_load_dataset():
    """Load and evaluate a dataset from JSON."""
    print("\nExample 3: Loading and evaluating a dataset")
    print("=" * 60)
    
    # First, create an example dataset file
    import json
    
    example_dataset = [
        {
            "problem_id": "problem_0",
            "ground_truth": "Solution: 600/3=200 series movies...",
            "good_question_chain": [
                "How many movies are in series?",
                "What is the cost of series movies?",
                "What is the total cost?"
            ],
            "weak_question_chain": [
                "Did you calculate it correctly?",
                "Is the answer right?"
            ]
        }
    ]
    
    # Save dataset
    with open("example_dataset.json", "w") as f:
        json.dump(example_dataset, f, indent=2)
    
    # Load and evaluate
    good_chains, weak_chains = load_dataset("example_dataset.json")
    
    print(f"\nLoaded {len(good_chains)} good chains and {len(weak_chains)} weak chains")
    
    # Evaluate all chains
    evaluator = FullEvaluator(use_semantic_model=False)
    
    for chain in good_chains[:1]:  # Just first chain for example
        report = evaluator.evaluate_chain(chain)
        print(f"Good chain score: {report.chain_score:.3f}")
    
    for chain in weak_chains[:1]:
        report = evaluator.evaluate_chain(chain)
        print(f"Weak chain score: {report.chain_score:.3f}")
    
    print()


def example_4_comparison():
    """Compare good vs weak chains."""
    print("\nExample 4: Comparing good vs weak chains")
    print("=" * 60)
    
    # Load dataset (using example from above)
    good_chains, weak_chains = load_dataset("example_dataset.json")
    
    # Evaluate all chains first
    evaluator = FullEvaluator(use_semantic_model=False)
    
    for chain in good_chains + weak_chains:
        evaluator.evaluate_chain(chain)
    
    # Compare
    ChainComparator.compare_chains(
        good_chains,
        weak_chains,
        save_dir="comparison_output"
    )
    
    print("\n✓ Comparison complete! Check comparison_output/ for visualizations")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("REASONING EVALUATOR - USAGE EXAMPLES")
    print("=" * 60)
    
    try:
        example_1_single_question()
        example_2_chain_evaluation()
        example_3_load_dataset()
        example_4_comparison()
        
        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("=" * 60)
        print("\nGenerated files:")
        print("  - example_dataset.json")
        print("  - example_output/ (chain evaluation results)")
        print("  - comparison_output/ (comparison visualizations)")
        print()
        
    except NotImplementedError as e:
        print("\n" + "=" * 60)
        print("⚠️  Examples cannot run yet!")
        print("=" * 60)
        print(f"\nReason: {e}")
        print("\nTo fix this:")
        print("1. Open src/reasoning_evaluator/evaluator.py")
        print("2. Copy your FullEvaluator and ChainComparator classes from the notebook")
        print("3. Save the file")
        print("4. Run: pip install -e .")
        print("5. Try this script again!")
        print()
