"""Utilities for loading and parsing reasoning chain data."""

import json
from typing import Dict, List, Tuple, Optional
from reasoning_evaluator.models import ReasoningQuestion, ReasoningChain


def parse_chain_from_dict(data: Dict, chain_type: str) -> Optional[ReasoningChain]:
    """Parse chain from dictionary format.
    
    Args:
        data: Dictionary containing chain data
        chain_type: Either 'good' or 'weak'
        
    Returns:
        ReasoningChain object or None if parsing fails
    """
    question_key = 'good_question_chain' if chain_type == 'good' else 'weak_question_chain'
    if question_key not in data:
        return None

    ground_truth = data.get('ground_truth', '')
    problem = ground_truth.split('\n')[0] if ground_truth else 'Problem context'

    questions = []
    context = problem
    for i, q_text in enumerate(data[question_key]):
        q = ReasoningQuestion(text=q_text, step_number=i+1, context=context, answer=None)
        questions.append(q)
        context += f"\nQ{i+1}: {q_text}"

    return ReasoningChain(
        problem=problem,
        questions=questions,
        final_answer='',
        ground_truth=ground_truth,
        problem_id=data.get('problem_id'),
        chain_type=chain_type
    )


def load_dataset(file_path: str) -> Tuple[List[ReasoningChain], List[ReasoningChain]]:
    """Load dataset and return (good_chains, weak_chains).
    
    Args:
        file_path: Path to JSON dataset file
        
    Returns:
        Tuple of (good_chains, weak_chains)
    """
    with open(file_path, 'r') as f:
        raw_data = json.load(f)

    good_chains = []
    weak_chains = []

    for problem_data in raw_data:
        good = parse_chain_from_dict(problem_data, 'good')
        if good:
            good_chains.append(good)
        weak = parse_chain_from_dict(problem_data, 'weak')
        if weak:
            weak_chains.append(weak)

    return good_chains, weak_chains
