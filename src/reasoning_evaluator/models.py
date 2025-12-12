"""Data models for reasoning evaluation."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class ReasoningQuestion:
    """Represents a single reasoning question with its evaluation scores."""
    
    text: str
    step_number: int
    context: str
    answer: Optional[str] = None

    # 12 dimensions
    clarity: float = 0.0
    specificity: float = 0.0
    contextfulness: float = 0.0
    necessity: float = 0.0
    progression: float = 0.0
    non_repetition: float = 0.0
    answerability: float = 0.0
    structural_coherence: float = 0.0
    semantic_coherence: float = 0.0
    diagnostic: float = 0.0
    neutrality: float = 0.0
    semantic_relevance: float = 0.0

    # meta
    overall: float = 0.0
    confidences: Dict[str, float] = field(default_factory=dict)
    explanation: str = ""
    diagnostics: Dict = field(default_factory=dict)


@dataclass
class ReasoningChain:
    """Represents a chain of reasoning questions."""
    
    problem: str
    questions: List[ReasoningQuestion]
    final_answer: str
    ground_truth: Optional[str] = None
    problem_id: Optional[str] = None
    chain_type: str = "unknown"  # 'good' or 'weak'

    avg_quality: float = 0.0
    coverage_score: float = 0.0
    diagnostics: Dict = field(default_factory=dict)


@dataclass
class EvaluationReport:
    """Report containing evaluation results."""
    
    question_scores: List[Dict]
    chain_score: float
    diagnostics: Dict
    recommendations: List[str]
