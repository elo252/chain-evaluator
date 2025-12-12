"""

1) Structural Coherence (Hybrid / Relative)
   - Uses local NLI + entities + roles for baseline coherence.
   - Then scales that by a factor depending on how the chain length compares
     to a TARGET length shared between the two chains (good vs weak).

2) Necessity (Ground-Truth Anchored)
   - Each question's necessity is based on how many NEW ground-truth facts it helps cover.
   - Novelty is computed against the union of ground-truth facts, minus what earlier questions already covered.
   - Combined with GT alignment similarity + non-repetition.

3) Progression (Ground-Truth Step Alignment)
   - Ground truth is split into sentences (steps).
   - Each question is matched to the GT sentence with highest semantic similarity.
   - Progression rewards moving forward through GT steps and penalizes regressions/big jumps.


Dependencies (recommended):

    pip install numpy scipy matplotlib seaborn pandas sentence-transformers spacy transformers torch
    python -m spacy download en_core_web_sm

"""

import os
import re
import json
import time
import math
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# -------------------------
# NLP libraries
# -------------------------
try:
    import spacy
    SPACY_AVAILABLE = True
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    SPACY_AVAILABLE = False
    nlp = None
    print(f"Warning: spaCy not available - {e}")

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False
    print("Note: transformers not available (NLI disabled)")

try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except Exception:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Note: sentence-transformers not available (semantic similarity falls back to Jaccard)")


# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class ReasoningQuestion:
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

    # fact-based
    facts: Set[str] = field(default_factory=set)
    entities: Set[str] = field(default_factory=set)
    relations: List[Tuple[str, str, str]] = field(default_factory=list)

    # structural role (partition / transform / aggregate / verify / other)
    role: str = "other"

    # GT mapping
    gt_index: Optional[int] = None  # best-matching ground-truth sentence index


@dataclass
class ReasoningChain:
    problem: str
    questions: List[ReasoningQuestion]
    final_answer: str
    ground_truth: Optional[str] = None
    problem_id: Optional[str] = None
    chain_type: str = "unknown"

    avg_quality: float = 0.0
    coverage_score: float = 0.0
    diagnostics: Dict = field(default_factory=dict)

    # ground-truth processed data
    gt_sentences: List[str] = field(default_factory=list)
    gt_sentence_facts: List[Set[str]] = field(default_factory=list)
    gt_all_facts: Set[str] = field(default_factory=set)


@dataclass
class EvaluationReport:
    question_scores: List[Dict]
    chain_score: float
    diagnostics: Dict
    recommendations: List[str]


# =============================================================================
# LogicAnalyzer (NLI + facts + entities + redundancy)
# =============================================================================

class LogicAnalyzer:
    """
    Handles:
      - NLI entailment / contradiction (using real MNLI models when available)
      - fact extraction (simple)
      - entity extraction
      - relation extraction (optional)
      - redundancy via fact overlap + NLI
    """
    def __init__(self):
        self.nli_model = None
        self.nli_tokenizer = None
        self.label_map = None  # maps logits index -> 'contradiction' / 'neutral' / 'entailment'

        if TRANSFORMERS_AVAILABLE:
            self._load_nli_model()

    def _load_nli_model(self):
        """Try a few NLI checkpoints, stop at first that loads."""
        candidate_models = [
            "roberta-large-mnli",
            "facebook/bart-large-mnli",
        ]

        for name in candidate_models:
            try:
                print(f"Loading NLI model: {name}")
                self.nli_tokenizer = AutoTokenizer.from_pretrained(name)
                self.nli_model = AutoModelForSequenceClassification.from_pretrained(name)
                self.nli_model.eval()
                self._build_label_map()
                print(f"✓ NLI model loaded: {name}\n")
                return
            except Exception as e:
                print(f"  ↳ Failed to load {name}: {e}")
                self.nli_model = None
                self.nli_tokenizer = None

        print("Warning: No NLI model available; logical entailment features disabled.")

    def _build_label_map(self):
        """Build robust mapping from model labels to {contradiction, neutral, entailment}."""
        id2label = getattr(self.nli_model.config, "id2label", None)
        if not id2label:
            self.label_map = {0: "contradiction", 1: "neutral", 2: "entailment"}
            return

        norm = {i: id2label[i].lower() for i in id2label}
        mapping = {}
        for idx, name in norm.items():
            if "entail" in name:
                mapping[idx] = "entailment"
            elif "contrad" in name:
                mapping[idx] = "contradiction"
            elif "neutral" in name:
                mapping[idx] = "neutral"

        remaining = set(norm.keys()) - set(mapping.keys())
        default_order = ["contradiction", "neutral", "entailment"]
        for idx, label in zip(sorted(remaining), default_order):
            mapping[idx] = label

        self.label_map = mapping

    def check_entailment(self, premise: str, hypothesis: str) -> Tuple[str, float]:
        """
        Check if 'premise' entails / contradicts / is neutral wrt 'hypothesis'.
        Returns: (label, confidence)
        """
        if not self.nli_model or not self.nli_tokenizer or not self.label_map:
            return "neutral", 0.5

        premise = (premise or "").strip()
        hypothesis = (hypothesis or "").strip()
        if not premise or not hypothesis:
            return "neutral", 0.5

        try:
            with torch.no_grad():
                inputs = self.nli_tokenizer(
                    premise,
                    hypothesis,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                )
                outputs = self.nli_model(**inputs)
                probs = torch.softmax(outputs.logits[0], dim=-1)
                idx = int(torch.argmax(probs).item())
                label = self.label_map.get(idx, "neutral")
                conf = float(probs[idx].item())
                return label, conf
        except Exception as e:
            print(f"NLI error: {e}")
            return "neutral", 0.5

    def extract_facts(self, text: str) -> Set[str]:
        """Simple fact extraction: sentences + shallow SVO triples (if spaCy available)."""
        text = (text or "").strip()
        if not text:
            return set()

        if not SPACY_AVAILABLE or not nlp:
            return {s.strip() for s in re.split(r"[.!?]", text) if s.strip()}

        doc = nlp(text)
        facts = set()

        # sentence-level facts
        for sent in doc.sents:
            s = sent.text.strip()
            if s:
                facts.add(s)

        # simple subject-verb-object patterns
        for token in doc:
            if token.dep_ in ("nsubj", "nsubjpass"):
                subj = token.text
                verb = token.head.text
                obj = None
                for child in token.head.children:
                    if child.dep_ in ("dobj", "attr", "pobj"):
                        obj = child.text
                        break
                if obj:
                    facts.add(f"{subj} {verb} {obj}")

        return facts

    def extract_entities(self, text: str) -> Set[str]:
        """Named entities + numbers + proper nouns (fallback: numbers + capitalized words)."""
        text = (text or "").strip()
        if not text:
            return set()

        if not SPACY_AVAILABLE or not nlp:
            entities = set(re.findall(r"\d+(?:\.\d+)?", text))
            entities.update(re.findall(r"\b[A-Z][a-z]+\b", text))
            return entities

        doc = nlp(text)
        entities = set()

        for ent in doc.ents:
            entities.add(ent.text)

        for tok in doc:
            if tok.pos_ in ("NUM", "PROPN") or tok.like_num:
                entities.add(tok.text)

        return entities

    def extract_relations(self, text: str) -> List[Tuple[str, str, str]]:
        """Optional: extract (subject, relation, object) triples."""
        if not SPACY_AVAILABLE or not nlp:
            return []

        doc = nlp(text)
        triples = []
        for token in doc:
            if token.dep_ in ("nsubj", "nsubjpass"):
                subj = token.text
                rel = token.head.text
                for child in token.head.children:
                    if child.dep_ in ("dobj", "attr", "pobj"):
                        triples.append((subj, rel, child.text))
                        break
        return triples

    @staticmethod
    def _fact_overlap(f1: Set[str], f2: Set[str]) -> float:
        if not f1 or not f2:
            return 0.0
        inter = len(f1 & f2)
        union = len(f1 | f2)
        return inter / union if union > 0 else 0.0

    def is_redundant(
        self,
        current_text: str,
        prior_texts: List[str],
        current_facts: Set[str],
        prior_facts: Set[str],
        threshold: float = 0.85,
    ) -> Tuple[bool, Dict]:
        """
        Decide if current step is redundant w.r.t. previous steps.
        Uses fact overlap + NLI entailment.
        """
        diagnostics = {}
        fact_overlap = self._fact_overlap(current_facts, prior_facts)
        diagnostics["fact_overlap"] = fact_overlap

        # NLI: do recent steps entail current?
        max_entail_conf = 0.0
        entail_count = 0

        for prev in prior_texts[-3:]:
            label, conf = self.check_entailment(prev, current_text)
            if label == "entailment":
                entail_count += 1
                max_entail_conf = max(max_entail_conf, conf)

        diagnostics["max_entail_conf"] = max_entail_conf
        diagnostics["entail_count"] = entail_count

        is_redundant = (fact_overlap >= threshold and max_entail_conf >= 0.7) or (entail_count >= 2)
        diagnostics["is_redundant"] = is_redundant
        return is_redundant, diagnostics


# =============================================================================
# FullEvaluator — Expert Reasoning Mode
# =============================================================================

class FullEvaluator:
    def __init__(self, use_semantic_model: bool = True):
        # vocabulary / heuristics
        self.vague_terms = {"it", "that", "this", "these", "those", "thing", "stuff", "part"}
        self.wh_words = {"what", "which", "how", "why", "when", "where"}
        self.leading_patterns = [
            r"(is|are|should).*(right|correct|true)\s*\?",
            r"the answer is.*\?",
            r"since.*,\s*(is|are|should)\s+\d+.*\?",
        ]
        self.diagnostic_patterns = [
            r"(verify|check|confirm|test|validate)",
            r"plug.*back|substitute",
            r"does.*satisfy|hold true",
            r"what if|suppose",
            r"(why|how) (does|did|can)",
        ]
        self.math_entities = [
            "equation", "variable", "value", "solve", "calculate", "formula",
            "step", "operation", "result", "total", "sum", "difference", "product",
            "quotient", "area", "perimeter", "angle", "length", "width", "height", "volume",
        ]

        # complexity weights (for progression if we need fallback)
        self.complexity_weights = {
            "ops": 0.35,
            "vars": 0.25,
            "length": 0.15,
            "numbers": 0.15,
            "rare_words": 0.10,
        }

        # final overall weighting
        self.final_weights = {
            "clarity": 0.10,
            "specificity": 0.10,
            "contextfulness": 0.08,
            "answerability": 0.09,
            "diagnostic": 0.08,
            "neutrality": 0.05,
            "semantic_relevance": 0.08,
            "semantic_coherence": 0.08,
            "structural_coherence": 0.08,
            "progression": 0.12,
            "non_repetition": 0.07,
            "necessity": 0.07,
        }

        # Semantic model
        self.use_semantic = use_semantic_model and SENTENCE_TRANSFORMERS_AVAILABLE
        self.model = None
        if self.use_semantic:
            try:
                print("Loading semantic model 'all-MiniLM-L6-v2' ...")
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
                print("✓ Semantic model loaded\n")
            except Exception as e:
                print("Warning: semantic model failed to load:", e)
                self.use_semantic = False
                self.model = None

        # Logic analyzer
        self.logic_analyzer = LogicAnalyzer()

    # ----------------- basic helpers -----------------

    @staticmethod
    def _tokens(text: str) -> List[str]:
        return re.findall(r"\b[a-zA-Z0-9_]+\b", (text or "").lower())

    @staticmethod
    def _word_count(text: str) -> int:
        return len((text or "").split())

    @staticmethod
    def _rare_word_score(text: str) -> float:
        toks = re.findall(r"\b[a-zA-Z]{4,}\b", (text or ""))
        if not toks:
            return 0.0
        long_words = [t for t in toks if len(t) > 7]
        return len(long_words) / max(1, len(toks))

    def _extract_concepts(self, text: str) -> List[str]:
        text_lower = (text or "").lower()
        concepts = []

        numbers = re.findall(r"\d+(?:\.\d+)?", text or "")
        concepts.extend([f"num_{n}" for n in numbers])

        if any(op in text_lower for op in ["+", "add", "sum", "total"]):
            concepts.append("addition")
        if any(op in text_lower for op in ["-", "subtract", "difference", "remain"]):
            concepts.append("subtraction")
        if any(op in text_lower for op in ["*", "multiply", "product", "times"]):
            concepts.append("multiplication")
        if any(op in text_lower for op in ["/", "divide", "quotient", "per"]):
            concepts.append("division")

        for ent in self.math_entities:
            if ent in text_lower:
                concepts.append(ent)

        tokens = self._tokens(text or "")
        concepts.extend([t for t in tokens if len(t) <= 2 and t.isalpha()])
        return concepts

    def _semantic_raw(self, a: str, b: str, cache: Dict[str, object] = None) -> float:
        if self.use_semantic and self.model is not None:
            try:
                if cache is not None:
                    if a not in cache:
                        cache[a] = self.model.encode(a, convert_to_tensor=True)
                    if b not in cache:
                        cache[b] = self.model.encode(b, convert_to_tensor=True)
                    va = cache[a]
                    vb = cache[b]
                else:
                    va = self.model.encode(a, convert_to_tensor=True)
                    vb = self.model.encode(b, convert_to_tensor=True)
                sim = util.cos_sim(va, vb).cpu().numpy().ravel()
                return float(np.mean(sim)) if sim.size > 0 else 0.0
            except Exception:
                pass

        a_toks = set(self._tokens(a))
        b_toks = set(self._tokens(b))
        if not a_toks and not b_toks:
            return 0.0
        j = len(a_toks & b_toks) / max(1, len(a_toks | b_toks))
        return 2.0 * j - 1.0

    def _semantic(self, a: str, b: str, cache: Dict[str, object] = None) -> float:
        """Normalize [-1,1] → [0,1]."""
        return (self._semantic_raw(a, b, cache) + 1.0) / 2.0

    # ----------------- role classification -----------------

    def classify_step_role(self, text: str) -> str:
        t = (text or "").lower()
        if any(kw in t for kw in ["how many", "split", "divide", "portion"]):
            return "partition"
        if any(kw in t for kw in ["percent", "percentage", "multiply", "cost per", "apply", "rate"]):
            return "transform"
        if any(kw in t for kw in ["total", "sum", "add", "combine", "altogether"]):
            return "aggregate"
        if any(kw in t for kw in ["verify", "check", "does this match", "confirm", "correct"]):
            return "verify"
        return "other"

    # ----------------- GT preprocessing -----------------

    def _preprocess_ground_truth(self, chain: ReasoningChain):
        """Split ground truth into sentences & fact sets for necessity & progression."""
        if chain.gt_sentences:
            return

        gt_text = (chain.ground_truth or chain.problem or "").strip()
        if not gt_text:
            chain.gt_sentences = []
            chain.gt_sentence_facts = []
            chain.gt_all_facts = set()
            return

        if SPACY_AVAILABLE and nlp is not None:
            doc = nlp(gt_text)
            sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
        else:
            sentences = [s.strip() for s in re.split(r"[.!?]", gt_text) if s.strip()]

        chain.gt_sentences = sentences
        sentence_facts = [self.logic_analyzer.extract_facts(s) for s in sentences]
        chain.gt_sentence_facts = sentence_facts
        all_facts = set()
        for s_facts in sentence_facts:
            all_facts |= s_facts
        chain.gt_all_facts = all_facts

    # ----------------- dimension scoring -----------------

    def score_clarity(self, q: ReasoningQuestion) -> Tuple[float, Dict]:
        text = (q.text or "").strip()
        diag: Dict = {}
        if not text:
            return 0.0, {"empty": True}

        wc = self._word_count(text)
        diag["word_count"] = wc
        penalties = 0.0

        if wc < 4:
            penalties += 0.4
            diag["too_short"] = True
        if wc > 50:
            penalties += 0.2
            diag["very_long"] = True

        vague_count = sum(
            1 for v in self.vague_terms
            if re.search(r"\b" + re.escape(v) + r"\b", text.lower())
        )
        diag["vague_count"] = vague_count
        if vague_count:
            penalties += 0.25

        math_count = sum(1 for me in self.math_entities if me in text.lower())
        diag["math_count"] = math_count
        if math_count == 0:
            penalties += 0.05

        score = float(np.clip(1.0 - penalties, 0.0, 1.0))
        diag["final"] = score
        return score, diag

    def score_specificity(self, q: ReasoningQuestion) -> Tuple[float, Dict]:
        t = (q.text or "").lower()
        toks = self._tokens(t)
        has_number = bool(re.search(r"\d", t))
        has_var = any(len(tok) <= 2 and tok.isalpha() for tok in toks)
        rare = self._rare_word_score(t)

        diag = {
            "has_number": has_number,
            "has_var": has_var,
            "rare_prop": rare,
        }

        base = 0.3
        if has_number:
            base += 0.3
        if has_var:
            base += 0.2
        base += 0.2 * rare

        score = float(np.clip(base, 0.0, 1.0))
        diag["final"] = score
        return score, diag

    def score_contextfulness(self, q: ReasoningQuestion) -> Tuple[float, Dict]:
        t = (q.text or "").lower()
        ctx = (q.context or "").lower()
        q_tokens = set(self._tokens(t))
        ctx_tokens = set(self._tokens(ctx))
        overlap = len(q_tokens & ctx_tokens)
        denom = max(1, len(q_tokens))
        score = float(np.clip(overlap / denom, 0.0, 1.0))
        return score, {"overlap": overlap, "q_tokens": len(q_tokens), "ctx_tokens": len(ctx_tokens)}

    def score_answerability(self, q: ReasoningQuestion) -> Tuple[float, Dict]:
        t = (q.text or "").strip().lower()

        # yes/no questions are answerable but weaker diagnostically; we treat them as lower
        starts_yesno = t.startswith("did ") or t.startswith("does ") or t.startswith("is ") or t.startswith("are ")
        has_wh = any(t.startswith(w) for w in self.wh_words)
        has_action = any(w in t for w in ["calculate", "solve", "find", "determine", "compute", "what is"])

        diag = {"has_wh": has_wh, "has_action": has_action, "yesno_like": starts_yesno}

        if starts_yesno:
            base = 0.25  # still answerable, but less strong
        else:
            base = 0.3

        base += 0.35 * int(has_wh) + 0.4 * int(has_action)
        score = float(np.clip(base, 0.0, 1.0))
        return score, diag

    def score_diagnostic(self, q: ReasoningQuestion) -> Tuple[float, Dict]:
        t = (q.text or "").lower()
        s = 0.0
        matched = []
        for pat in self.diagnostic_patterns:
            if re.search(pat, t):
                s += 0.3
                matched.append(pat)
        if re.search(r"\b(why|how)\b", t):
            s += 0.2
            matched.append("why_how")
        diag = {"matched": matched}
        return float(np.clip(s, 0.0, 1.0)), diag

    def score_neutrality(self, q: ReasoningQuestion) -> Tuple[float, Dict]:
        t = (q.text or "").lower()
        score = 1.0
        hits = []

        for pat in self.leading_patterns:
            if re.search(pat, t):
                score -= 0.3
                hits.append(pat)

        # yes/no leading structures
        if t.startswith("did ") or t.startswith("does ") or t.startswith("is ") or t.startswith("are "):
            score -= 0.2
            hits.append("yesno_leading")

        if re.search(r"(answer|result|value)\s+is\s+\d+", t):
            score -= 0.3
            hits.append("numeric_suggest")
        if any(k in t for k in ["right", "correct", "confirm that", "agree"]):
            score -= 0.15
            hits.append("confirmation_word")

        score = float(np.clip(score, 0.0, 1.0))
        return score, {"hits": hits}

    def step_complexity(self, text: str) -> float:
        t = text or ""
        num_ops = len(re.findall(r"[\+\-\*\/=]", t))
        var_count = len([tok for tok in self._tokens(t) if tok.isalpha() and len(tok) <= 2])
        length = max(1, self._word_count(t))
        numbers = len(re.findall(r"\d+", t))
        rare = self._rare_word_score(t)

        comp_ops = min(1.0, num_ops / 3.0)
        comp_vars = min(1.0, var_count / 3.0)
        comp_length = min(1.0, length / 30.0)
        comp_numbers = min(1.0, numbers / 3.0)
        comp_rare = rare

        weighted = (
            self.complexity_weights["ops"] * comp_ops
            + self.complexity_weights["vars"] * comp_vars
            + self.complexity_weights["length"] * comp_length
            + self.complexity_weights["numbers"] * comp_numbers
            + self.complexity_weights["rare_words"] * comp_rare
        )
        return float(np.clip(weighted, 0.0, 1.0))

    def score_semantic_coherence(
        self,
        q: ReasoningQuestion,
        previous_qs: List[ReasoningQuestion],
        emb_cache: Dict[str, object],
    ) -> Tuple[float, Dict]:
        diag: Dict = {}
        if not previous_qs:
            return 0.5, {"no_previous": True}

        prev_texts = [pq.text for pq in previous_qs[-2:]]
        base_sims = [self._semantic(q.text, pt, cache=emb_cache) for pt in prev_texts]
        base_avg = float(np.clip(np.mean(base_sims) if base_sims else 0.5, 0.0, 1.0))
        diag["base_prev_sims"] = base_sims
        diag["base_avg_sim"] = base_avg
        return base_avg, diag

    def score_non_repetition(
        self,
        q: ReasoningQuestion,
        prior_qs: List[ReasoningQuestion],
        emb_cache: Dict[str, object],
    ) -> Tuple[float, Dict]:
        if not prior_qs:
            return 1.0, {"first": True}

        diag: Dict = {}
        current_facts = self.logic_analyzer.extract_facts(q.text)
        q.facts = current_facts
        diag["current_facts_count"] = len(current_facts)

        prior_facts = set()
        prior_texts = []
        for pq in prior_qs:
            if not pq.facts:
                pq.facts = self.logic_analyzer.extract_facts(pq.text)
            prior_facts.update(pq.facts)
            prior_texts.append(pq.text)

        diag["prior_facts_count"] = len(prior_facts)

        is_redundant, red_diag = self.logic_analyzer.is_redundant(
            q.text, prior_texts, current_facts, prior_facts
        )
        diag.update(red_diag)

        if is_redundant:
            base = 0.2
        else:
            fact_overlap = red_diag.get("fact_overlap", 0.0)
            base = 1.0 - 0.5 * fact_overlap

        base = float(np.clip(base, 0.0, 1.0))
        diag["final_non_repetition"] = base
        return base, diag

    def score_structural_coherence(
        self,
        q: ReasoningQuestion,
        prev_q: Optional[ReasoningQuestion],
        chain_length: int,
        target_length: float,
    ) -> Tuple[float, Dict]:
        diag: Dict = {}
        if prev_q is None:
            # still apply length factor
            base = 1.0
        else:
            # NLI logical relation (prev -> current)
            label, conf = self.logic_analyzer.check_entailment(prev_q.text, q.text)
            diag["nli_label"] = label
            diag["nli_confidence"] = conf

            if label == "contradiction":
                entail_score = 0.15 * (1.0 - conf)
            elif label == "entailment":
                entail_score = 0.4 + 0.6 * conf
            else:  # neutral
                entail_score = 0.4

            # entity anchoring
            if not q.entities:
                q.entities = self.logic_analyzer.extract_entities(q.text)
            if not prev_q.entities:
                prev_q.entities = self.logic_analyzer.extract_entities(prev_q.text)

            shared = q.entities & prev_q.entities
            retained_ratio = len(shared) / max(1, len(prev_q.entities))
            entity_score = 0.6 * retained_ratio + 0.4  # 0.4–1.0
            diag["entity_overlap"] = list(shared)
            diag["entity_score"] = entity_score

            # concept flow
            curr_concepts = set(self._extract_concepts(q.text))
            prev_concepts = set(self._extract_concepts(prev_q.text))
            overlap = len(curr_concepts & prev_concepts)
            total = len(curr_concepts | prev_concepts)
            concept_score = overlap / max(1, total)
            diag["concept_score"] = concept_score

            # role transition
            q.role = self.classify_step_role(q.text)
            prev_role = getattr(prev_q, "role", self.classify_step_role(prev_q.text))

            allowed_transitions = {
                "partition": {"transform", "aggregate"},
                "transform": {"aggregate", "verify"},
                "aggregate": {"verify"},
                "verify": set(),
                "other": {"partition", "transform", "other"},
            }

            if q.role in allowed_transitions.get(prev_role, set()):
                role_score = 1.0
            else:
                role_score = 0.6

            diag["role_transition"] = f"{prev_role} → {q.role}"
            diag["role_score"] = role_score

            base = (
                0.4 * entail_score
                + 0.25 * entity_score
                + 0.2 * concept_score
                + 0.15 * role_score
            )

        # specificity gate: vague steps can't be "strongly coherent"
        specificity, _ = self.score_specificity(q)
        diag["specificity_gate"] = specificity
        if specificity < 0.5:
            base *= 0.6

        # Relative chain-length factor (Hybrid Mode C)
        # target_length is the average length of the two chains (good & weak) for that problem.
        if target_length is None or target_length <= 0:
            length_factor = 1.0
        else:
            gap_ratio = (chain_length - target_length) / max(target_length, 1.0)
            if gap_ratio < 0:
                # shorter than target → stronger penalty (under-scaffold)
                length_factor = 1.0 + gap_ratio  # e.g., half as long => 0.5
            else:
                # longer than target → mild penalty (over-fragmentation)
                length_factor = 1.0 - 0.5 * gap_ratio  # 1.5x => 0.75
            length_factor = float(np.clip(length_factor, 0.4, 1.05))

        diag["length_factor"] = length_factor
        diag["target_length"] = target_length
        diag["chain_length"] = chain_length

        final = float(np.clip(base * length_factor, 0.0, 1.0))
        diag["final_structural_coherence"] = final
        return final, diag

    def score_semantic_relevance(self, q: ReasoningQuestion, emb_cache: Dict[str, object]) -> Tuple[float, Dict]:
        s = self._semantic(q.text, q.context, cache=emb_cache)
        return s, {"semantic_similarity": s}

    def compute_progression_gt(self, chain: ReasoningChain, emb_cache: Dict[str, object]) -> List[float]:
        """
        Progression based on ground-truth step alignment.

        For each question Qi:
          - Find best-matching GT sentence index k.
          - Progression rewards monotonic movement forward in k.
        """
        qs = chain.questions
        n = len(qs)
        if n == 0:
            return []

        if not chain.gt_sentences:
            # fall back to complexity-based if no GT
            return [0.7 for _ in qs]

        # 1) find GT index for each question
        gt_indices = []
        for q in qs:
            sims = [self._semantic(q.text, s, cache=emb_cache) for s in chain.gt_sentences]
            if sims:
                best_idx = int(np.argmax(sims))
                best_sim = float(max(sims))
            else:
                best_idx = None
                best_sim = 0.0
            q.gt_index = best_idx
            if "gt_alignment" not in q.diagnostics:
                q.diagnostics["gt_alignment"] = {}
            q.diagnostics["gt_alignment"]["best_index"] = best_idx
            q.diagnostics["gt_alignment"]["best_sim"] = best_sim
            gt_indices.append(best_idx)

        # 2) progression score from index differences
        scores = []
        for i, q in enumerate(qs):
            idx_i = gt_indices[i]
            if idx_i is None:
                scores.append(0.5)
                continue

            if i == 0:
                # first question – baseline if it aligns somewhere reasonable
                scores.append(0.8)
                continue

            prev_idx = gt_indices[i - 1]
            if prev_idx is None:
                scores.append(0.6)
                continue

            delta = idx_i - prev_idx
            if delta == 1:
                score = 1.0
            elif delta > 1:
                score = max(0.5, 1.0 - 0.15 * (delta - 1))  # further jumps penalized
            elif delta == 0:
                score = 0.6
            else:  # going backward
                score = max(0.1, 0.3 - 0.1 * abs(delta))

            scores.append(float(np.clip(score, 0.0, 1.0)))

        return scores

    def score_necessity(
        self,
        chain: ReasoningChain,
        idx: int,
        emb_cache: Dict[str, object],
    ) -> Tuple[float, Dict]:
        """
        Necessity = GT-anchored novelty + GT alignment, gated by non-repetition.

        - GT novelty: how many NEW GT facts this question covers that previous questions didn't.
        - GT alignment: semantic similarity to best GT sentence.
        - non_repetition: from redundancy module.
        """
        qs = chain.questions
        q = qs[idx]
        diag: Dict = {}

        # ensure facts
        if not q.facts:
            q.facts = self.logic_analyzer.extract_facts(q.text)

        if not chain.gt_all_facts:
            diag["no_gt_facts"] = True
            # fallback: mid necessity
            return 0.5, diag

        gt_facts = chain.gt_all_facts

        # overlap with GT
        overlap_all = q.facts & gt_facts

        # prior coverage
        prior_gt_facts = set()
        for j in range(idx):
            prior_gt_facts |= (qs[j].facts & gt_facts)

        novel_gt_facts = overlap_all - prior_gt_facts
        novelty = len(novel_gt_facts) / max(1, len(gt_facts))

        diag["gt_overlap_all"] = len(overlap_all)
        diag["novel_gt_facts"] = len(novel_gt_facts)
        diag["novelty_ratio"] = novelty

        # GT alignment: best semantic similarity to GT sentences
        if chain.gt_sentences:
            sims = [self._semantic(q.text, s, cache=emb_cache) for s in chain.gt_sentences]
            best_sim = float(max(sims)) if sims else 0.0
        else:
            best_sim = self._semantic(q.text, chain.ground_truth or chain.problem or "", cache=emb_cache)

        diag["gt_best_sim"] = best_sim

        base = 0.6 * novelty + 0.4 * best_sim

        # Non-repetition gate (if not already computed, treat as neutral gate)
        non_rep = q.non_repetition if q.non_repetition > 0 else 1.0
        diag["non_repetition_gate"] = non_rep

        final = base * non_rep

        # Last-step bonus: if aligned with final GT step / uses "total" / "verify" etc.
        if idx == len(qs) - 1:
            t = (q.text or "").lower()
            if any(k in t for k in ["total", "final", "answer", "overall", "check", "verify"]) or best_sim > 0.8:
                final += 0.1

        final = float(np.clip(final, 0.0, 1.0))
        diag["final_necessity"] = final
        return final, diag

    # ----------------- coverage & explanation -----------------

    def evaluate_coverage(self, chain: ReasoningChain) -> Tuple[float, Dict]:
        qs = chain.questions
        if not qs:
            return 0.0, {"error": "no_questions"}

        texts = [q.text.lower() for q in qs]
        unique_texts = len(set(texts))
        redundancy = 1.0 - (unique_texts / len(texts))

        # approximate problem complexity by GT sentences count
        problem_complexity = max(len(chain.gt_sentences), 1)
        expected_steps = max(1, int(problem_complexity * 0.8))  # slightly lenient

        coverage_ratio = min(1.0, len(qs) / expected_steps)
        score = coverage_ratio * (1.0 - 0.5 * redundancy)
        diag = {
            "redundancy": redundancy,
            "coverage_ratio": coverage_ratio,
            "problem_complexity": problem_complexity,
        }
        return float(np.clip(score, 0.0, 1.0)), diag

    def _explain(self, q: ReasoningQuestion) -> str:
        parts = []

        # Necessity view
        if q.necessity > 0.7:
            parts.append("Strongly necessary: it captures new ground-truth reasoning not covered by earlier questions.")
        elif q.necessity > 0.4:
            parts.append("Moderately necessary: it contributes some ground-truth facts but may overlap with previous steps.")
        else:
            parts.append("Weak necessity: it adds little new ground-truth content or repeats earlier reasoning.")

        # Clarity
        if q.clarity < 0.5:
            parts.append("Wording is somewhat unclear or underspecified.")
        else:
            parts.append("Wording is reasonably clear.")

        # Redundancy
        if q.non_repetition < 0.5:
            parts.append("It significantly overlaps with prior questions.")

        # Progression
        if q.progression < 0.4:
            parts.append("Sequence-wise, it disrupts the progression through the solution (e.g., jumping backward or skipping).")
        elif q.progression > 0.8:
            parts.append("It fits naturally into the flow of the ground-truth solution.")

        # Diagnostic
        if q.diagnostic > 0.5:
            parts.append("Provides diagnostic value (asks for justification / verification).")

        # Neutrality
        if q.neutrality < 0.5:
            parts.append("May be leading or suggestive, rather than neutral.")

        conf = q.confidences.get("aggregate")
        if conf is not None:
            parts.append(f"Score confidence: {conf:.2f}.")

        return " ".join(parts)

    # ----------------- JSON sanitization -----------------

    def _sanitize_for_json(self, obj):
        """Recursively convert numpy types, etc., to JSON-safe types."""
        if isinstance(obj, dict):
            return {self._sanitize_for_json(k): self._sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize_for_json(x) for x in obj]
        elif isinstance(obj, tuple):
            return tuple(self._sanitize_for_json(x) for x in obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif hasattr(obj, "item"):
            try:
                return obj.item()
            except Exception:
                return str(obj)
        else:
            return obj

    # ----------------- pretty printing -----------------

    def _print_question_results(self, questions: List[ReasoningQuestion]):
        print("\n" + "=" * 80)
        print("PER-QUESTION EVALUATION RESULTS")
        print("=" * 80)

        for q in questions:
            print(f"\n{'─' * 80}")
            print(f"Question {q.step_number}:")
            print(f"  Text: {q.text}")
            print(f"{'─' * 80}")
            print(f"  Overall Score: {q.overall:.3f} (Confidence: {q.confidences.get('aggregate', 0):.3f})")
            print("\n  Dimension Scores:")
            print(f"    Clarity:              {q.clarity:.3f}")
            print(f"    Specificity:          {q.specificity:.3f}")
            print(f"    Contextfulness:       {q.contextfulness:.3f}")
            print(f"    Answerability:        {q.answerability:.3f}")
            print(f"    Diagnostic:           {q.diagnostic:.3f}")
            print(f"    Neutrality:           {q.neutrality:.3f}")
            print(f"    Semantic Relevance:   {q.semantic_relevance:.3f}")
            print(f"    Structural Coherence: {q.structural_coherence:.3f}")
            print(f"    Semantic Coherence:   {q.semantic_coherence:.3f}")
            print(f"    Necessity:            {q.necessity:.3f}")
            print(f"    Progression:          {q.progression:.3f}")
            print(f"    Non-Repetition:       {q.non_repetition:.3f}")
            print(f"\n  Explanation: {q.explanation}")

    def _print_chain_results(self, chain: ReasoningChain, recommendations: List[str]):
        print("\n" + "=" * 80)
        print("CHAIN-LEVEL SUMMARY")
        print("=" * 80)
        print(f"  Chain Type: {chain.chain_type.upper()}")
        print(f"  Problem ID: {chain.problem_id}")
        print(f"  Total Questions: {len(chain.questions)}")
        print(f"  Average Quality Score: {chain.avg_quality:.3f}")
        print(f"  Coverage Score: {chain.coverage_score:.3f}")

        scores = [q.overall for q in chain.questions]
        if scores:
            print("\n  Score Distribution:")
            print(f"    Min:    {min(scores):.3f}")
            print(f"    Max:    {max(scores):.3f}")
            print(f"    Median: {np.median(scores):.3f}")
            print(f"    Std:    {np.std(scores):.3f}")

        cov = chain.diagnostics.get("coverage_diag", {})
        print("\n  Coverage Diagnostics:")
        print(f"    Redundancy:         {cov.get('redundancy', 0):.3f}")
        print(f"    Coverage Ratio:     {cov.get('coverage_ratio', 0):.3f}")
        print(f"    Problem Complexity: {cov.get('problem_complexity', 0):.0f}")

        if recommendations:
            print("\n  Recommendations:")
            for rec in recommendations:
                print(f"    • {rec}")

        print("=" * 80)

    # ----------------- plotting helpers -----------------

    def _plot_chain_heatmap(self, chain: ReasoningChain, save_path: str):
        """Heatmap of all 12 dimensions across questions."""
        dims = [
            "clarity", "specificity", "contextfulness", "necessity", "progression",
            "non_repetition", "answerability", "structural_coherence", "semantic_coherence",
            "diagnostic", "neutrality", "semantic_relevance",
        ]

        matrix = []
        for q in chain.questions:
            row = [getattr(q, dim, 0.0) for dim in dims]
            matrix.append(row)

        fig, ax = plt.subplots(figsize=(14, max(6, len(chain.questions) * 0.6)))
        im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

        ax.set_xticks(np.arange(len(dims)))
        ax.set_yticks(np.arange(len(chain.questions)))
        ax.set_xticklabels([d.replace("_", "\n") for d in dims], fontsize=9)
        ax.set_yticklabels([f"Q{i+1}" for i in range(len(chain.questions))])

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        for i in range(len(chain.questions)):
            for j in range(len(dims)):
                val = matrix[i][j]
                color = "white" if val < 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)

        ax.set_title(
            f"12-Dimension Heatmap: {chain.problem_id} ({chain.chain_type})",
            fontweight="bold",
            pad=20,
        )
        plt.colorbar(im, ax=ax, label="Score")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_chain_radar(self, chain: ReasoningChain, save_path: str):
        """Radar plot showing average dimension scores."""
        dims = [
            "clarity", "specificity", "contextfulness", "necessity", "progression",
            "non_repetition", "answerability", "structural_coherence", "semantic_coherence",
            "diagnostic", "neutrality", "semantic_relevance",
        ]
        avg_scores = [np.mean([getattr(q, dim, 0.0) for q in chain.questions]) for dim in dims]

        angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
        avg_scores += avg_scores[:1]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))
        ax.plot(angles, avg_scores, "o-", linewidth=2)
        ax.fill(angles, avg_scores, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([d.replace("_", " ").title() for d in dims], fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title(
            f"Average Dimension Profile: {chain.problem_id} ({chain.chain_type})",
            fontweight="bold",
            pad=20,
            fontsize=12,
        )
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_progression_chart(self, chain: ReasoningChain, save_path: str):
        """Line chart showing progression of key metrics."""
        steps = [q.step_number for q in chain.questions]
        overall = [q.overall for q in chain.questions]
        necessity = [q.necessity for q in chain.questions]
        progression = [q.progression for q in chain.questions]
        non_rep = [q.non_repetition for q in chain.questions]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Top plot: overall score
        ax1.plot(steps, overall, "o-", linewidth=2, label="Overall Score")
        ax1.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
        ax1.set_ylabel("Score")
        ax1.set_ylim([0, 1])
        ax1.set_title(
            f"Question Quality Progression: {chain.problem_id} ({chain.chain_type})",
            fontweight="bold",
        )
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Bottom plot: key dimensions
        ax2.plot(steps, necessity, "o-", linewidth=2, label="Necessity")
        ax2.plot(steps, progression, "s-", linewidth=2, label="Progression")
        ax2.plot(steps, non_rep, "^-", linewidth=2, label="Non-Repetition")
        ax2.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
        ax2.set_xlabel("Question Number")
        ax2.set_ylabel("Score")
        ax2.set_ylim([0, 1])
        ax2.set_title("Key Chain-Level Dimensions", fontweight="bold")
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def _save_chain_outputs(self, chain: ReasoningChain, report: EvaluationReport, save_dir: str):
        """Save per-chain outputs and visualizations."""
        chain_dir = os.path.join(save_dir, f"{chain.problem_id}_{chain.chain_type}")
        os.makedirs(chain_dir, exist_ok=True)

        # Save JSON report
        json_path = os.path.join(chain_dir, "evaluation_report.json")
        with open(json_path, "w") as f:
            safe_payload = self._sanitize_for_json(
                {
                    "chain_info": {
                        "problem_id": chain.problem_id,
                        "chain_type": chain.chain_type,
                        "avg_quality": chain.avg_quality,
                        "coverage_score": chain.coverage_score,
                    },
                    "question_scores": report.question_scores,
                    "diagnostics": chain.diagnostics,
                    "recommendations": report.recommendations,
                }
            )
            json.dump(safe_payload, f, indent=2)

        # Generate visualizations
        self._plot_chain_heatmap(chain, os.path.join(chain_dir, "dimension_heatmap.png"))
        self._plot_chain_radar(chain, os.path.join(chain_dir, "dimension_radar.png"))
        self._plot_progression_chart(chain, os.path.join(chain_dir, "progression.png"))

        print(f"  ✓ Saved outputs to {chain_dir}/")

    # ----------------- main chain evaluation -----------------

    def evaluate_chain(
        self,
        chain: ReasoningChain,
        save_dir: Optional[str] = None,
        target_length: Optional[float] = None,
    ) -> EvaluationReport:
        """
        Evaluate a single chain.
        target_length is the relative chain-length target (average length of paired chains).
        """
        start = time.time()
        emb_cache: Dict[str, object] = {}
        qs = chain.questions

        print("\n" + "=" * 80)
        print(f"EVALUATING CHAIN: {chain.problem_id} ({chain.chain_type.upper()})")
        print("=" * 80)
        print(f"Problem: {chain.problem[:120]}...")
        print(f"Questions: {len(qs)}")
        print()

        # preprocess GT
        self._preprocess_ground_truth(chain)

        # 1) per-question local metrics
        for i, q in enumerate(qs):
            q.clarity, d_cl = self.score_clarity(q)
            q.specificity, d_sp = self.score_specificity(q)
            q.contextfulness, d_ctx = self.score_contextfulness(q)
            q.answerability, d_ans = self.score_answerability(q)
            q.diagnostic, d_diag = self.score_diagnostic(q)
            q.neutrality, d_neu = self.score_neutrality(q)
            q.semantic_relevance, d_semrel = self.score_semantic_relevance(q, emb_cache)

            prev = qs[i - 1] if i > 0 else None
            q.structural_coherence, d_struct = self.score_structural_coherence(
                q, prev, chain_length=len(qs), target_length=target_length or len(qs)
            )
            q.semantic_coherence, d_semcoh = self.score_semantic_coherence(q, qs[:i], emb_cache)

            q.diagnostics.update(
                {
                    "clarity": d_cl,
                    "specificity": d_sp,
                    "context": d_ctx,
                    "answerability": d_ans,
                    "diagnostic": d_diag,
                    "neutrality": d_neu,
                    "semantic_relevance": d_semrel,
                    "structural_coherence": d_struct,
                    "semantic_coherence": d_semcoh,
                }
            )

        # 2) non-repetition
        for i, q in enumerate(qs):
            prior = qs[:i]
            q.non_repetition, d_nr = self.score_non_repetition(q, prior, emb_cache)
            q.diagnostics["non_repetition"] = d_nr

        # 3) necessity (GT-anchored)
        for i, q in enumerate(qs):
            q.necessity, d_n = self.score_necessity(chain, i, emb_cache)
            q.diagnostics["necessity"] = d_n

        # 4) progression (GT-anchored)
        progression_scores = self.compute_progression_gt(chain, emb_cache)
        for i, q in enumerate(qs):
            q.progression = progression_scores[i]

        # 5) overall + confidences + explanations
        question_scores = []
        for q in qs:
            total = 0.0
            dim_values = []
            for dim, w in self.final_weights.items():
                val = getattr(q, dim, 0.5)
                dim_values.append(val)
                total += w * float(val)

            q.overall = float(np.clip(total, 0.0, 1.0))
            std = float(np.std(dim_values))
            base_conf = float(np.clip(1.0 - std, 0.0, 1.0))

            # embedding-based stability (optional)
            emb_conf = 0.0
            if self.use_semantic and self.model is not None and q.step_number > 1:
                prev_texts = [pq.text for pq in qs[max(0, q.step_number - 3): q.step_number - 1]]
                sims = [self._semantic(q.text, p, cache=emb_cache) for p in prev_texts] if prev_texts else []
                if sims:
                    emb_conf = float(np.clip(1.0 - np.std(sims), 0.0, 1.0))

            agg_conf = float(np.clip(0.7 * base_conf + 0.3 * emb_conf, 0.0, 1.0))
            q.confidences = {"aggregate": agg_conf, "base_std": std, "emb_conf": emb_conf}
            q.explanation = self._explain(q)

            question_scores.append(
                {
                    "index": q.step_number,
                    "text": q.text,
                    "scores": {
                        "clarity": round(q.clarity, 3),
                        "specificity": round(q.specificity, 3),
                        "contextfulness": round(q.contextfulness, 3),
                        "answerability": round(q.answerability, 3),
                        "diagnostic": round(q.diagnostic, 3),
                        "neutrality": round(q.neutrality, 3),
                        "semantic_relevance": round(q.semantic_relevance, 3),
                        "structural_coherence": round(q.structural_coherence, 3),
                        "semantic_coherence": round(q.semantic_coherence, 3),
                        "necessity": round(q.necessity, 3),
                        "progression": round(q.progression, 3),
                        "non_repetition": round(q.non_repetition, 3),
                        "overall": round(q.overall, 3),
                        "confidence": round(agg_conf, 3),
                    },
                    "diagnostics": q.diagnostics,
                    "explanation": q.explanation,
                }
            )

        # 6) chain-level stats
        dims = [
            "clarity", "specificity", "contextfulness", "necessity", "progression",
            "non_repetition", "answerability", "structural_coherence", "semantic_coherence",
            "diagnostic", "neutrality", "semantic_relevance",
        ]
        matrix = {d: [getattr(q, d) for q in qs] for d in dims}
        corr = {}
        for i in range(len(dims)):
            for j in range(i + 1, len(dims)):
                a = matrix[dims[i]]
                b = matrix[dims[j]]
                try:
                    rho, p = stats.spearmanr(a, b)
                    corr[f"{dims[i]}__{dims[j]}"] = {
                        "rho": float(rho) if not math.isnan(rho) else None,
                        "p": float(p) if not math.isnan(p) else None,
                    }
                except Exception:
                    corr[f"{dims[i]}__{dims[j]}"] = {"rho": None, "p": None}

        coverage_score, coverage_diag = self.evaluate_coverage(chain)
        chain.coverage_score = coverage_score
        chain.avg_quality = float(np.mean([q.overall for q in qs])) if qs else 0.0
        chain.diagnostics = {
            "correlations": corr,
            "coverage_diag": coverage_diag,
            "n_questions": len(qs),
        }

        recommendations = []
        if chain.avg_quality < 0.5:
            recommendations.append("Chain avg quality < 0.5 — inspect low scoring questions.")
        if coverage_score < 0.6:
            recommendations.append("Coverage low — consider adding more intermediate steps (not just rephrases).")

        end = time.time()
        print(f"\n✓ Evaluation completed in {end - start:.2f}s")
        print(f"  Average question quality: {chain.avg_quality:.3f}")
        print(f"  Coverage score: {coverage_score:.3f}")

        # Pretty-print
        self._print_question_results(qs)
        self._print_chain_results(chain, recommendations)

        report = EvaluationReport(
            question_scores=question_scores,
            chain_score=chain.avg_quality,
            diagnostics=chain.diagnostics,
            recommendations=recommendations,
        )

        # Save outputs if requested
        if save_dir is not None:
            self._save_chain_outputs(chain, report, save_dir)

        return report


# =============================================================================
# ChainComparator
# =============================================================================

class ChainComparator:
    """Compare multiple chains (good vs weak)."""

    @staticmethod
    def compare_chains(
        good_chains: List[ReasoningChain],
        weak_chains: List[ReasoningChain],
        save_dir: Optional[str] = None,
    ):
        print("\n" + "=" * 80)
        print("CROSS-CHAIN COMPARISON: GOOD vs WEAK")
        print("=" * 80)

        good_scores = [q.overall for c in good_chains for q in c.questions]
        weak_scores = [q.overall for c in weak_chains for q in c.questions]

        good_chain_avgs = [c.avg_quality for c in good_chains]
        weak_chain_avgs = [c.avg_quality for c in weak_chains]

        print("\nQuestion-Level Statistics:")
        print(
            f"  Good Questions: n={len(good_scores)}, "
            f"mean={np.mean(good_scores):.3f}, std={np.std(good_scores):.3f}"
        )
        print(
            f"  Weak Questions: n={len(weak_scores)}, "
            f"mean={np.mean(weak_scores):.3f}, std={np.std(weak_scores):.3f}"
        )

        print("\nChain-Level Statistics:")
        print(
            f"  Good Chains: n={len(good_chain_avgs)}, "
            f"mean={np.mean(good_chain_avgs):.3f}, std={np.std(good_chain_avgs):.3f}"
        )
        print(
            f"  Weak Chains: n={len(weak_chain_avgs)}, "
            f"mean={np.mean(weak_chain_avgs):.3f}, std={np.std(weak_chain_avgs):.3f}"
        )

        if len(good_scores) > 1 and len(weak_scores) > 1:
            t_stat, p_value = stats.ttest_ind(good_scores, weak_scores)
            print("\nStatistical Test (t-test):")
            print(f"  t-statistic: {t_stat:.4f}")
            print(f"  p-value: {p_value:.6f}")
            print(f"  Significant: {'YES' if p_value < 0.05 else 'NO'} (α=0.05)")

        dims = [
            "clarity", "specificity", "contextfulness", "necessity", "progression",
            "non_repetition", "answerability", "structural_coherence", "semantic_coherence",
            "diagnostic", "neutrality", "semantic_relevance",
        ]

        print("\nDimension-Level Breakdown:")
        print(f"  {'Dimension':<22} {'Good':<8} {'Weak':<8} {'Δ':<8}")
        print(f"  {'-' * 50}")

        for dim in dims:
            good_dim = [getattr(q, dim, 0.0) for c in good_chains for q in c.questions]
            weak_dim = [getattr(q, dim, 0.0) for c in weak_chains for q in c.questions]
            good_avg = np.mean(good_dim) if good_dim else 0.0
            weak_avg = np.mean(weak_dim) if weak_dim else 0.0
            delta = good_avg - weak_avg
            print(f"  {dim:<22} {good_avg:<8.3f} {weak_avg:<8.3f} {delta:+8.3f}")

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            ChainComparator._plot_comparison_boxplots(
                good_chains,
                weak_chains,
                os.path.join(save_dir, "comparison_boxplots.png"),
            )
            ChainComparator._plot_comparison_radar(
                good_chains,
                weak_chains,
                os.path.join(save_dir, "comparison_radar.png"),
            )
            ChainComparator._plot_dimension_correlations(
                good_chains + weak_chains,
                os.path.join(save_dir, "correlations.png"),
            )
            print(f"\n  ✓ Saved comparison visualizations to {save_dir}/")

        print("=" * 80)

    @staticmethod
    def _plot_comparison_boxplots(
        good_chains: List[ReasoningChain],
        weak_chains: List[ReasoningChain],
        save_path: str,
    ):
        dims = [
            "clarity", "specificity", "contextfulness", "necessity", "progression",
            "non_repetition", "answerability", "structural_coherence", "semantic_coherence",
            "diagnostic", "neutrality", "semantic_relevance",
        ]

        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.flatten()

        for idx, dim in enumerate(dims):
            good_vals = [getattr(q, dim, 0.0) for c in good_chains for q in c.questions]
            weak_vals = [getattr(q, dim, 0.0) for c in weak_chains for q in c.questions]

            ax = axes[idx]
            bp = ax.boxplot([good_vals, weak_vals], labels=["Good", "Weak"], patch_artist=True)
            bp["boxes"][0].set_facecolor("lightgreen")
            bp["boxes"][1].set_facecolor("lightcoral")
            ax.set_title(dim.replace("_", " ").title(), fontweight="bold")
            ax.set_ylim([0, 1])
            ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)
            ax.grid(axis="y", alpha=0.3)

        plt.suptitle("Dimension Comparison: Good vs Weak Questions", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    @staticmethod
    def _plot_comparison_radar(
        good_chains: List[ReasoningChain],
        weak_chains: List[ReasoningChain],
        save_path: str,
    ):
        dims = [
            "clarity", "specificity", "contextfulness", "necessity", "progression",
            "non_repetition", "answerability", "structural_coherence", "semantic_coherence",
            "diagnostic", "neutrality", "semantic_relevance",
        ]

        good_avgs = [np.mean([getattr(q, dim, 0.0) for c in good_chains for q in c.questions]) for dim in dims]
        weak_avgs = [np.mean([getattr(q, dim, 0.0) for c in weak_chains for q in c.questions]) for dim in dims]

        angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
        good_avgs += good_avgs[:1]
        weak_avgs += weak_avgs[:1]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection="polar"))
        ax.plot(angles, good_avgs, "o-", linewidth=2, label="Good Questions")
        ax.fill(angles, good_avgs, alpha=0.25)
        ax.plot(angles, weak_avgs, "s-", linewidth=2, label="Weak Questions")
        ax.fill(angles, weak_avgs, alpha=0.25)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([d.replace("_", " ").title() for d in dims], fontsize=10)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        ax.set_title("Average Dimension Profiles: Good vs Weak", fontweight="bold", pad=20, fontsize=14)
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    @staticmethod
    def _plot_dimension_correlations(chains: List[ReasoningChain], save_path: str):
        dims = [
            "clarity", "specificity", "contextfulness", "necessity", "progression",
            "non_repetition", "answerability", "structural_coherence", "semantic_coherence",
            "diagnostic", "neutrality", "semantic_relevance",
        ]

        data = {dim: [getattr(q, dim, 0.0) for c in chains for q in c.questions] for dim in dims}
        df = pd.DataFrame(data)
        corr_matrix = df.corr()

        plt.figure(figsize=(14, 12))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
        )
        plt.title("Dimension Correlation Matrix (All Chains)", fontsize=14, fontweight="bold", pad=20)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()


# =============================================================================
# Data loading helpers
# =============================================================================

def parse_chain_from_dict(data: Dict, chain_type: str) -> Optional[ReasoningChain]:
    """Parse chain from dictionary format (expects 'good_question_chain' or 'weak_question_chain')."""
    question_key = "good_question_chain" if chain_type == "good" else "weak_question_chain"
    if question_key not in data:
        return None

    ground_truth = data.get("ground_truth", "")
    problem = ground_truth.split("\n")[0] if ground_truth else "Problem context"

    questions = []
    context = problem
    for i, q_text in enumerate(data[question_key]):
        q = ReasoningQuestion(text=q_text, step_number=i + 1, context=context, answer=None)
        questions.append(q)
        context += f"\nQ{i+1}: {q_text}"

    return ReasoningChain(
        problem=problem,
        questions=questions,
        final_answer="",
        ground_truth=ground_truth,
        problem_id=data.get("problem_id"),
        chain_type=chain_type,
    )




def run_full_evaluation(dataset_path: str, output_dir: str = "evaluation_results_expert"):
    """Complete evaluation pipeline (Hybrid Mode C)."""
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("FULL 12-DIMENSION REASONING QUESTION EVALUATOR — Expert Reasoning Mode")
    print("=" * 80)

    with open(dataset_path, "r") as f:
        raw_data = json.load(f)

    good_chains: List[ReasoningChain] = []
    weak_chains: List[ReasoningChain] = []

    evaluator = FullEvaluator(use_semantic_model=True)

    print("\n" + "=" * 80)
    print("EVALUATING CHAINS (PAIRWISE BY PROBLEM)")
    print("=" * 80)

    for item in raw_data:
        pid = item.get("problem_id")
        good = parse_chain_from_dict(item, "good")
        weak = parse_chain_from_dict(item, "weak")

        lengths = []
        if good:
            lengths.append(len(good.questions))
        if weak:
            lengths.append(len(weak.questions))
        target_length = float(sum(lengths) / len(lengths)) if lengths else 1.0

        if good:
            print(f"\n--- Problem {pid} — GOOD chain ---")
            report_g = evaluator.evaluate_chain(good, save_dir=output_dir, target_length=target_length)
            good_chains.append(good)

        if weak:
            print(f"\n--- Problem {pid} — WEAK chain ---")
            report_w = evaluator.evaluate_chain(weak, save_dir=output_dir, target_length=target_length)
            weak_chains.append(weak)

   
    comparisons_dir = os.path.join(output_dir, "comparisons")
    ChainComparator.compare_chains(good_chains, weak_chains, save_dir=comparisons_dir)


    results_file = os.path.join(output_dir, "aggregate_results.json")
    with open(results_file, "w") as f:
        json.dump(
            {
                "good_chains": [
                    {
                        "problem_id": c.problem_id,
                        "avg_quality": c.avg_quality,
                        "coverage": c.coverage_score,
                    }
                    for c in good_chains
                ],
                "weak_chains": [
                    {
                        "problem_id": c.problem_id,
                        "avg_quality": c.avg_quality,
                        "coverage": c.coverage_score,
                    }
                    for c in weak_chains
                ],
                "summary": {
                    "good_avg": float(np.mean([c.avg_quality for c in good_chains])) if good_chains else 0.0,
                    "weak_avg": float(np.mean([c.avg_quality for c in weak_chains])) if weak_chains else 0.0,
                    "difference": (
                        float(np.mean([c.avg_quality for c in good_chains]) - np.mean([c.avg_quality for c in weak_chains]))
                        if good_chains and weak_chains
                        else 0.0
                    ),
                },
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 80)
    print("✅ EVALUATION COMPLETE (Expert Reasoning Mode)!")
    print("=" * 80)
    print(f"Results saved to: {output_dir}/")
    print("  • aggregate_results.json - Summary statistics")
    print("  • [problem_id]_[type]/ - Per-chain outputs (JSON + plots)")
    print("  • comparisons/ - Cross-chain visualizations")
    print("=" * 80)


# =============================================================================
# __main__ example
# =============================================================================

if __name__ == "__main__":
    example_data = [
        {
            "problem_id": "problem_0",
            "ground_truth": (
                "There are 600/3=200 movies in series. So he pays 200*6=$1200. "
                "He has 600-200=400 movies not in series. 400*0.4=160 movies are old. "
                "So those cost 160*5=$800. 400-160=240 movies are neither old nor in series. "
                "They cost 240*10=$2400. So the total cost is 1200+800+2400=$4400"
            ),
            "good_question_chain": [
                "From the total of 600 movies, how did you compute the number of movies that belong to series, given that a third of the movies are in series?",
                "Once you determined how many movies are in series, how did you translate the statement about series costing 60% of the normal price into a per-movie cost and then into a total cost for all series movies?",
                "After removing the series movies from the total, how many movies remain, and how did you apply the 40% figure to find how many of these remaining movies are older $5 movies?",
                "How did you determine how many movies were neither in series nor older, and what unit price did you apply to compute their total cost?",
                "When you combined the three partial costs (series, older, and remaining normal-price movies), what total did you get, and how did you verify that the total number of movies accounted for matches the original 600?",
            ],
            "weak_question_chain": [
                "Did you separate the movies into series, older, and regular categories before calculating the cost?",
                "Did you multiply the number of movies in each category by the correct price for that category?",
            ],
        }
    ]

    dataset_path = "example_dataset_expert.json"
    with open(dataset_path, "w") as f:
        json.dump(example_data, f, indent=2)
    print(f"✓ Example dataset created: {dataset_path}")


    run_full_evaluation(dataset_path, "evaluation_results_expert")

    print("\n🎉 All done!")


# ========================================================================
# HEAD-TO-HEAD CHAIN COMPARATOR WITH GRAPHS & VERDICT (DROP-IN EXTENSION)
# ========================================================================

def compare_chains_head_to_head(
    chain_a,
    chain_b,
    ground_truth,
    evaluator,
    show_plots=True,
    save_dir=None
):
    """
    Evaluates two reasoning chains against the same problem + ground truth.
    Produces: scores, dimension breakdowns, graphs, and final verdict.
    """

    print("\n===============================================")
    print("EVALUATING CHAIN A")
    print("===============================================")

    result_a = evaluator.evaluate_chain(
        chain_a,
        ground_truth=ground_truth,
        save_dir=save_dir
    )

    print("\n===============================================")
    print("EVALUATING CHAIN B")
    print("===============================================")

    result_b = evaluator.evaluate_chain(
        chain_b,
        ground_truth=ground_truth,
        save_dir=save_dir
    )

    score_a = result_a["chain_score"]
    score_b = result_b["chain_score"]

    # -------------------------------
    # Determine Winner
    # -------------------------------
    if score_a > score_b:
        winner = "CHAIN A"
    elif score_b > score_a:
        winner = "CHAIN B"
    else:
        winner = "TIE"

    print("\n===============================================")
    print("              FINAL VERDICT")
    print("===============================================")
    print(f" Chain A Score: {score_a:.3f}")
    print(f" Chain B Score: {score_b:.3f}")
    print("-----------------------------------------------")
    print(f" RESULT: {winner} IS BETTER")
    print("===============================================\n")

    # -------------------------------
    # Generate comparison plots
    # -------------------------------
    if show_plots:
        import matplotlib.pyplot as plt
        import numpy as np

        dims = list(result_a["avg_dimension_scores"].keys())
        vals_a = [result_a["avg_dimension_scores"][d] for d in dims]
        vals_b = [result_b["avg_dimension_scores"][d] for d in dims]

        x = np.arange(len(dims))
        width = 0.35

        plt.figure(figsize=(14, 6))
        plt.bar(x - width/2, vals_a, width, label="Chain A")
        plt.bar(x + width/2, vals_b, width, label="Chain B")
        plt.xticks(x, dims, rotation=45, ha='right')
        plt.ylabel("Avg Dimension Score")
        plt.title("Dimension-by-Dimension Comparison")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Radar Plot
        def radar_plot(values_a, values_b, labels):
            N = len(labels)
            angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
            values_a += values_a[:1]
            values_b += values_b[:1]
            angles += angles[:1]

            fig, ax = plt.subplots(figsize=(8,8), subplot_kw=dict(polar=True))
            ax.plot(angles, values_a, label="Chain A", linewidth=2)
            ax.fill(angles, values_a, alpha=0.25)
            ax.plot(angles, values_b, label="Chain B", linewidth=2)
            ax.fill(angles, values_b, alpha=0.25)
            ax.set_thetagrids(np.degrees(angles[:-1]), labels)
            plt.title("Radar Comparison of Chains")
            plt.legend(loc="upper right")
            plt.show()

        radar_plot(vals_a.copy(), vals_b.copy(), dims)

    # -------------------------------
    # Return JSON-like verdict object
    # -------------------------------
    return {
        "winner": winner,
        "score_a": score_a,
        "score_b": score_b,
        "result_a": result_a,
        "result_b": result_b
    }


# ------------------------------------------------------------------------------
# Convenience Wrapper: Compare Two Text-Based Chains Directly
# ------------------------------------------------------------------------------

def compare_two_text_chains(
    problem_text,
    chainA_questions,
    chainB_questions,
    ground_truth,
    evaluator,
    show_plots=True,
    save_dir=None
):
    """Create ReasoningChain objects from raw text and compare them."""
    chain_a = ReasoningChain(
        problem=problem_text,
        questions=[
            ReasoningQuestion(
                text=q,
                step_number=i+1,
                context=problem_text
            )
            for i, q in enumerate(chainA_questions)
        ],
        chain_type="A"
    )

    chain_b = ReasoningChain(
        problem=problem_text,
        questions=[
            ReasoningQuestion(
                text=q,
                step_number=i+1,
                context=problem_text
            )
            for i, q in enumerate(chainB_questions)
        ],
        chain_type="B"
    )

    return compare_chains_head_to_head(
        chain_a,
        chain_b,
        ground_truth,
        evaluator,
        show_plots=show_plots,
        save_dir=save_dir
    )

# ========================================================================
# END — HEAD-TO-HEAD CHAIN COMPARISON MODULE
# ========================================================================
