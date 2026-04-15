"""
Canonical evaluation metrics for CSAM benchmarks.

Shared by all benchmark scripts to guarantee consistent F1/EM computation
across datasets (HotPotQA, MuSiQue, LoCoMo).

All functions are deterministic and dependency-free (stdlib + stdlib re/Counter).
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable


# ── Text normalisation ────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """
    Lower-case and strip punctuation for token-level comparison.

    Matches the normalisation used in the original SQuAD eval script:
    - lower-case
    - remove punctuation / special characters
    - collapse whitespace

    Args:
        text: Raw prediction or reference string.

    Returns:
        Normalised string, space-tokenisable.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)   # replace punctuation with space
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Token-level F1 ────────────────────────────────────────────────────────────

def token_f1(prediction: str, ground_truth: str) -> float:
    """
    Token-level F1 score (bag-of-words).

    Identical to the SQuAD / HotPotQA evaluation formula:
        precision = |pred ∩ truth| / |pred|
        recall    = |pred ∩ truth| / |truth|
        F1        = 2 * P * R / (P + R)

    Empty prediction or truth that matches empty returns 1.0; any other
    empty pair returns 0.0.

    Args:
        prediction: Model's answer (raw text).
        ground_truth: Reference answer (raw text).

    Returns:
        F1 score in [0.0, 1.0].
    """
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(ground_truth).split()

    # Both empty → exact match
    if not pred_tokens and not truth_tokens:
        return 1.0
    # One empty → no overlap
    if not pred_tokens or not truth_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def max_token_f1(prediction: str, ground_truths: Iterable[str]) -> float:
    """
    Maximum F1 across multiple reference answers.

    Args:
        prediction: Model's answer.
        ground_truths: Iterable of acceptable reference strings.

    Returns:
        Maximum F1 score over all references.
    """
    return max((token_f1(prediction, gt) for gt in ground_truths), default=0.0)


# ── Exact Match ───────────────────────────────────────────────────────────────

def exact_match(prediction: str, ground_truth: str) -> float:
    """
    Exact match after normalisation (returns 1.0 or 0.0).

    Args:
        prediction: Model's answer.
        ground_truth: Reference answer.

    Returns:
        1.0 if normalised strings are identical, else 0.0.
    """
    return 1.0 if normalize_text(prediction) == normalize_text(ground_truth) else 0.0


def max_exact_match(prediction: str, ground_truths: Iterable[str]) -> float:
    """
    Maximum exact match across multiple reference answers.

    Args:
        prediction: Model's answer.
        ground_truths: Iterable of acceptable reference strings.

    Returns:
        1.0 if prediction exactly matches any reference, else 0.0.
    """
    return max((exact_match(prediction, gt) for gt in ground_truths), default=0.0)


# ── BLEU-1 ────────────────────────────────────────────────────────────────────

def bleu1(prediction: str, ground_truth: str) -> float:
    """
    Unigram precision (clipped) — BLEU-1.

    Args:
        prediction: Model's answer.
        ground_truth: Reference answer.

    Returns:
        BLEU-1 score in [0.0, 1.0].
    """
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(ground_truth).split()

    if not pred_tokens or not truth_tokens:
        return 0.0

    truth_counter = Counter(truth_tokens)
    clipped = 0
    for tok in pred_tokens:
        if truth_counter[tok] > 0:
            clipped += 1
            truth_counter[tok] -= 1

    return clipped / len(pred_tokens)


# ── Aggregate helpers ─────────────────────────────────────────────────────────

def aggregate_f1(scores: list[float]) -> dict[str, float]:
    """
    Compute mean F1 and per-percentile stats from a list of question-level scores.

    Args:
        scores: List of per-question F1 values.

    Returns:
        Dict with keys: mean, median, p25, p75, min, max, n.
    """
    if not scores:
        return {"mean": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0,
                "min": 0.0, "max": 0.0, "n": 0}

    sorted_scores = sorted(scores)
    n = len(sorted_scores)

    def _percentile(p: float) -> float:
        idx = p * (n - 1)
        lo, hi = int(idx), min(int(idx) + 1, n - 1)
        frac = idx - lo
        return sorted_scores[lo] * (1 - frac) + sorted_scores[hi] * frac

    return {
        "mean": sum(scores) / n,
        "median": _percentile(0.5),
        "p25": _percentile(0.25),
        "p75": _percentile(0.75),
        "min": sorted_scores[0],
        "max": sorted_scores[-1],
        "n": n,
    }
