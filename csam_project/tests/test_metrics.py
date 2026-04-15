"""
Unit tests for benchmarks/metrics.py.

Covers: normalize_text, token_f1, max_token_f1, exact_match,
        max_exact_match, bleu1, aggregate_f1.
"""

import pytest

from csam_project.benchmarks.metrics import (
    normalize_text,
    token_f1,
    max_token_f1,
    exact_match,
    max_exact_match,
    bleu1,
    aggregate_f1,
)


# ── normalize_text ────────────────────────────────────────────────────────────

class TestNormalizeText:
    def test_lower_case(self) -> None:
        assert normalize_text("Hello World") == "hello world"

    def test_strips_punctuation(self) -> None:
        # Punctuation is replaced with space then whitespace is collapsed
        assert normalize_text("Hello, World!") == "hello world"

    def test_collapses_whitespace(self) -> None:
        result = normalize_text("hello   world")
        assert result == "hello world"

    def test_empty_string(self) -> None:
        assert normalize_text("") == ""

    def test_punctuation_only(self) -> None:
        assert normalize_text("!?.") == ""

    def test_numbers_preserved(self) -> None:
        assert "2023" in normalize_text("Year 2023")


# ── token_f1 ─────────────────────────────────────────────────────────────────

class TestTokenF1:
    def test_exact_match_is_one(self) -> None:
        assert token_f1("Paris", "Paris") == 1.0

    def test_no_overlap_is_zero(self) -> None:
        assert token_f1("London", "Paris") == 0.0

    def test_partial_overlap(self) -> None:
        # pred = ["the", "cat"], truth = ["the", "dog"]
        # common = {"the": 1}, precision = 1/2, recall = 1/2, F1 = 0.5
        f1 = token_f1("the cat", "the dog")
        assert abs(f1 - 0.5) < 1e-9

    def test_case_insensitive(self) -> None:
        assert token_f1("Paris", "paris") == 1.0

    def test_both_empty_returns_one(self) -> None:
        assert token_f1("", "") == 1.0

    def test_empty_prediction_returns_zero(self) -> None:
        assert token_f1("", "Paris") == 0.0

    def test_empty_truth_returns_zero(self) -> None:
        assert token_f1("Paris", "") == 0.0

    def test_repeated_tokens_counted_correctly(self) -> None:
        # pred = ["cat", "cat"], truth = ["cat"]
        # common = min(2,1)=1, precision=1/2, recall=1/1, F1=2/3
        f1 = token_f1("cat cat", "cat")
        assert abs(f1 - 2 / 3) < 1e-9

    def test_superset_prediction(self) -> None:
        # pred = ["may", "7", "2023"], truth = ["7", "may", "2023"]
        # all tokens match, F1 = 1.0
        assert token_f1("may 7 2023", "7 may 2023") == pytest.approx(1.0)

    def test_date_format_partial_match(self) -> None:
        # Simulates a typical QA partial answer
        f1 = token_f1("8 May 2023", "7 May 2023")
        # "May" and "2023" overlap → 2 out of 3 each side
        assert 0.0 < f1 < 1.0


# ── max_token_f1 ──────────────────────────────────────────────────────────────

class TestMaxTokenF1:
    def test_returns_best_reference(self) -> None:
        f1 = max_token_f1("Paris", ["London", "Paris", "Berlin"])
        assert f1 == 1.0

    def test_empty_references_returns_zero(self) -> None:
        assert max_token_f1("Paris", []) == 0.0

    def test_picks_highest(self) -> None:
        f1 = max_token_f1("the cat", ["dog", "the cat sat"])
        # "the cat" vs "dog" → 0.0
        # "the cat" vs "the cat sat" → 2 common tokens, pred=2, truth=3 → F1=4/5=0.8
        assert f1 == pytest.approx(0.8, abs=1e-9)


# ── exact_match ───────────────────────────────────────────────────────────────

class TestExactMatch:
    def test_identical(self) -> None:
        assert exact_match("Paris", "Paris") == 1.0

    def test_case_insensitive(self) -> None:
        assert exact_match("paris", "PARIS") == 1.0

    def test_different_strings(self) -> None:
        assert exact_match("Paris", "London") == 0.0

    def test_punctuation_stripped(self) -> None:
        assert exact_match("Paris!", "Paris.") == 1.0

    def test_partial_match_is_zero(self) -> None:
        assert exact_match("the cat", "the cat sat") == 0.0


# ── max_exact_match ───────────────────────────────────────────────────────────

class TestMaxExactMatch:
    def test_matches_one_of_multiple(self) -> None:
        assert max_exact_match("Paris", ["London", "Paris"]) == 1.0

    def test_no_match(self) -> None:
        assert max_exact_match("Berlin", ["London", "Paris"]) == 0.0

    def test_empty_references(self) -> None:
        assert max_exact_match("Paris", []) == 0.0


# ── bleu1 ─────────────────────────────────────────────────────────────────────

class TestBleu1:
    def test_exact_match(self) -> None:
        assert bleu1("the cat sat", "the cat sat") == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        assert bleu1("dog runs fast", "the cat sat") == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        # pred = ["the", "cat", "ran"], truth = ["the", "cat", "sat"]
        # clipped matches: "the", "cat" → 2 → 2/3
        assert bleu1("the cat ran", "the cat sat") == pytest.approx(2 / 3, abs=1e-9)

    def test_empty_prediction(self) -> None:
        assert bleu1("", "some text") == 0.0

    def test_empty_truth(self) -> None:
        assert bleu1("some text", "") == 0.0

    def test_clipping_prevents_inflation(self) -> None:
        # pred = ["the", "the", "the"], truth = ["the", "cat"]
        # Without clipping: 3/3=1.0; with clipping: min(3,1)=1 → 1/3
        assert bleu1("the the the", "the cat") == pytest.approx(1 / 3, abs=1e-9)


# ── aggregate_f1 ──────────────────────────────────────────────────────────────

class TestAggregateF1:
    def test_empty_list_returns_zeros(self) -> None:
        result = aggregate_f1([])
        assert result["mean"] == 0.0
        assert result["n"] == 0

    def test_single_score(self) -> None:
        result = aggregate_f1([0.8])
        assert result["mean"] == pytest.approx(0.8)
        assert result["min"] == pytest.approx(0.8)
        assert result["max"] == pytest.approx(0.8)
        assert result["n"] == 1

    def test_mean_is_correct(self) -> None:
        scores = [0.0, 0.5, 1.0]
        result = aggregate_f1(scores)
        assert result["mean"] == pytest.approx(0.5)

    def test_includes_zeros(self) -> None:
        # Ensures zero-scores are NOT excluded from mean (BUG-01 regression)
        scores = [1.0, 0.0, 0.0]
        result = aggregate_f1(scores)
        assert result["mean"] == pytest.approx(1 / 3, abs=1e-9)

    def test_median(self) -> None:
        scores = [0.1, 0.5, 0.9]
        result = aggregate_f1(scores)
        assert result["median"] == pytest.approx(0.5)

    def test_all_keys_present(self) -> None:
        result = aggregate_f1([0.5])
        assert set(result.keys()) == {"mean", "median", "p25", "p75", "min", "max", "n"}

    @pytest.mark.parametrize("scores,expected_mean", [
        ([0.0] * 10, 0.0),
        ([1.0] * 10, 1.0),
        ([0.4, 0.6], 0.5),
    ])
    def test_parametrized_means(
        self, scores: list[float], expected_mean: float
    ) -> None:
        assert aggregate_f1(scores)["mean"] == pytest.approx(expected_mean)
