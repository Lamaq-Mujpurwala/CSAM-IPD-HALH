"""
compare_csam_vs_baseline.py — Protocol-verified CSAM vs. Flat-RAG comparison.

Validates that two benchmark result JSON files (CSAM and Flat-RAG baseline) used
identical experimental settings before computing the F1 delta and bootstrap CI.

Usage:
    python benchmarks/compare_csam_vs_baseline.py \\
        --csam    benchmarks/results_csam_groq_llama-3.1-8b-instant_s42.json \\
        --baseline benchmarks/results_baseline_groq_llama-3.1-8b-instant_s42.json

    # Compare all matched pairs in a directory:
    python benchmarks/compare_csam_vs_baseline.py --dir benchmarks/

Output:
    results_comparison_<model>_s<seed>.json   — per-conversation deltas + CI
    (also prints a summary table to stdout)

Protocol parity check:
    MANDATORY fields (error if mismatch): model_id, provider, seed,
        retrieval_k, context_top_k, dataset, questions_per_conv
    ADVISORY fields (warning if mismatch): num_conversations
    EXCLUDED from check: architecture, skip_consolidation, timestamp
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Protocol fields that MUST be identical for a valid comparison
MANDATORY_PARITY_FIELDS = [
    "model_id",
    "provider",
    "seed",
    "retrieval_k",
    "context_top_k",
    "dataset",
    "questions_per_conv",
]

# Protocol fields that trigger a warning when they differ
ADVISORY_PARITY_FIELDS = [
    "num_conversations",
]


# ── Bootstrap CI (same implementation as benchmarks for reproducibility) ──────

def bootstrap_ci(
    scores: list[float],
    n_samples: int = 1000,
    ci: float = 0.95,
    rng_seed: int = 42,
) -> tuple[float, float]:
    if not scores:
        return 0.0, 0.0
    rng = np.random.default_rng(rng_seed)
    boot_means = [
        float(np.mean(rng.choice(scores, size=len(scores), replace=True)))
        for _ in range(n_samples)
    ]
    lo = (1 - ci) / 2
    return float(np.quantile(boot_means, lo)), float(np.quantile(boot_means, 1 - lo))


# ── Protocol parity validation ────────────────────────────────────────────────

class ProtocolMismatchError(ValueError):
    """Raised when mandatory protocol fields differ between CSAM and baseline."""


def validate_protocol_parity(
    csam_protocol: dict,
    baseline_protocol: dict,
    csam_path: str,
    baseline_path: str,
) -> list[str]:
    """
    Compare protocol fingerprints between CSAM and baseline results.

    Returns:
        List of advisory warning strings (may be empty).

    Raises:
        ProtocolMismatchError: if any MANDATORY_PARITY_FIELDS differ.
    """
    errors: list[str] = []
    warnings: list[str] = []

    for field in MANDATORY_PARITY_FIELDS:
        cv = csam_protocol.get(field)
        bv = baseline_protocol.get(field)
        if cv != bv:
            errors.append(
                f"  MANDATORY mismatch — '{field}': "
                f"CSAM={cv!r}  BASELINE={bv!r}"
            )

    for field in ADVISORY_PARITY_FIELDS:
        cv = csam_protocol.get(field)
        bv = baseline_protocol.get(field)
        if cv != bv:
            warnings.append(
                f"  ADVISORY mismatch — '{field}': "
                f"CSAM={cv!r}  BASELINE={bv!r}"
            )

    if errors:
        msg = (
            "\nProtocol parity check FAILED — the two result files used different "
            "experimental settings and cannot be compared directly.\n"
            + "\n".join(errors)
            + f"\n\nFiles:\n  CSAM:     {csam_path}\n  Baseline: {baseline_path}"
        )
        raise ProtocolMismatchError(msg)

    return warnings


# ── Per-conversation alignment ────────────────────────────────────────────────

def align_conversations(
    csam_convs: list[dict],
    baseline_convs: list[dict],
) -> list[tuple[dict, dict]]:
    """
    Pair CSAM and baseline conversations by conv_index.

    Returns list of (csam_conv, baseline_conv) pairs for conversations present
    in both runs.  Logs a warning for any unpaired conversations.
    """
    csam_by_idx = {c["conv_index"]: c for c in csam_convs}
    base_by_idx = {c["conv_index"]: c for c in baseline_convs}

    shared = sorted(set(csam_by_idx) & set(base_by_idx))
    csam_only = set(csam_by_idx) - set(base_by_idx)
    base_only = set(base_by_idx) - set(csam_by_idx)

    if csam_only:
        logger.warning("Conversations in CSAM but not baseline: %s", sorted(csam_only))
    if base_only:
        logger.warning("Conversations in baseline but not CSAM: %s", sorted(base_only))

    return [(csam_by_idx[i], base_by_idx[i]) for i in shared]


# ── Core comparison logic ─────────────────────────────────────────────────────

def compare(
    csam_path: str,
    baseline_path: str,
    output_dir: str | None = None,
    n_bootstrap: int = 1000,
) -> dict:
    """
    Load, validate, and compare CSAM vs. baseline result files.

    Returns:
        Comparison result dict (also written to disk).

    Raises:
        FileNotFoundError: if either input file is missing.
        ProtocolMismatchError: if mandatory protocol fields differ.
        KeyError: if a result file is missing required top-level keys.
    """
    for path in (csam_path, baseline_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Result file not found: {path}")

    with open(csam_path, encoding="utf-8") as f:
        csam_data: dict = json.load(f)
    with open(baseline_path, encoding="utf-8") as f:
        base_data: dict = json.load(f)

    # ── Protocol validation ──────────────────────────────────────────────────
    csam_proto = csam_data.get("protocol", {})
    base_proto = base_data.get("protocol", {})

    if not csam_proto:
        logger.warning(
            "CSAM file has no 'protocol' block — parity check skipped: %s",
            csam_path,
        )
    if not base_proto:
        logger.warning(
            "Baseline file has no 'protocol' block — parity check skipped: %s",
            baseline_path,
        )

    advisory_warnings: list[str] = []
    if csam_proto and base_proto:
        advisory_warnings = validate_protocol_parity(
            csam_proto, base_proto, csam_path, baseline_path
        )
        for w in advisory_warnings:
            logger.warning(w)

    # ── Per-conversation delta ───────────────────────────────────────────────
    csam_convs: list[dict] = csam_data.get("per_conversation", [])
    base_convs: list[dict] = base_data.get("per_conversation", [])

    # Fall back to single-conv format (old benchmark outputs without per_conversation)
    if not csam_convs and "f1_scores" in csam_data:
        csam_convs = [{"conv_index": 0, "avg_f1": csam_data["avg_f1"],
                       "f1_scores": csam_data["f1_scores"]}]
    if not base_convs and "f1_scores" in base_data:
        base_convs = [{"conv_index": 0, "avg_f1": base_data["avg_f1"],
                       "f1_scores": base_data["f1_scores"]}]

    pairs = align_conversations(csam_convs, base_convs)
    if not pairs:
        raise ValueError(
            "No paired conversations found between CSAM and baseline results."
        )

    per_conv_comparisons: list[dict] = []
    all_csam_q_f1: list[float] = []
    all_base_q_f1: list[float] = []
    conv_deltas: list[float] = []

    for csam_c, base_c in pairs:
        conv_idx = csam_c["conv_index"]
        csam_f1 = float(csam_c.get("avg_f1", 0.0))
        base_f1 = float(base_c.get("avg_f1", 0.0))
        delta = csam_f1 - base_f1
        conv_deltas.append(delta)

        csam_qs: list[float] = csam_c.get("f1_scores", [])
        base_qs: list[float] = base_c.get("f1_scores", [])
        all_csam_q_f1.extend(csam_qs)
        all_base_q_f1.extend(base_qs)

        per_conv_comparisons.append({
            "conv_index": conv_idx,
            "csam_avg_f1": csam_f1,
            "baseline_avg_f1": base_f1,
            "delta_f1": delta,
            "csam_num_questions": len(csam_qs),
            "baseline_num_questions": len(base_qs),
        })

    # ── Aggregate metrics ────────────────────────────────────────────────────
    mean_conv_delta = float(np.mean(conv_deltas)) if conv_deltas else 0.0
    ci_lo, ci_hi = bootstrap_ci(conv_deltas, n_samples=n_bootstrap)

    csam_macro_f1 = float(np.mean([p["csam_avg_f1"] for p in per_conv_comparisons]))
    base_macro_f1 = float(np.mean([p["baseline_avg_f1"] for p in per_conv_comparisons]))
    csam_micro_f1 = float(np.mean(all_csam_q_f1)) if all_csam_q_f1 else 0.0
    base_micro_f1 = float(np.mean(all_base_q_f1)) if all_base_q_f1 else 0.0

    seed = csam_proto.get("seed", csam_data.get("seed", 42))
    model_id = csam_proto.get("model_id", csam_data.get("model", "unknown"))
    safe_model = str(model_id).replace("/", "_").replace(":", "_")

    # ── Print summary ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"CSAM vs. Baseline Comparison — {model_id}")
    print(f"{'='*70}")
    print(f"  CSAM:           {os.path.basename(csam_path)}")
    print(f"  Baseline:       {os.path.basename(baseline_path)}")
    print(f"  Conversations:  {len(pairs)}")
    print(f"  Questions:      CSAM={len(all_csam_q_f1)}  Baseline={len(all_base_q_f1)}")
    print(f"\n{'─'*70}")
    print(f"  {'Metric':<25} {'CSAM':>10} {'Baseline':>10} {'Δ (CSAM-Base)':>14}")
    print(f"{'─'*70}")
    print(f"  {'Macro F1':<25} {csam_macro_f1:>10.4f} {base_macro_f1:>10.4f} {csam_macro_f1-base_macro_f1:>+14.4f}")
    print(f"  {'Micro F1':<25} {csam_micro_f1:>10.4f} {base_micro_f1:>10.4f} {csam_micro_f1-base_micro_f1:>+14.4f}")
    print(f"  {'Mean conv Δ':<25} {'':>10} {'':>10} {mean_conv_delta:>+14.4f}")
    print(f"  {'95% CI on Δ':<25} {'':>10} {'':>10} "
          f"  [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"{'─'*70}")

    if advisory_warnings:
        print("\n  Advisory warnings:")
        for w in advisory_warnings:
            print(w)

    significance = "POSITIVE" if ci_lo > 0 else ("NEGATIVE" if ci_hi < 0 else "NOT_SIGNIFICANT")
    print(f"\n  Statistical significance: {significance} at 95% CI")
    print(f"{'='*70}\n")

    # ── Per-conversation table ───────────────────────────────────────────────
    print(f"  {'Conv':<6} {'CSAM F1':>9} {'Base F1':>9} {'Δ':>9}")
    print(f"  {'-'*36}")
    for p in per_conv_comparisons:
        sign = "+" if p["delta_f1"] >= 0 else ""
        print(
            f"  {p['conv_index']:<6} {p['csam_avg_f1']:>9.4f} "
            f"{p['baseline_avg_f1']:>9.4f} {sign}{p['delta_f1']:>8.4f}"
        )
    print()

    # ── Build output dict ────────────────────────────────────────────────────
    comparison = {
        "comparison_timestamp": datetime.now().isoformat(),
        "csam_file": csam_path,
        "baseline_file": baseline_path,
        "model_id": model_id,
        "provider": csam_proto.get("provider", csam_data.get("provider", "unknown")),
        "seed": seed,
        "num_conversations": len(pairs),
        "csam_macro_f1": csam_macro_f1,
        "baseline_macro_f1": base_macro_f1,
        "macro_f1_delta": csam_macro_f1 - base_macro_f1,
        "csam_micro_f1": csam_micro_f1,
        "baseline_micro_f1": base_micro_f1,
        "micro_f1_delta": csam_micro_f1 - base_micro_f1,
        "mean_conv_delta_f1": mean_conv_delta,
        "delta_ci_95": [ci_lo, ci_hi],
        "statistical_significance": significance,
        "advisory_warnings": advisory_warnings,
        "per_conversation": per_conv_comparisons,
        "csam_protocol": csam_proto,
        "baseline_protocol": base_proto,
    }

    # ── Save to disk ─────────────────────────────────────────────────────────
    if output_dir is None:
        output_dir = os.path.dirname(csam_path)
    out_path = os.path.join(
        output_dir,
        f"results_comparison_csam_vs_baseline_{safe_model}_s{seed}.json",
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    print(f"Comparison saved: {out_path}\n")

    return comparison


# ── Directory auto-pairing ────────────────────────────────────────────────────

def _extract_pair_key(filename: str) -> tuple[str, str] | None:
    """
    Extract (model_safe, seed) from a result filename.

    Handles:
      results_csam_<provider>_<model>_s<seed>.json
      results_baseline_<provider>_<model>_s<seed>.json
    """
    m = re.match(
        r"results_(?:csam|baseline)_\w+_(.+)_s(\d+)\.json$",
        filename,
    )
    if m:
        return m.group(1), m.group(2)
    return None


def run_directory(
    directory: str,
    output_dir: str | None = None,
    n_bootstrap: int = 1000,
) -> list[dict]:
    """Find and compare all matched CSAM/baseline pairs in a directory."""
    files = os.listdir(directory)
    csam_files = {
        f: _extract_pair_key(f) for f in files if f.startswith("results_csam_")
    }
    base_files = {
        f: _extract_pair_key(f) for f in files if f.startswith("results_baseline_")
    }

    # Build reverse lookup: (model_safe, seed) → filename
    csam_by_key: dict[tuple, str] = {
        v: f for f, v in csam_files.items() if v is not None
    }
    base_by_key: dict[tuple, str] = {
        v: f for f, v in base_files.items() if v is not None
    }

    shared_keys = set(csam_by_key) & set(base_by_key)
    if not shared_keys:
        logger.warning("No matched CSAM/baseline pairs found in %s", directory)
        return []

    logger.info("Found %d matched pair(s) to compare", len(shared_keys))
    results = []
    for key in sorted(shared_keys):
        csam_path = os.path.join(directory, csam_by_key[key])
        base_path = os.path.join(directory, base_by_key[key])
        try:
            result = compare(csam_path, base_path, output_dir=output_dir,
                             n_bootstrap=n_bootstrap)
            results.append(result)
        except ProtocolMismatchError as e:
            logger.error("Protocol mismatch for pair %s:\n%s", key, e)
        except Exception:
            logger.exception("Comparison failed for pair %s", key)

    return results


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare CSAM vs. Flat-RAG baseline results with protocol parity check"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--csam", type=str, help="Path to CSAM result JSON")
    group.add_argument("--dir", type=str,
                       help="Directory to auto-pair all CSAM/baseline result files")

    parser.add_argument("--baseline", type=str,
                        help="Path to baseline result JSON (required with --csam)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to write comparison JSON (default: same as --csam)")
    parser.add_argument("--n-bootstrap", type=int, default=1000,
                        help="Bootstrap resampling iterations (default: 1000)")

    args = parser.parse_args()

    if args.csam and not args.baseline:
        parser.error("--csam requires --baseline")

    if args.dir:
        results = run_directory(
            args.dir,
            output_dir=args.output_dir,
            n_bootstrap=args.n_bootstrap,
        )
        return 0 if results else 1

    try:
        compare(
            csam_path=args.csam,
            baseline_path=args.baseline,
            output_dir=args.output_dir,
            n_bootstrap=args.n_bootstrap,
        )
    except FileNotFoundError as e:
        logger.error("%s", e)
        return 1
    except ProtocolMismatchError as e:
        logger.error("%s", e)
        return 2
    except KeyError as e:
        logger.error("Result file is missing required key: %s", e)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
