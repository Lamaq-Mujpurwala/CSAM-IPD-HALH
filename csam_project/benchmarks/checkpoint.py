"""
Checkpoint utility for session-resilient benchmark runs.

Saves per-question results to a JSON file so interrupted runs can resume
from where they left off. Designed for Kaggle/Colab environments with
session timeout constraints (typically 8–12 hours).

Usage::

    ckpt = BenchmarkCheckpoint.for_run(
        benchmark="musique", provider="groq",
        model="llama-3.1-8b-instant", seed=42, questions=100,
    )
    for entry in dataset:
        if ckpt.is_done(entry["id"]):
            continue
        result = run_qa(entry)
        ckpt.add_result(entry["id"], result)

    all_results = ckpt.get_results()
    ckpt.delete()  # clean up after successful completion
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkCheckpoint:
    """
    Atomic, resumable checkpoint for per-question benchmark results.

    Each result is flushed to disk immediately after being computed, so a
    session timeout (Colab / Kaggle) loses at most one in-progress question.
    The checkpoint file path is deterministic from the run config, so
    re-running the same command automatically resumes.
    """

    path: Path
    _data: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._load()

    # ── Factory ──────────────────────────────────────────────────────────────────

    @classmethod
    def for_run(
        cls,
        benchmark: str,
        provider: str,
        model: str,
        seed: int,
        questions: int,
        checkpoint_dir: str | Path | None = None,
    ) -> "BenchmarkCheckpoint":
        """Create a checkpoint with a deterministic path for this run config."""
        safe_model = model.replace("/", "_").replace(":", "_")
        filename = f"ckpt_{benchmark}_{provider}_{safe_model}_s{seed}_n{questions}.json"
        if checkpoint_dir is not None:
            path = Path(checkpoint_dir) / filename
        else:
            path = Path(__file__).parent / filename
        return cls(path=path)

    # ── Internal ──────────────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self.path.exists():
            self._data = {}
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                self._data = json.load(f)
            n = len(self._data.get("results", {}))
            logger.info("Checkpoint loaded: %s (%d completed)", self.path.name, n)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Corrupt checkpoint, starting fresh: %s", exc)
            self._data = {}

    def _save_atomic(self) -> None:
        """Write to temp file then rename — prevents corruption on crash."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)
            os.replace(tmp_path, self.path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # ── Public API ───────────────────────────────────────────────────────────────

    def is_done(self, question_id: str) -> bool:
        """Return True if this question already has a saved result."""
        return question_id in self._data.get("results", {})

    def add_result(self, question_id: str, result: dict[str, Any]) -> None:
        """Persist a per-question result immediately to disk."""
        if "results" not in self._data:
            self._data["results"] = {}
        self._data["results"][question_id] = result
        self._save_atomic()

    def get_result(self, question_id: str) -> dict[str, Any] | None:
        """Return a saved result by ID, or None if not found."""
        return self._data.get("results", {}).get(question_id)

    def get_results(self) -> list[dict[str, Any]]:
        """Return all saved results in insertion order."""
        return list(self._data.get("results", {}).values())

    def get_completed_ids(self) -> set[str]:
        """Return the set of question IDs that are already complete."""
        return set(self._data.get("results", {}).keys())

    def num_completed(self) -> int:
        """Number of questions completed so far."""
        return len(self._data.get("results", {}))

    def delete(self) -> None:
        """Remove the checkpoint file after a successful full run."""
        try:
            self.path.unlink(missing_ok=True)
            logger.info("Checkpoint deleted: %s", self.path.name)
        except OSError as exc:
            logger.warning("Could not delete checkpoint: %s", exc)
