"""Calibrate: set each guard's threshold for a target false-positive rate (split conformal).

A guard acts when its score reaches its threshold. Given the scores of n normal examples
(normal prompts for the input guard, the model's honest replies for the leak detector), the
threshold is set just above the ceil((n+1)(1-fpr))-th smallest score. If future normal
traffic is exchangeable with the calibration examples, the chance that the guard acts on a
normal example is then at most ``fpr`` (split conformal prediction; see Angelopoulos &
Bates, "A Gentle Introduction to Conformal Prediction", 2021).

Part of the data is held back and never used to set the threshold; the rate on it is
reported with a one-sided 95% Clopper-Pearson upper bound. The guarantee only covers
traffic like the calibration set: calibrate on prompts and replies from your own users.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class HeldOut(BaseModel):
    """How often the calibrated guard acted on normal examples it was not calibrated on."""

    n: int
    acted: int
    rate: float
    upper95: float


class GuardCalibration(BaseModel):
    threshold: float
    fpr: float
    n_calibration: int
    held_out: HeldOut | None = None
    source: str = ""
    created: str = ""


class Calibration(BaseModel):
    """Calibrated thresholds for one deployment, written by ``unmimic calibrate``.

    The input guard has one threshold for all traffic; the leak detector has one per model,
    since each model restates its reasoning in its answers differently.
    """

    input_guard: GuardCalibration | None = None
    leak_detector: dict[str, GuardCalibration] = {}

    @classmethod
    def load(cls, path: Path) -> Calibration:
        """The saved calibration, or an empty one (all defaults) if there is none."""
        if not path.is_file():
            return cls()
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")

    def options(self, guard: str) -> dict[str, Any]:
        """Constructor arguments for a registered guard, by its registry name."""
        if guard == "extraction_guard" and self.input_guard:
            return {"memory_threshold": self.input_guard.threshold}
        if guard == "leak_detector" and self.leak_detector:
            return {"model_thresholds": {m: c.threshold for m, c in self.leak_detector.items()}}
        return {}


def conformal_threshold(scores: Sequence[float], fpr: float) -> float:
    """Smallest threshold with at most ``fpr`` false positives on exchangeable normal data.

    The guard acts on ``score >= threshold``, i.e. on scores above the
    ceil((n+1)(1-fpr))-th smallest calibration score.
    """
    if not 0 < fpr < 1:
        raise ValueError(f"fpr must be between 0 and 1, got {fpr}")
    n = len(scores)
    # The small offset keeps float error in (n+1)(1-fpr) from rounding k up by one.
    k = math.ceil((n + 1) * (1 - fpr) - 1e-9)
    if k > n:
        need = math.ceil((1 - fpr) / fpr - 1e-9)
        raise ValueError(f"fpr={fpr} needs at least {need} normal examples to calibrate, got {n}")
    return math.nextafter(sorted(scores)[k - 1], math.inf)


def upper_bound(errors: int, n: int, confidence: float = 0.95) -> float:
    """One-sided Clopper-Pearson upper bound on a rate."""
    from scipy.stats import beta

    return 1.0 if errors >= n else float(beta.ppf(confidence, errors + 1, n - errors))


def split(items: Sequence[T], held_out: float, seed: int) -> tuple[list[T], list[T]]:
    """Shuffle, then cut into (calibration, held out)."""
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    cut = len(shuffled) - round(len(shuffled) * held_out)
    return shuffled[:cut], shuffled[cut:]


def calibrate(
    examples: Sequence[T],
    score: Callable[[T], float],
    fpr: float,
    held_out: float = 0.5,
    seed: int = 0,
    source: str = "",
    on_score: Callable[[int, int], None] | None = None,
) -> GuardCalibration:
    """Score normal examples, set the conformal threshold, check it on the held-out part."""
    calibration_set, test_set = split(examples, held_out, seed)
    scores = []
    total = len(examples)
    for example in calibration_set + test_set:
        scores.append(score(example))
        if on_score:
            on_score(len(scores), total)
    calibration_scores, test_scores = scores[: len(calibration_set)], scores[len(calibration_set) :]
    threshold = conformal_threshold(calibration_scores, fpr)
    check = None
    if test_scores:
        acted = sum(s >= threshold for s in test_scores)
        n = len(test_scores)
        check = HeldOut(n=n, acted=acted, rate=acted / n, upper95=upper_bound(acted, n))
    return GuardCalibration(
        threshold=threshold,
        fpr=fpr,
        n_calibration=len(calibration_scores),
        held_out=check,
        source=source,
        created=datetime.now(UTC).isoformat(timespec="seconds"),
    )


def load_replies(path: Path) -> list[tuple[str, str]]:
    """(reasoning, answer) of every complete reply in a JSONL file.

    Each line needs ``reasoning`` and ``answer`` (or ``visible_answer``, as in the teacher
    traces). Replies cut off by the length limit or missing either part are skipped.
    """
    import json

    replies = []
    # split("\n"), not splitlines(): model output can hold U+2028 and other characters that
    # splitlines() treats as line breaks inside a JSON string.
    for line in path.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        row = json.loads(line)
        reasoning = row.get("reasoning") or ""
        answer = row.get("answer") or row.get("visible_answer") or ""
        if row.get("finish_reason") == "length" or not reasoning.strip() or not answer.strip():
            continue
        replies.append((reasoning, answer))
    return replies
