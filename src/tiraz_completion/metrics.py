"""Calibration and set-valued predictions with explicitly separate data roles."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tiraz_completion.data import N_CLASSES


def validate_labels(y: NDArray[np.int64], count: int) -> None:
    if (
        count <= 0
        or y.ndim != 1
        or len(y) != count
        or not np.issubdtype(y.dtype, np.integer)
        or np.any(y < 0)
        or np.any(y >= N_CLASSES)
    ):
        raise ValueError("labels must be a nonempty matching vector of valid category integers")


def validate_probabilities(p: NDArray[np.float64]) -> None:
    if (
        p.ndim != 2
        or p.shape[0] == 0
        or p.shape[1] != N_CLASSES
        or not np.isfinite(p).all()
        or np.any(p < 0)
        or np.any(p > 1)
        or not np.allclose(p.sum(axis=1), 1, rtol=0, atol=1e-8)
    ):
        raise ValueError("probability matrix must contain finite category distributions")


def probabilities(logits: NDArray[np.float64], temperature: float = 1.0) -> NDArray[np.float64]:
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if (
        logits.ndim != 2
        or logits.shape[0] == 0
        or logits.shape[1] != N_CLASSES
        or not np.isfinite(logits).all()
    ):
        raise ValueError("logits must be a nonempty finite category matrix")
    scaled = logits / temperature
    weights = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    return weights / weights.sum(axis=1, keepdims=True)


def fit_temperature(logits: NDArray[np.float64], y: NDArray[np.int64]) -> float:
    validate_labels(y, len(logits))
    candidates = np.geomspace(0.25, 4, 121)
    losses = [
        -np.log(probabilities(logits, float(t))[np.arange(len(y)), y].clip(1e-12)).mean()
        for t in candidates
    ]
    return float(candidates[int(np.argmin(losses))])


def conformal_threshold(p: NDArray[np.float64], y: NDArray[np.int64], alpha: float) -> float:
    validate_probabilities(p)
    validate_labels(y, len(p))
    if not 0 < alpha < 1 or len(y) == 0:
        raise ValueError("nonempty calibration data and alpha in (0,1) required")
    if p.shape != (len(y), N_CLASSES) or not np.isfinite(p).all():
        raise ValueError("invalid calibration probability matrix")
    if np.any(p < 0) or not np.allclose(p.sum(axis=1), 1):
        raise ValueError("calibration rows must be distributions")
    rank = math.ceil((len(y) + 1) * (1 - alpha))
    if rank > len(y):
        return 1.0
    scores = 1 - p[np.arange(len(y)), y]
    return float(np.sort(scores)[rank - 1])


def prediction_sets(p: NDArray[np.float64], x: NDArray[np.float32], q: float) -> NDArray[np.bool_]:
    validate_probabilities(p)
    if not 0 <= q <= 1 or p.shape != x.shape:
        raise ValueError("invalid prediction-set inputs")
    if not np.isfinite(x).all() or np.any((x != 0) & (x != 1)):
        raise ValueError("context must be a binary category matrix")
    return np.asarray(((1 - p) <= q) & ~x.astype(bool), dtype=np.bool_)


def wilson(successes: int, count: int) -> list[float] | None:
    if count == 0:
        return None
    z = 1.959963984540054
    rate = successes / count
    denom = 1 + z * z / count
    center = (rate + z * z / (2 * count)) / denom
    half = z * math.sqrt(rate * (1 - rate) / count + z * z / (4 * count * count)) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def summarize(
    p: NDArray[np.float64], y: NDArray[np.int64], sets: NDArray[np.bool_]
) -> dict[str, Any]:
    validate_probabilities(p)
    validate_labels(y, len(p))
    if sets.shape != p.shape or sets.dtype != np.bool_:
        raise ValueError("prediction sets must be a matching boolean matrix")
    predicted = p.argmax(axis=1)
    correct = predicted == y
    one_hot = np.eye(N_CLASSES)[y]
    f1 = []
    for category in range(N_CLASSES):
        tp = int(((predicted == category) & (y == category)).sum())
        fp = int(((predicted == category) & (y != category)).sum())
        fn = int(((predicted != category) & (y == category)).sum())
        f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    confidence = p.max(axis=1)
    bins = np.minimum((confidence * 10).astype(int), 9)
    ece = 0.0
    for index in range(10):
        mask = bins == index
        if mask.any():
            ece += float(mask.mean() * abs(correct[mask].mean() - confidence[mask].mean()))
    covered = sets[np.arange(len(y)), y]
    sizes = sets.sum(axis=1)
    singleton = sizes == 1
    singleton_correct = (sets.argmax(axis=1) == y) & singleton
    return {
        "n": len(y),
        "top1_accuracy": float(correct.mean()),
        "top1_accuracy_wilson95": wilson(int(correct.sum()), len(y)),
        "macro_f1_27_classes": float(np.mean(f1)),
        "negative_log_likelihood": float(-np.log(p[np.arange(len(y)), y].clip(1e-12)).mean()),
        "brier": float(np.square(p - one_hot).sum(axis=1).mean()),
        "ece_10_bins": ece,
        "prediction_set_coverage": float(covered.mean()),
        "set_coverage_wilson95": wilson(int(covered.sum()), len(y)),
        "mean_set_size": float(sizes.mean()),
        "empty_set_rate": float((sizes == 0).mean()),
        "singleton_rate": float(singleton.mean()),
        "singleton_accuracy": (
            float(singleton_correct.sum() / singleton.sum()) if singleton.any() else None
        ),
    }


def paired_accuracy_interval(
    first: NDArray[np.float64],
    second: NDArray[np.float64],
    y: NDArray[np.int64],
) -> dict[str, Any]:
    validate_probabilities(first)
    validate_probabilities(second)
    validate_labels(y, len(first))
    if first.shape != second.shape:
        raise ValueError("paired matrices must have identical shape")
    difference = (first.argmax(axis=1) == y).astype(float) - (second.argmax(axis=1) == y)
    rng = np.random.default_rng(20260830)
    draws = np.array([difference[rng.integers(0, len(y), len(y))].mean() for _ in range(2000)])
    return {
        "difference": float(difference.mean()),
        "paired_unit_bootstrap95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "unit": "source-image group",
        "replicates": 2000,
        "interpretation": "descriptive resampling interval, not causal or population proof",
    }
