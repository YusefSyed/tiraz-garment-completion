"""Reject configurations the fixed experiment implementation does not support."""

from __future__ import annotations

import math
import re
from typing import Any

from tiraz_completion.data import N_CLASSES


def validate_training_config(config: dict[str, Any]) -> None:
    for field in ("epochs", "batch_size", "embedding_dim", "hidden_dim"):
        if type(config.get(field)) is not int or config[field] <= 0:
            raise ValueError(f"{field} must be a positive integer")
    for field in ("learning_rate", "weight_decay"):
        value = config.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field} must be numeric")
        if not math.isfinite(value):
            raise ValueError(f"{field} must be finite")
        if value < 0 or (field == "learning_rate" and value == 0):
            raise ValueError(f"invalid {field}")


def validate_protocol(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError("protocol must be an object")
    protocol: dict[str, Any] = value
    fixed = {
        "schema_version": 1,
        "main_category_ids": list(range(N_CLASSES)),
        "minimum_distinct_categories": 2,
        "train_partition": [0, 80],
        "tune_partition": [80, 90],
        "calibration_partition": [90, 100],
        "test_partition": "official val2020, excluding source groups present in train2020",
        "baselines": ["training target frequency", "smoothed training category co-occurrence"],
    }
    for field, expected in fixed.items():
        actual = protocol.get(field)
        if actual != expected or type(actual) is not type(expected):
            raise ValueError(f"unsupported fixed protocol field: {field}")
        if isinstance(expected, list):
            if not isinstance(actual, list) or any(
                type(a) is not type(b) for a, b in zip(actual, expected, strict=True)
            ):
                raise ValueError(f"unsupported fixed protocol types: {field}")
    for field in (
        "experiment_id",
        "split_seed",
        "task",
        "unit",
        "model_selection",
        "calibration",
        "conformal_score",
        "stress",
        "uncertainty",
        "policy",
        "statistical_limit",
    ):
        if not isinstance(protocol.get(field), str) or not protocol[field].strip():
            raise ValueError(f"{field} must be a nonempty string")
    validate_training_config(protocol)
    seeds = protocol.get("model_seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("model_seeds must be distinct nonnegative integers")
    alpha = protocol.get("conformal_alpha")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ValueError("conformal_alpha must be numeric")
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("conformal_alpha must be finite and in (0,1)")
    hashes = protocol.get("dataset_sha256")
    if not isinstance(hashes, dict) or set(hashes) != {"train", "official_validation"}:
        raise ValueError("both dataset hashes are required")
    if any(
        not isinstance(item, str) or not re.fullmatch(r"[a-f0-9]{64}", item)
        for item in hashes.values()
    ):
        raise ValueError("dataset hashes must be SHA256 digests")
    return protocol
