"""Streaming public annotations into image-group-disjoint completion cases."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import ijson
import numpy as np
from numpy.typing import NDArray

N_CLASSES = 27


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest(), 16)


def source_group(image: dict[str, Any]) -> str:
    """Group repeated source URLs; discard fragments but preserve path/query case."""
    original = image.get("original_url")
    if original is not None and not isinstance(original, str):
        raise ValueError("source URL must be a string or null")
    url = original.strip() if original else ""
    if url:
        parts = urlsplit(url)
        key = urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, parts.query, ""))
    else:
        key = f"public-image-id:{image['id']}"
    return hashlib.sha256(key.encode()).hexdigest()


@dataclass(frozen=True)
class Example:
    group: str
    context: tuple[int, ...]
    target: int

    def __post_init__(self) -> None:
        if not self.context or tuple(sorted(set(self.context))) != self.context:
            raise ValueError("context must be a nonempty sorted set")
        if self.target in self.context:
            raise ValueError("masked target leaked into context")
        if any(not 0 <= value < N_CLASSES for value in (*self.context, self.target)):
            raise ValueError("category out of range")


def read_group_sets(path: Path) -> tuple[dict[str, tuple[int, ...]], dict[str, int]]:
    """No images or attributes enter training; memory is bounded by image/category sets."""
    images: dict[int, str] = {}
    with path.open("rb") as stream:
        for image in ijson.items(stream, "images.item"):
            image_id = int(image["id"])
            if image_id in images:
                raise ValueError("duplicate image identity")
            images[image_id] = source_group(image)
    categories: dict[int, set[int]] = defaultdict(set)
    annotation_count = 0
    with path.open("rb") as stream:
        for annotation in ijson.items(stream, "annotations.item"):
            annotation_count += 1
            image_id = int(annotation["image_id"])
            if image_id not in images:
                raise ValueError("annotation references absent image")
            category = int(annotation["category_id"])
            if 0 <= category < N_CLASSES and not annotation.get("iscrowd", 0):
                categories[image_id].add(category)
    # Deterministic representative prevents one repeated source inflating sample size.
    representatives: dict[str, int] = {}
    for image_id in sorted(images):
        representatives.setdefault(images[image_id], image_id)
    groups = {
        group: tuple(sorted(categories[image_id])) for group, image_id in representatives.items()
    }
    return groups, {
        "images": len(images),
        "annotations": annotation_count,
        "source_groups": len(groups),
        "duplicate_source_images_removed": len(images) - len(groups),
    }


def make_example(group: str, categories: tuple[int, ...], salt: str) -> Example:
    if len(categories) < 2 or len(set(categories)) != len(categories):
        raise ValueError("at least two distinct categories required")
    ordered = tuple(sorted(categories))
    target = ordered[stable_int(f"{salt}:target:{group}") % len(ordered)]
    return Example(group, tuple(value for value in ordered if value != target), target)


def split_examples(
    train_groups: dict[str, tuple[int, ...]],
    test_groups: dict[str, tuple[int, ...]],
    salt: str,
) -> tuple[dict[str, list[Example]], dict[str, int]]:
    splits: dict[str, list[Example]] = {
        name: [] for name in ("train", "tune", "temp", "cal", "test")
    }
    for group, categories in sorted(train_groups.items()):
        if len(categories) < 2:
            continue
        bucket = stable_int(f"{salt}:split:{group}") % 100
        split = (
            "train"
            if bucket < 80
            else "tune"
            if bucket < 90
            else ("temp" if stable_int(f"{salt}:cal-half:{group}") % 2 == 0 else "cal")
        )
        splits[split].append(make_example(group, categories, salt))
    overlap = set(train_groups) & set(test_groups)
    for group, categories in sorted(test_groups.items()):
        if group not in overlap and len(categories) >= 2:
            splits["test"].append(make_example(group, categories, salt))
    seen: set[str] = set()
    for name, examples in splits.items():
        current = {example.group for example in examples}
        if seen & current:
            raise ValueError(f"source group leakage in {name}")
        seen.update(current)
    return splits, {"official_test_groups_excluded_for_train_overlap": len(overlap)}


def arrays(examples: list[Example]) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    x = np.zeros((len(examples), N_CLASSES), dtype=np.float32)
    y = np.array([example.target for example in examples], dtype=np.int64)
    for row, example in enumerate(examples):
        x[row, list(example.context)] = 1
    return x, y


def stressed(examples: list[Example], salt: str) -> list[Example]:
    result = []
    for example in examples:
        keep = max(1, len(example.context) // 2)
        ranked = sorted(
            example.context, key=lambda cat: stable_int(f"{salt}:stress:{example.group}:{cat}")
        )
        result.append(Example(example.group, tuple(sorted(ranked[:keep])), example.target))
    return result


def prepare(
    train_path: Path, test_path: Path, protocol: dict[str, Any]
) -> tuple[dict[str, list[Example]], dict[str, Any]]:
    train, train_counts = read_group_sets(train_path)
    test, test_counts = read_group_sets(test_path)
    splits, overlap = split_examples(train, test, str(protocol["split_seed"]))
    if any(not examples for examples in splits.values()):
        raise ValueError("every frozen partition must be nonempty")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "sources": {
            "train": {"sha256": digest_file(train_path), **train_counts},
            "official_validation": {"sha256": digest_file(test_path), **test_counts},
        },
        "partition_counts": {name: len(items) for name, items in splits.items()},
        "partition_group_digests": {
            name: hashlib.sha256("\n".join(item.group for item in items).encode()).hexdigest()
            for name, items in splits.items()
        },
        "overlap_checks": {**overlap, "partition_group_intersection": 0},
        "input_scope": "main-apparel category IDs only; no pixels, URLs or attributes",
    }
    return splits, manifest


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
