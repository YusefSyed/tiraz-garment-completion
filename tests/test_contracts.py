from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tiraz_completion.data import (
    N_CLASSES,
    Example,
    arrays,
    digest_file,
    make_example,
    read_group_sets,
    source_group,
    split_examples,
    stressed,
)
from tiraz_completion.metrics import (
    conformal_threshold,
    fit_temperature,
    prediction_sets,
    probabilities,
    summarize,
)
from tiraz_completion.models import CompletionNet, baseline_logits, train_model
from tiraz_completion.protocol import validate_protocol
from tiraz_completion.run import execute


def test_source_identity_preserves_query_and_removes_fragment() -> None:
    first = source_group({"id": 1, "original_url": "HTTPS://Example.org/A.jpg?q=1#x"})
    same = source_group({"id": 2, "original_url": "https://example.org/A.jpg?q=1#y"})
    other = source_group({"id": 3, "original_url": "https://example.org/A.jpg?q=2"})
    assert first == same and first != other


def test_absent_source_urls_fall_back_to_distinct_image_ids() -> None:
    assert source_group({"id": 1, "original_url": None}) != source_group(
        {"id": 2, "original_url": None}
    )
    assert source_group({"id": 1, "original_url": None}) == source_group({"id": 1})
    with pytest.raises(ValueError, match="string or null"):
        source_group({"id": 1, "original_url": ["malformed"]})


def test_split_is_group_disjoint_and_removes_official_overlap() -> None:
    train = {f"group-{index}": (0, 1, 6) for index in range(1000)}
    test = {"group-7": (1, 2), "unseen": (0, 6)}
    splits, counts = split_examples(train, test, "test")
    assert counts["official_test_groups_excluded_for_train_overlap"] == 1
    assert [item.group for item in splits["test"]] == ["unseen"]
    groups = [{item.group for item in items} for items in splits.values()]
    assert sum(map(len, groups)) == len(set.union(*groups))
    assert splits == split_examples(dict(reversed(list(train.items()))), test, "test")[0]


def test_target_mask_and_context_corruption_preserve_unit_and_label() -> None:
    example = make_example("opaque", (0, 1, 6, 23), "salt")
    other = make_example("opaque", (23, 6, 1, 0), "salt")
    assert example == other
    changed = stressed([example], "salt")[0]
    assert (changed.group, changed.target) == (example.group, example.target)
    assert 0 < len(changed.context) < len(example.context)
    assert set(changed.context) < set(example.context)
    with pytest.raises(ValueError, match="leaked"):
        Example("leak", (0, 1), 0)


def annotation_fixture(path: Path, start: int, count: int) -> None:
    images = [
        {"id": i, "original_url": f"https://example.invalid/source/{i}"}
        for i in range(start, start + count)
    ]
    annotations = [
        {"image_id": image["id"], "category_id": cat, "iscrowd": 0}
        for image in images
        for cat in (0, 6, 23)
    ]
    path.write_text(json.dumps({"images": images, "annotations": annotations}))


def test_streaming_reader_excludes_parts_and_repeated_sources(tmp_path: Path) -> None:
    path = tmp_path / "small.json"
    path.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "original_url": "https://example.invalid/a"},
                    {"id": 2, "original_url": "https://example.invalid/a#repeat"},
                ],
                "annotations": [{"image_id": 1, "category_id": x} for x in (0, 6, 27, 40)],
            }
        )
    )
    groups, stats = read_group_sets(path)
    assert list(groups.values()) == [(0, 6)]
    assert stats["duplicate_source_images_removed"] == 1


def test_baselines_and_neural_model_never_predict_observed_category() -> None:
    x, y = arrays([Example("a", (0, 6), 23), Example("b", (1, 23), 6)])
    for logits in baseline_logits(x, y, x).values():
        assert np.all(probabilities(logits)[x.astype(bool)] == 0)
    model = CompletionNet()
    logits = model(torch.from_numpy(x)).detach().numpy().astype(np.float64)
    assert np.all(probabilities(logits)[x.astype(bool)] == 0)


def test_tiny_training_learns_context_and_is_seed_reproducible() -> None:
    examples = [Example(f"a{i}", (0,), 1) for i in range(32)] + [
        Example(f"b{i}", (2,), 3) for i in range(32)
    ]
    x, y = arrays(examples)
    config = {
        "embedding_dim": 8,
        "hidden_dim": 16,
        "learning_rate": 0.03,
        "weight_decay": 0,
        "epochs": 12,
        "batch_size": 16,
    }
    model, history = train_model(x, y, x, y, config, 11)
    repeated, second_history = train_model(x, y, x, y, config, 11)
    assert history == second_history
    assert history[-1]["tuning_nll"] < history[0]["tuning_nll"]
    with torch.no_grad():
        assert torch.equal(model(torch.from_numpy(x)), repeated(torch.from_numpy(x)))
        assert (model(torch.from_numpy(x)).argmax(1).numpy() == y).mean() == 1


def test_conformal_order_statistic_uses_n_plus_one_and_inclusive_ties() -> None:
    p = np.zeros((9, N_CLASSES), dtype=np.float64)
    p[:, 0] = np.arange(1, 10) / 10
    p[:, 1] = 1 - p[:, 0]
    y = np.zeros(9, dtype=np.int64)
    assert conformal_threshold(p, y, 0.2) == pytest.approx(0.8)
    x = np.zeros_like(p, dtype=np.float32)
    sets = prediction_sets(p, x, 0.8)
    assert sets[1, 0]
    assert not sets[0, 0]
    assert conformal_threshold(p[:1], y[:1], 0.1) == 1
    x[0, 2] = 1
    assert not prediction_sets(p, x, 1)[0, 2]


def test_temperature_fit_reduces_overconfidence_without_changing_ranking() -> None:
    logits = np.full((10, N_CLASSES), -1e9)
    logits[:, 0], logits[:, 1] = 5, 0
    y = np.array([0] * 6 + [1] * 4, dtype=np.int64)
    t = fit_temperature(logits, y)
    assert t > 1
    before, after = probabilities(logits), probabilities(logits, t)
    assert -np.log(after[np.arange(10), y]).mean() < -np.log(before[np.arange(10), y]).mean()
    assert np.array_equal(before.argmax(1), after.argmax(1))


def test_metrics_match_hand_counted_predictions_and_singletons() -> None:
    p = np.zeros((2, N_CLASSES))
    p[0, :2] = [0.9, 0.1]
    p[1, :2] = [0.6, 0.4]
    y = np.array([0, 1], dtype=np.int64)
    sets = np.zeros_like(p, dtype=bool)
    sets[0, 0], sets[1, :2] = True, True
    result = summarize(p, y, sets)
    assert result["top1_accuracy"] == 0.5
    assert result["prediction_set_coverage"] == 1
    assert result["mean_set_size"] == 1.5
    assert result["singleton_rate"] == 0.5
    assert result["singleton_accuracy"] == 1


def test_end_to_end_exports_recompute_and_cannot_overwrite(tmp_path: Path) -> None:
    train_path, test_path = tmp_path / "train.json", tmp_path / "test.json"
    annotation_fixture(train_path, 0, 250)
    annotation_fixture(test_path, 1000, 15)
    protocol = json.loads(Path("protocol.json").read_text())
    protocol.update({"epochs": 1, "model_seeds": [17]})
    protocol["dataset_sha256"] = {
        "train": digest_file(train_path),
        "official_validation": digest_file(test_path),
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    output = tmp_path / "output"
    result = execute(train_path, test_path, protocol_path, output)
    rows = [
        json.loads(line)
        for line in gzip.decompress((output / "predictions.jsonl.gz").read_bytes()).splitlines()
    ]
    for name in result["models"]:
        subset = [row for row in rows if row["model"] == name and row["condition"] == "test"]
        correct = sum(int(np.argmax(row["probabilities"])) == row["target"] for row in subset)
        assert correct / len(subset) == result["models"][name]["test"]["top1_accuracy"]
    hashes = (output / "SHA256SUMS").read_text().splitlines()
    assert any("set_encoder_seed_17.pt" in line for line in hashes)
    for line in hashes:
        expected, name = line.split("  ", 1)
        assert digest_file(output / name) == expected
    with pytest.raises(ValueError, match="overwrite"):
        execute(train_path, test_path, protocol_path, output)


@pytest.mark.parametrize(
    "field,value",
    [
        ("epochs", 0),
        ("epochs", True),
        ("batch_size", 1.5),
        ("learning_rate", float("nan")),
        ("weight_decay", -1),
        ("model_seeds", []),
        ("model_seeds", [17, 17]),
        ("model_seeds", [True]),
        ("main_category_ids", [0, 1]),
        ("train_partition", [0, 90]),
        ("minimum_distinct_categories", 3),
        ("conformal_alpha", 1),
    ],
)
def test_protocol_rejects_silent_method_drift(field: str, value: object) -> None:
    protocol = json.loads(Path("protocol.json").read_text())
    protocol[field] = value
    with pytest.raises(ValueError):
        validate_protocol(protocol)


def test_numerical_inputs_fail_before_plausible_statistics() -> None:
    p = np.zeros((1, N_CLASSES))
    p[0, 26] = 1
    negative = np.array([-1], dtype=np.int64)
    with pytest.raises(ValueError, match="labels"):
        conformal_threshold(p, negative, 0.1)
    with pytest.raises(ValueError, match="labels"):
        summarize(p, negative, np.ones_like(p, dtype=bool))
    with pytest.raises(ValueError, match="finite"):
        fit_temperature(np.full((1, N_CLASSES), np.nan), np.array([0], dtype=np.int64))
    with pytest.raises(ValueError, match="labels"):
        fit_temperature(np.zeros((2, N_CLASSES)), np.array([0], dtype=np.int64))
    with pytest.raises(ValueError, match="distributions"):
        conformal_threshold(p * 2, np.array([26], dtype=np.int64), 0.1)
    with pytest.raises(ValueError, match="boolean"):
        summarize(p, np.array([26], dtype=np.int64), np.ones_like(p))
