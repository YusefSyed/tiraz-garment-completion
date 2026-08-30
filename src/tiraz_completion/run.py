"""Frozen, CPU-only experiment entrypoint. No images, providers, or app writes."""

from __future__ import annotations

import argparse
import gzip
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tiraz_completion.data import arrays, digest_file, prepare, stressed, write_json
from tiraz_completion.metrics import (
    conformal_threshold,
    fit_temperature,
    paired_accuracy_interval,
    prediction_sets,
    probabilities,
    summarize,
)
from tiraz_completion.models import baseline_logits, predict_logits, train_model
from tiraz_completion.protocol import validate_protocol


def execute(train_path: Path, test_path: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol = validate_protocol(json.loads(protocol_path.read_text()))
    for name, path in (("train", train_path), ("official_validation", test_path)):
        if digest_file(path) != protocol["dataset_sha256"][name]:
            raise ValueError(f"frozen {name} dataset hash mismatch")
    if output.exists() and any(output.iterdir()):
        raise ValueError("output must be empty; never overwrite a prior experiment")
    output.mkdir(parents=True, exist_ok=True)
    # Write the specification before loading data or computing any result.
    write_json(output / "frozen-protocol.json", protocol)
    code_root = Path(__file__).parent
    source_digests = {path.name: digest_file(path) for path in sorted(code_root.glob("*.py"))}
    write_json(
        output / "source-manifest.json",
        {
            "protocol_sha256": digest_file(protocol_path),
            "source_sha256": source_digests,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": "cpu",
            "deterministic_algorithms": True,
            "threads": 2,
        },
    )
    print("Preparing annotation-only source-group partitions", flush=True)
    examples, data_manifest = prepare(train_path, test_path, protocol)
    write_json(output / "data-manifest.json", data_manifest)
    data = {name: arrays(items) for name, items in examples.items()}
    data["stress"] = arrays(stressed(examples["test"], str(protocol["split_seed"])))
    train_x, train_y = data["train"]
    tune_x, tune_y = data["tune"]
    print(json.dumps(data_manifest["partition_counts"], sort_keys=True), flush=True)

    all_logits: dict[str, dict[str, Any]] = {}
    for partition in ("temp", "cal", "test", "stress"):
        baselines = baseline_logits(train_x, train_y, data[partition][0])
        for name, logits in baselines.items():
            all_logits.setdefault(name, {})[partition] = logits
    training: dict[str, Any] = {}
    for seed in protocol["model_seeds"]:
        name = f"set_encoder_seed_{seed}"
        print(f"Training {name}", flush=True)
        model, history = train_model(train_x, train_y, tune_x, tune_y, protocol, int(seed))
        training[name] = history
        torch.save(model.state_dict(), output / f"{name}.pt")
        all_logits[name] = {
            partition: predict_logits(model, data[partition][0])
            for partition in ("temp", "cal", "test", "stress")
        }
    write_json(output / "training-history.json", training)

    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": protocol["experiment_id"],
        "scope": protocol["task"],
        "models": {},
        "limitations": [protocol["policy"], protocol["statistical_limit"]],
    }
    output_rows: list[dict[str, Any]] = []
    calibrated: dict[str, dict[str, Any]] = {}
    for name, logits_by_partition in all_logits.items():
        temperature = fit_temperature(logits_by_partition["temp"], data["temp"][1])
        cal_p = probabilities(logits_by_partition["cal"], temperature)
        q = conformal_threshold(cal_p, data["cal"][1], float(protocol["conformal_alpha"]))
        model_result: dict[str, Any] = {
            "temperature": temperature,
            "conformal_threshold": q,
            "calibration_score_n": len(data["cal"][1]),
        }
        calibrated[name] = {}
        for partition in ("test", "stress"):
            x, y = data[partition]
            p = probabilities(logits_by_partition[partition], temperature)
            calibrated[name][partition] = p
            sets = prediction_sets(p, x, q)
            model_result[partition] = summarize(p, y, sets)
            model_result[partition]["uncalibrated_nll"] = summarize(
                probabilities(logits_by_partition[partition]), y, sets
            )["negative_log_likelihood"]
            for index, example in enumerate(examples["test"]):
                output_rows.append(
                    {
                        "model": name,
                        "condition": partition,
                        "source_group": example.group,
                        "context": np.flatnonzero(x[index]).tolist(),
                        "target": int(y[index]),
                        "probabilities": p[index].tolist(),
                        "prediction_set": np.flatnonzero(sets[index]).tolist(),
                    }
                )
        result["models"][name] = model_result
    for name in training:
        for partition in ("test", "stress"):
            result["models"][name][partition]["vs_cooccurrence"] = paired_accuracy_interval(
                calibrated[name][partition],
                calibrated["cooccurrence"][partition],
                data[partition][1],
            )
    # Sorted, deterministic compression; no timestamp or original URLs in published rows.
    encoded = (
        "\n".join(
            json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
            for row in output_rows
        ).encode()
        + b"\n"
    )
    (output / "predictions.jsonl.gz").write_bytes(gzip.compress(encoded, mtime=0))
    write_json(output / "metrics.json", result)
    (output / "report.md").write_text(render_report(result, data_manifest))
    checksums = []
    for path in sorted(output.iterdir()):
        if path.is_file():
            checksums.append(f"{digest_file(path)}  {path.name}")
    (output / "SHA256SUMS").write_text("\n".join(checksums) + "\n")
    print("Completed frozen experiment; all model seeds and conditions reported", flush=True)
    return result


def render_report(result: dict[str, Any], manifest: dict[str, Any]) -> str:
    lines = [
        "# Tiraz annotation-only garment-category completion",
        "",
        "This is category-set completion, not image recognition, taste, preference, or "
        "production recommendation quality.",
        "",
        f"Source-group partitions: `{json.dumps(manifest['partition_counts'], sort_keys=True)}`.",
        "",
        "| Model | Condition | Top-1 | Macro F1 (27) | Set coverage | Mean set size | "
        "Singleton rate |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, record in result["models"].items():
        for condition in ("test", "stress"):
            m = record[condition]
            lines.append(
                f"| {name} | {condition} | {m['top1_accuracy']:.3f} | "
                f"{m['macro_f1_27_classes']:.3f} | {m['prediction_set_coverage']:.3f} | "
                f"{m['mean_set_size']:.2f} | {m['singleton_rate']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "All three neural seeds are shown, including unfavorable outcomes. The "
            "frequency and co-occurrence baselines use the same training partition. "
            "Checkpoint selection uses tuning data; temperatures and conformal quantiles "
            "use disjoint calibration halves. No final-test tuning occurs.",
            "",
            "The stress condition removes additional visible categories without changing "
            "the target or source-image unit. Its rows are paired observations, not extra "
            "independent samples. Conformal nominal coverage requires exchangeability; "
            "neither official validation nor context corruption guarantees it.",
            "",
            "Inspect metrics.json for calibration, paired resampling intervals and "
            "singleton accuracy. Empty/small prediction sets are not silently coerced "
            "into confident recommendations. The existing deterministic app ranker is unchanged.",
            "",
            "## Provenance",
            "",
            "Fashionpedia Consortium annotations/ontology: CC BY 4.0. "
            "https://fashionpedia.github.io/home/data_license.html . No images downloaded. "
            "Deep Sets method: https://arxiv.org/abs/1703.06114 . Split conformal reference: "
            "https://arxiv.org/abs/2107.07511 . These methods are prior art; the contribution "
            "is this reproducible task, leakage controls and comparative audit.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-annotations", required=True, type=Path)
    parser.add_argument("--test-annotations", required=True, type=Path)
    parser.add_argument("--protocol", default=Path("protocol.json"), type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    execute(args.train_annotations, args.test_annotations, args.protocol, args.output)


if __name__ == "__main__":
    main()
