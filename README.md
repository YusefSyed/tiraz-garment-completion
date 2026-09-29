# Tiraz: garment completion with calibrated uncertainty

[![CI](https://github.com/YusefSyed/tiraz-garment-completion/actions/workflows/ci.yml/badge.svg)](https://github.com/YusefSyed/tiraz-garment-completion/actions/workflows/ci.yml)

An annotation-only PyTorch experiment: hide one garment category in an annotated
image and predict it from the remaining unordered category set. The package
compares frequency and co-occurrence baselines with three seeded Deep Sets-style
encoders, then audits prediction sets when additional context is removed.

This adds actual training, data curation, calibration and experimental analysis
to Tiraz. It is **not** an image classifier, human preference study, fashion-taste
model, or production recommendation-quality claim. The existing app ranker and
historical extraction benchmark are unchanged.

## Completed result

The frozen experiment trained on **27,683 source-grouped examples**, with separate
tuning and two calibration partitions. On **962 official-validation source groups**,
the three neural seeds reached **58.5-59.0% top-1 accuracy**, versus **52.7%** for
co-occurrence and **37.8%** for frequency. Nominal 90% prediction-set coverage was
91.1-91.2% on the original condition but fell to 55.6-57.9% when more context was
withheld. That limitation is retained, not tuned away.

[Read the complete result](artifacts/v1/report.md) and
[all metrics](artifacts/v1/metrics.json). Two complete CPU runs reproduced all
11 artifacts byte-for-byte, including three trained checkpoints. These figures
describe this fixed category-completion task, not user-outfit performance.

## Inspect without training

The [result report](artifacts/v1/report.md), [metrics](artifacts/v1/metrics.json),
and [data manifest](artifacts/v1/data-manifest.json) are committed for review.
The test suite below uses small fixtures and does not download Fashionpedia or
rerun the full training experiment. The full reproduction is a separate step.

## Design

- Stream the official Fashionpedia annotations; do not download any images.
- Retain main-apparel categories 0-26; exclude garment parts, decorations and crowds.
- Group exact normalized source URLs, choose one deterministic image representative,
  and remove official-validation groups that occur in the training source.
- Create one deterministic masked target per eligible source group. Target category
  never appears in the visible context. IDs and source URLs are never model features.
- Split training-source groups into train, tuning, temperature-calibration and
  conformal-calibration partitions. The latter two are separate halves of the
  declared calibration allocation.
- Select checkpoints by tuning NLL; calibrate without seeing final-test outcomes.
- Report all three seeds, all baselines and both original/missing-context conditions.
  The stress condition is paired with the same source units, not extra samples.
- Bind data, source, configuration, model weights and output artifacts to SHA-256.
  Refuse unsupported protocol values, malformed numerical inputs and overwritten runs.

The methods themselves are established prior art. The contribution is the complete
task construction, reproducible comparison and explicit failure/uncertainty audit.

## Reproduce

Use Python 3.12 and uv from this directory. Linux resolves CPU-only PyTorch wheels.
The annotation downloads total approximately 557 MB; they remain outside Git.

```sh
git clone https://github.com/YusefSyed/tiraz-garment-completion.git
cd tiraz-garment-completion
uv sync --frozen
uv run ruff check .
uv run mypy
uv run pytest
mkdir -p ../fashionpedia-cache
curl -fL -o ../fashionpedia-cache/train.json https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_train2020.json
curl -fL -o ../fashionpedia-cache/val.json https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_val2020.json
uv run tiraz-completion --train-annotations ../fashionpedia-cache/train.json --test-annotations ../fashionpedia-cache/val.json --output artifacts/reproduction
```

The protocol contains expected input hashes. A changed download fails before any
training. Keep the output directory absent or empty; a previous result is never
silently overwritten. Run twice into separate directories to compare numerical
artifacts. Byte reproducibility is expected in the pinned CPU/software environment,
not promised across arbitrary hardware or framework versions.

## Read the result

- `data-manifest.json`: source sizes/counts, group overlap and partition digests.
- `training-history.json`: all epochs for all seeds, selected only by tuning loss.
- `metrics.json`: top-1, all-27-class macro F1, NLL, Brier score, calibration error,
  prediction-set coverage/size, singleton rate/accuracy and paired resampling intervals.
- `predictions.jsonl.gz`: all final predictions, including unfavorable outcomes;
  only opaque source-group hashes and category/probability data.
- `source-manifest.json`, `frozen-protocol.json`, `SHA256SUMS`: exact provenance.
- Seeded `.pt` state dictionaries: trained weights, included in integrity checks.

Conformal nominal coverage assumes exchangeability. Official validation may differ
from training/calibration, and the missing-context stress deliberately changes the
input distribution. Report measured coverage; do not claim an unconditional safety
guarantee. Wilson intervals and source-group bootstraps are descriptive diagnostics,
not proof that the dataset represents user wardrobes. Exact-URL grouping does not
establish pixel-level or semantic deduplication.

## Sources and license

See [DATA_LICENSE.md](DATA_LICENSE.md). Fashionpedia annotations/ontology are CC BY
4.0, with separate image rights. No private media is used. Method references:
[Deep Sets](https://arxiv.org/abs/1703.06114) and
[split conformal prediction](https://arxiv.org/abs/2107.07511).
