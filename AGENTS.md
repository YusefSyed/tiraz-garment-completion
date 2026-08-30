# Project agent map

## Purpose and authority

This is Tiraz's annotation-only category-completion research companion. It trains
and evaluates models on unordered category sets; it is not a vision model or a
human preference evaluator. Current source and executable tests outrank prose.
`protocol.json` and `src/tiraz_completion/protocol.py` jointly define supported
experiments; `.python-version`, `pyproject.toml` and `uv.lock` define the runtime.

## Evidence boundaries

- Preserve the committed `artifacts/v1` experiment. New analysis or changed methods
  require a new protocol/output version; never tune against v1 test outcomes or
  silently regenerate its published results.
- Fashionpedia annotations and ontology have their own attribution terms in
  `DATA_LICENSE.md`. Do not download or publish source images under that license.
- Keep annotation caches outside Git. Source URLs are hashed for grouping and
  never enter model features or published prediction rows.
- Source groups, masked targets and calibration roles are distinct identities.
  Keep train, tuning, temperature, conformal and test groups disjoint.
- Prediction-set guarantees require assumptions; measured stress coverage does
  not establish real-world coverage. Report all seeds and both conditions.

## First files and checks

| Symptom | Inspect first | Acceptance evidence |
| --- | --- | --- |
| Split or duplicate discrepancy | `data.py`: `source_group`, `split_examples` | Group-disjoint and null-URL regression tests; data manifest |
| Protocol differs from execution | `protocol.py`, `protocol.json` | Unsupported settings and hash mismatches fail before training |
| Unexpected training result | `models.py`, frozen tuning histories | Seeded tiny-signal/reproducibility tests; no test-set selection |
| Misleading confidence or metric | `metrics.py` | Invalid-label/NaN guards, quantile/tie and hand-counted tests |
| Artifact mismatch | `run.py`, source/weight manifests | Check all hashes, compare frozen independent runs |

## Workflow

Inspect the relevant diff and preserve others' changes. Implement the smallest
coherent change, add a regression for an actual failure, and update both sides of
changed contracts. Do not call providers or change the mobile app from this repo.

From the repository root:

```sh
uv sync --frozen
uv run ruff check .
uv run mypy
uv run pytest
```

Use the exact reproduction command in `README.md` with a new empty output path.
Validation is complete when focused tests, typing, lint, data/weight checksums,
and applicable reproducibility checks pass. Record finite-sample limits and any
deviation; do not infer production outcomes from this experiment.
