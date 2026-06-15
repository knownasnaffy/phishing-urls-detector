# Implementation Plan

## ✅ Completed

Phases 1–5 (dataset preparation, training, benchmarking, model selection, inference layer) are done.
Deployment model: **RandomForest** (`model/model.pkl`). See `README.md` for benchmark results.

---

## Phase 6: Feature Expansion (28 features)

Six new features are being added and one renamed. Both `src/prepare_dataset.py` and `src/features.py` must be updated together, then the dataset regenerated and models retrained.

### 6.1 Rename `URLCharProb` → `URLCharFreqScore` in `src/prepare_dataset.py` and `src/features.py`

The existing feature measures average character frequency in legitimate URLs — a score, not a probability.

### 6.2 Add 6 new features to `extract_features()` in both files

| Feature | Implementation note |
|---------|---------------------|
| `DomainShannonEntropy` | Shannon entropy over `hostname` characters only |
| `PathShannonEntropy` | Shannon entropy over `p.path` characters only |
| `NoOfHyphensInDomain` | `hostname.count("-")` |
| `NoOfDotsInURL` | `url.count(".")` |
| `DomainHasDigit` | `int(any(c.isdigit() for c in hostname))` |
| `PathDepth` | `len([s for s in p.path.split("/") if s])` |
| `TLDIsFreeAbuse` | `int(tld_str in {"tk", "ml", "ga", "cf", "gq"})` |

### 6.3 Regenerate `datasets/features.csv`

Re-run `src/prepare_dataset.py`. The new CSV will have 29 columns (28 features + `label`).

### 6.4 Retrain and re-benchmark

- Run `src/train_model.py` to update `model/model.pkl` and `model/features.pkl`.
- Run `src/test_models.py` to produce a new benchmark CSV in `benchmarks/`.
- Update `README.md` with new benchmark results.

### 6.5 Update `src/streamlit.py`

Change the feature count reference from 22 to 28 in the expandable section label (if hardcoded).
