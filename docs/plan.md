# Implementation Plan

## ✅ Completed

Phases 1–4 (dataset preparation, training, benchmarking, model selection) are done.
Deployment model: **RandomForest** (`model/model.pkl`). See `README.md` for benchmark results.

---

## Phase 5: Inference Layer

### 5.1 Update `src/prepare_dataset.py`

- After building the Tranco rank dict, save it to `datasets/tranco_map.pkl` (joblib) so inference doesn't reload the 1M-row CSV.

### 5.2 Update `src/train_model.py`

- After the train/test split, save `list(X.columns)` to `model/features.pkl` so inference can reindex feature dicts to the exact column order the model was trained on.

### 5.3 Create `src/features.py`

Extract reusable inference logic from `prepare_dataset.py`:

- Load lookup artifacts once at import time: `datasets/tld_probs.csv`, `datasets/char_probs.csv`, `datasets/tranco_map.pkl`.
- Expose `extract_features(url: str) -> dict` (the URL-string parsing logic).
- Expose `url_to_feature_row(url: str) -> pd.DataFrame` — applies lookup tables and Tranco rank, returns a single-row DataFrame with columns in the order from `model/features.pkl`.

### 5.4 Create `src/predict.py`

Shared prediction helper used by both UIs:

- Load `model/model.pkl` and `model/features.pkl` once at import.
- Expose a single function: `predict(url: str) -> dict` returning:
  ```python
  {"label": "Phishing" | "Legitimate", "confidence": float, "features": dict}
  ```
  `confidence` is `predict_proba` probability of the predicted class (0–1).

### 5.5 Create `src/cli.py`

- Accept one URL as a CLI argument.
- Call `predict(url)`, print a one-line result:
  ```
  ✓ Legitimate  (confidence: 97.3%)
  ✗ Phishing    (confidence: 99.1%)
  ```

### 5.6 Create `src/streamlit.py`

- Single-column layout: URL text input → "Check" button → result card.
- Result card shows verdict (large text, green/red) and confidence.
- Expandable section shows the 22 extracted features.
