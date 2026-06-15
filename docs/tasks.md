# Tasks

## ✅ Phase 5: Inference Layer

- [x] `src/prepare_dataset.py` — save Tranco rank dict to `datasets/tranco_map.pkl`
- [x] `src/train_model.py` — save `list(X.columns)` to `model/features.pkl`
- [x] Create `src/features.py` — load lookup artifacts at import; expose `extract_features(url)` and `url_to_feature_row(url)`
- [x] Create `src/predict.py` — load model and features artifact at import; expose `predict(url) -> dict`
- [x] Create `src/cli.py` — single URL argument, print one-line verdict with confidence
- [x] Create `src/streamlit.py` — URL input, result card (verdict + confidence), expandable feature breakdown

## ✅ Phase 6: Feature Expansion

- [x] Rename `URLCharProb` → `URLCharFreqScore` in `src/prepare_dataset.py` and `src/features.py`
- [x] Add `URLShannonEntropy`, `NoOfHyphensInDomain`, `NoOfDotsInURL`, `DomainHasDigit`, `PathDepth`, `TLDIsFreeAbuse` to `extract_features()` in both files
- [x] Regenerate `datasets/features.csv` by re-running `src/prepare_dataset.py`
- [x] Retrain model — run `src/train_model.py`, update `model/model.pkl` and `model/features.pkl`
- [x] Re-benchmark — run `src/test_models.py`, save results to `benchmarks/`
- [x] Update `README.md` with new benchmark results
