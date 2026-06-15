# Tasks

## Phase 5: Inference Layer

- [x] `src/prepare_dataset.py` — after building the Tranco rank dict, save it to `datasets/tranco_map.pkl`
- [x] `src/train_model.py` — after the train/test split, save `list(X.columns)` to `model/features.pkl`
- [x] Create `src/features.py` — load lookup artifacts at import; expose `extract_features(url)` and `url_to_feature_row(url)`
- [x] Create `src/predict.py` — load model and features artifact at import; expose `predict(url) -> dict`
- [x] Create `src/cli.py` — single URL argument, print one-line verdict with confidence
- [x] Create `src/streamlit.py` — URL input, result card (verdict + confidence), expandable feature breakdown
