# Tasks

## Phase 1: Dataset Preparation (`src/prepare_dataset.py`)

- [x] Load legitimate URLs from PhiUSIIL (`datasets/dataset.csv`, `label=0` rows, `URL` column)
- [x] Load phishing URLs from PhiUSIIL (`label=1` rows) and PhishTank (`datasets/phishing.csv`, `url` column); deduplicate by URL string
- [x] Extract all URL-only features (see `docs/concept.md` feature table) from every URL
- [x] Compute `TLDLegitimateProb` lookup table from legitimate training URLs; save to `datasets/tld_probs.csv`
- [x] Compute `URLCharProb` lookup table from legitimate training URLs; save to `datasets/char_probs.csv`
- [x] Load Tranco list (`datasets/legitimate.csv`) as a rank lookup and compute `TopDomainRank` per URL
- [x] Write final feature matrix + `label` column to `datasets/features.csv`

## Phase 2: Model Training (`src/train_model.py`)

- [x] Load `datasets/features.csv` and split 80/20 train/test with stratification (fixed `random_state`)
- [x] Train a `LogisticRegression` pipeline (StandardScaler + classifier, `class_weight='balanced'`)
- [x] Save trained model to `model/model.pkl`

## Phase 3: Benchmarking (`src/test_models.py`)

- [ ] Reuse the same train/test split (same `random_state`) as Phase 2
- [ ] Train and evaluate LogisticRegression, RandomForest, ExtraTrees, and GradientBoosting (all with `class_weight='balanced'`)
- [ ] Record accuracy, precision, recall, F1, train time, predict time, and model file size
- [ ] Save benchmark results to `benchmarks/`

## Phase 4: Evaluation

- [ ] Review benchmark CSV and compare models
- [ ] Confirm LogisticRegression as deployment model, or document reason for switching
- [ ] Update `README.md` with new benchmark tables and dataset composition notes
