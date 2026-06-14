# Implementation Plan

## Phase 1: Dataset Preparation (`src/prepare_dataset.py`)

Extract raw URLs from their sources and produce a single feature matrix ready for training.

1. Load legitimate URLs from PhiUSIIL (`datasets/dataset.csv`, `label=0` rows, `URL` column).
2. Load phishing URLs by combining PhiUSIIL (`label=1` rows) and PhishTank (`datasets/phishing.csv`, `url` column). Deduplicate by URL string.
3. Extract URL-only features from every URL (see `docs/concept.md` for the full feature list).
4. For statistically-derived features (`TLDLegitimateProb`, `URLCharProb`), compute lookup tables from the legitimate training URLs and save them to `datasets/tld_probs.csv` and `datasets/char_probs.csv` so reruns skip recomputation.
5. Load the Tranco list (`datasets/legitimate.csv`) into memory as a rank lookup and compute `TopDomainRank` for every URL.
6. Write the final feature matrix with a `label` column to `datasets/features.csv`.

## Phase 2: Model Training (`src/train_model.py`)

Train the primary deployment model and save it.

1. Load `datasets/features.csv`, split 80/20 train/test with stratification.
2. Train a `LogisticRegression` pipeline (StandardScaler + classifier, `class_weight='balanced'`).
3. Print metrics and save the model to `model/model.pkl`.

## Phase 3: Benchmarking (`src/test_models.py`)

Compare all four candidate models on the same split.

1. Use the same train/test split as Phase 2 (fixed `random_state`).
2. Train and evaluate LogisticRegression, RandomForest, ExtraTrees, and GradientBoosting -- all with `class_weight='balanced'`.
3. Save a benchmark CSV to `benchmarks/` with accuracy, precision, recall, F1, train time, predict time, and model file size.

## Phase 4: Evaluation

Review the benchmark results and decide on the production model. Update `README.md` with the new results. If any model significantly outperforms LogisticRegression, revisit the deployment choice from `docs/concept.md`.
