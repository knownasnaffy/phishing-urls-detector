# Concept: Phishing URL Detector (New Pipeline)

## Overview

This pipeline replaces the old one built on the PhiUSIIL pre-extracted dataset. Instead of relying on pre-extracted features, we extract URL-only features from raw URLs sourced from Tranco (legitimate) and PhishTank (phishing), then train and benchmark multiple classifiers.

The goal is a lightweight model that classifies a URL using only information available in the URL string itself -- no DNS lookups, no webpage fetching, no external queries at inference time.

---

## Datasets

### Raw source files

| File | Source | Format | Notes |
|------|--------|--------|-------|
| `datasets/legitimate.csv` | Tranco top-1M | `rank,domain` (no header) | Used as a lookup artifact only, not as training samples |
| `datasets/phishing.csv` | PhishTank | CSV with header, use `url` column | ~65,048 phishing URLs from June 2026 |
| `datasets/dataset.csv` | PhiUSIIL (Prasad & Chandra, 2023) | Pre-extracted features + `URL` + `label` columns | Source of both legitimate and phishing training URLs |

### Training data composition

| Split | Source | Approx. count |
|-------|--------|---------------|
| Legitimate (label=0) | PhiUSIIL rows where `label=0`, `URL` column | ~100K |
| Phishing (label=1) | PhiUSIIL rows where `label=1` + PhishTank `url` column, deduplicated | ~200K |

Combining PhiUSIIL phishing URLs (~135K, collected up to 2023) with PhishTank URLs (~65K, June 2026) makes the phishing set more temporally diverse, exposing the model to both older and newer phishing patterns.

Class imbalance (~1:2 legitimate to phishing) is handled with `class_weight='balanced'` in all sklearn models.

---

## Features

Derived by stripping out the webpage columns and `URLSimilarityIndex` (requires external lookup) from the original PhiUSIIL feature set. The remaining URL-only features are:

| Feature | Description |
|---------|-------------|
| `URLLength` | Total character length of the URL |
| `DomainLength` | Character length of the domain part |
| `IsDomainIP` | 1 if the hostname is an IP address, 0 otherwise |
| `TLDLength` | Character length of the top-level domain |
| `NoOfSubDomain` | Number of subdomains (dots in hostname minus 1) |
| `HasObfuscation` | 1 if URL contains obfuscated characters (e.g. `%xx`) |
| `NoOfObfuscatedChar` | Count of percent-encoded characters |
| `ObfuscationRatio` | `NoOfObfuscatedChar / URLLength` |
| `NoOfLettersInURL` | Count of alphabetic characters in the URL |
| `LetterRatioInURL` | `NoOfLettersInURL / URLLength` |
| `NoOfDegitsInURL` | Count of digit characters in the URL |
| `DegitRatioInURL` | `NoOfDegitsInURL / URLLength` |
| `NoOfEqualsInURL` | Count of `=` characters |
| `NoOfQMarkInURL` | Count of `?` characters |
| `NoOfAmpersandInURL` | Count of `&` characters |
| `NoOfOtherSpecialCharsInURL` | Count of special chars excluding the above and common URL chars |
| `SpacialCharRatioInURL` | Special char count / URLLength |
| `IsHTTPS` | 1 if scheme is `https`, 0 otherwise |
| `CharContinuationRate` | Ratio of max run-length of any single character to URLLength |
| `TLDLegitimateProb` | Probability score of the TLD being benign (precomputed lookup table from training data) |
| `URLCharProb` | Average per-character probability based on character frequency in legitimate URLs |
| `TopDomainRank` | Rank of the domain in the Tranco top-1M list; 0 if not present |

Note: `TLD` (the raw TLD string) is dropped in favour of `TLDLegitimateProb` to avoid high-cardinality categorical encoding. `URLSimilarityIndex` is excluded entirely since it requires comparing against a reference corpus at inference time.

---

## Pipeline

### Step 1: `src/prepare_dataset.py`

- Load PhiUSIIL `datasets/dataset.csv`, use `URL` column. Label=0 rows → legitimate, label=1 rows → phishing.
- Load PhishTank `datasets/phishing.csv`, use `url` column → phishing. Deduplicate against PhiUSIIL phishing set.
- Extract the features listed above from every URL.
- Compute `TLDLegitimateProb` and `URLCharProb` lookup tables from legitimate URLs; save to `datasets/tld_probs.csv` and `datasets/char_probs.csv`.
- Load Tranco list (`datasets/legitimate.csv`) as a rank lookup; save compact dict to `datasets/tranco_map.pkl`.
- Output: `datasets/features.csv` with one row per URL and a `label` column.

### Step 2: `src/train_model.py`

- Load `datasets/features.csv`, split 80/20 train/test with stratification.
- Train RandomForest (`class_weight='balanced'`).
- Save model to `model/model.pkl`.
- Save feature column order to `model/features.pkl`.

### Step 3: `src/test_models.py`

- Same split as above.
- Train and evaluate LogisticRegression, RandomForest, ExtraTrees, GradientBoosting — all with `class_weight='balanced'`.
- Save benchmark CSV to `benchmarks/`.

### Step 4: `src/features.py` (inference)

- Loads lookup artifacts once at import (`tld_probs.csv`, `char_probs.csv`, `tranco_map.pkl`).
- `extract_features(url)` — pure URL string parsing.
- `url_to_feature_row(url)` — applies lookups, returns a single-row DataFrame with columns in `model/features.pkl` order.

### Step 5: `src/predict.py` (shared prediction helper)

- Loads `model/model.pkl` and `model/features.pkl` once at import.
- `predict(url) -> dict` — returns `{"label": str, "confidence": float, "features": dict}`.
  - `label`: `"Phishing"` or `"Legitimate"`
  - `confidence`: `predict_proba` probability of the predicted class

### Step 6: `src/cli.py`

- Takes one URL as a CLI argument.
- Calls `predict(url)`, prints a one-line verdict with confidence.

### Step 7: `src/streamlit.py`

- Single-column layout: URL input → "Check" button → result card (verdict + confidence).
- Expandable section shows the 22 extracted features.

---

## Model Selection Rationale

Benchmarked on the combined PhiUSIIL + PhishTank dataset (URL-only features):

| Rank | Model | Reason |
|------|-------|--------|
| 1 | **RandomForest** | Best accuracy (0.9357) and F1 (0.9510); reasonable size (235 MB) |
| 2 | ExtraTrees | Slightly lower F1, 2.5× larger model file |
| 3 | GradientBoosting | Lower scores, 4× slower to train |
| 4 | LogisticRegression | ~6 F1 points behind tree models on this dataset |

Deployment model: **RandomForest**. LogisticRegression was the best choice on the old PhiUSIIL-only dataset (near-perfect scores, tiny size), but the combined dataset is harder and the 6-point F1 gap justifies the larger model.
