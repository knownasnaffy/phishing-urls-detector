# Project: Phishing URL Detector

Academic project for a CSE grad student (AI for Cybersecurity training course).

## Goal

Build a phishing URL classifier using only URL-derived features (no webpage content, DNS lookups, or external data at inference time). Train and benchmark multiple ML models, then select the best for deployment.

## Datasets

| File | Source | Format | Size |
|------|--------|--------|------|
| `datasets/legitimate.csv` | Tranco top-1M list | `rank,domain` (no header) | ~1,000,000 rows |
| `datasets/phishing.csv` | PhishTank | CSV with header: `phish_id,url,phish_detail_url,submission_time,verified,verification_time,online,target` | ~65,048 rows |
| `datasets/dataset.csv` | PhiUSIIL (Prasad & Chandra, 2023) | Pre-extracted features, multi-column | **Abandoned** — kept for reference only |

Legitimate file has no header; columns are rank and domain. Only the domain/URL column is needed.
Phishing file has a header; the `url` column is the relevant field.

## Source Files

Files under `src/` are **reference only** — they were built against `datasets/dataset.csv` and are not being reused.

| File | Purpose |
|------|---------|
| `src/columns.py` | Feature column definitions (webpage vs URL-only split) |
| `src/prepare_dataset.py` | Empty — was placeholder |
| `src/train_model.py` | Trains a single LogisticRegression pipeline, outputs benchmark CSV |
| `src/test_models.py` | Benchmarks all four model types |

## Previous Benchmark Results (reference, from dataset.csv)

See `README.md` for full tables. Key takeaway: **URL-only features achieved near-identical performance to URL+webpage features**, making a URL-only model viable and lightweight.

Best deployment choice from prior work: **LogisticRegression** — 99.98%+ F1, 2.6 KB model, fastest train/inference.

## New Pipeline Plan

1. `src/prepare_dataset.py` — Load both CSVs, label them (0 = legitimate, 1 = phishing), extract URL features, output a combined `datasets/features.csv`
2. `src/train_model.py` — Train models on `features.csv`, save best model to `model/`
3. `src/test_models.py` — Benchmark all models, save results to `benchmarks/`

## Feature Engineering (URL-only)

Features to extract from raw URLs (informed by prior work on PhiUSIIL dataset):

- URL length
- Domain length
- Number of dots, hyphens, underscores, slashes, `@`, `?`, `=`, `&` in URL
- Subdomain count (dots in hostname)
- Has IP address as hostname
- Has HTTPS
- TLD (encoded or binary flags for suspicious TLDs)
- Path depth
- Digit ratio in URL
- Letter ratio in URL
- Presence of suspicious keywords (login, secure, account, update, verify, etc.)
- URL entropy (randomness measure)

Additional features from prior work worth keeping:
- `IsHTTPS`
- `URLLength`
- `DomainLength`
- `NoOfSubDomain`
- `NoOfDots`, `NoOfHyphen`, `NoOfAt`, `NoOfQmark`, `NoOfEqual`, `NoOfAmpersand`
- `NoOfDigit`, `NoOfLetter`
- `HyphenCount` (added manually in old pipeline)
