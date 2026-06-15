# Phishing URL Detection

Classifies URLs as phishing or legitimate using only URL-derived features - no DNS lookups, no webpage fetching, no external queries at inference time.

## Dataset

Training data comes from two sources combined:

| Source | Label | Approx. count |
| ------ | ----- | ------------: |
| PhiUSIIL (`datasets/dataset.csv`, `label=0` rows) | Legitimate | ~100K |
| PhiUSIIL (`label=1` rows) + PhishTank (`datasets/phishing.csv`) | Phishing | ~200K |

PhiUSIIL covers phishing URLs up to 2023; PhishTank adds ~65K URLs from June 2026, making the phishing set more temporally diverse. The class imbalance (~1:2) is handled with `class_weight='balanced'` in all models.

The Tranco top-1M list (`datasets/legitimate.csv`) is used only as a domain rank lookup, not as training samples. Bare domains without scheme or path don't represent real legitimate traffic and would distort the feature distribution.

## Features

All features are derived from the URL string only:

`URLLength`, `DomainLength`, `IsDomainIP`, `TLDLength`, `NoOfSubDomain`, `HasObfuscation`, `NoOfObfuscatedChar`, `ObfuscationRatio`, `NoOfLettersInURL`, `LetterRatioInURL`, `NoOfDegitsInURL`, `DegitRatioInURL`, `NoOfEqualsInURL`, `NoOfQMarkInURL`, `NoOfAmpersandInURL`, `NoOfOtherSpecialCharsInURL`, `SpacialCharRatioInURL`, `IsHTTPS`, `CharContinuationRate`, `TLDLegitimateProb`, `URLCharProb`, `TopDomainRank`

`TLDLegitimateProb` and `URLCharProb` are precomputed from the legitimate training split and saved to `datasets/tld_probs.csv` and `datasets/char_probs.csv`.

## Benchmark Results

Trained and evaluated on the combined dataset above. 80/20 stratified train/test split, all models use `class_weight='balanced'`.

| Model | Accuracy | Precision | Recall | F1 | Train Time (s) | Predict Time (s) | File Size (MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RandomForest | 0.9357 | 0.9635 | 0.9387 | 0.9510 | 6.44 | 0.135 | 235.38 |
| ExtraTrees | 0.9331 | 0.9525 | 0.9464 | 0.9495 | 5.17 | 0.255 | 584.37 |
| GradientBoosting | 0.9144 | 0.9366 | 0.9343 | 0.9355 | 27.25 | 0.055 | 0.14 |
| LogisticRegression | 0.8654 | 0.9389 | 0.8530 | 0.8939 | 1.40 | 0.040 | 0.003 |

### Model Selection

RandomForest is the deployment model.

On the previous PhiUSIIL-only dataset, all models scored above 0.9997 F1 and LogisticRegression was a reasonable choice given its speed and tiny size. The combined dataset is harder - raw URLs with class imbalance and more temporal diversity - and the gap between LogisticRegression and tree models is now about 6 F1 points (0.8939 vs 0.9510). That's a meaningful accuracy difference that justifies the larger model.

Between RandomForest and ExtraTrees, RandomForest wins on accuracy and F1 while ExtraTrees takes over twice as much disk space (584 MB vs 235 MB) with slightly slower inference. GradientBoosting scores lower than both despite taking 4x longer to train.

| Rank | Model | Reason |
| ---- | ----- | ------ |
| 1 | RandomForest | Best accuracy and F1, reasonable training time |
| 2 | ExtraTrees | Slightly lower F1, much larger model size |
| 3 | GradientBoosting | Lower scores, slowest to train |
| 4 | LogisticRegression | Fast and tiny, but ~6 F1 points behind on this dataset |

### Previous Benchmarks (PhiUSIIL only, for reference)

These were taken on the old pipeline using the PhiUSIIL pre-extracted feature set. Scores are near-perfect because the dataset was cleaner and less diverse. See [this commit](https://github.com/knownasnaffy/phishing-urls-detector/tree/360c6fa) for the full old pipeline.

#### URL + Webpage features

| Model | Accuracy | Precision | Recall | F1 | Train Time (s) | Predict Time (s) | File Size (MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RandomForest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 3.04 | 0.045 | 2.14 |
| ExtraTrees | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 2.88 | 0.057 | 10.77 |
| GradientBoosting | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 35.17 | 0.045 | 0.11 |
| LogisticRegression | 0.9998 | 0.9997 | 1.0000 | 0.9999 | 0.42 | 0.035 | 0.004 |

#### URL-only features

| Model | Accuracy | Precision | Recall | F1 | Train Time (s) | Predict Time (s) | File Size (MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GradientBoosting | 0.9999 | 0.9998 | 1.0000 | 0.9999 | 15.91 | 0.038 | 0.10 |
| RandomForest | 0.9999 | 0.9998 | 1.0000 | 0.9999 | 2.43 | 0.042 | 3.06 |
| ExtraTrees | 0.9998 | 0.9997 | 1.0000 | 0.9999 | 2.04 | 0.052 | 10.82 |
| LogisticRegression | 0.9998 | 0.9996 | 1.0000 | 0.9998 | 0.30 | 0.017 | 0.003 |

Removing webpage features had almost no effect on performance, which confirms that URL-derived features carry most of the predictive signal on that dataset.

## Credits and Citations

### PhiUSIIL dataset `datasets/dataset.csv`

Prasad, A., & Chandra, S. (2023). PhiUSIIL: A diverse security profile empowered phishing URL detection framework based on similarity index and incremental learning. Computers & Security, 103545. doi: https://doi.org/10.1016/j.cose.2023.103545

### Tranco top-1M list `datasets/legitimate.csv`

Victor Le Pochat, Tom Van Goethem, Samaneh Tajalizadehkhoob, Maciej Korczyński, and Wouter Joosen. 2019. "Tranco: A Research-Oriented Top Sites Ranking Hardened Against Manipulation," Proceedings of the 26th Annual Network and Distributed System Security Symposium (NDSS 2019). https://doi.org/10.14722/ndss.2019.23386

### PhishTank `datasets/phishing.csv`

https://phishtank.org/
