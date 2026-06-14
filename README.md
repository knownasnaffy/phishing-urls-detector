# Phishing URL Detection

## Selecting a model

Deciding which model to use in production and various other things.

> These benchmarks were taken with previous dataset from [this dataset](#Phishing_and_Legitimate_URL_combo). Checkout [this commit](https://github.com/knownasnaffy/phishing-urls-detector/tree/360c6fa) for a more complete version. New stuff is being prepared from scratch rn.

### Full Suite: URL + Webpage

#### Benchmark Results

| Model              | Accuracy | Precision | Recall |     F1 | Train Time (s) | Predict Time (s) | File Size (MB) |
| ------------------ | -------: | --------: | -----: | -----: | -------------: | ---------------: | -------------: |
| RandomForest       |   1.0000 |    1.0000 | 1.0000 | 1.0000 |         3.0358 |           0.0449 |         2.1384 |
| ExtraTrees         |   1.0000 |    1.0000 | 1.0000 | 1.0000 |         2.8838 |           0.0573 |        10.7696 |
| GradientBoosting   |   1.0000 |    1.0000 | 1.0000 | 1.0000 |        35.1655 |           0.0446 |         0.1051 |
| LogisticRegression |   0.9998 |    0.9997 | 1.0000 | 0.9999 |         0.4182 |           0.0354 |         0.0038 |

#### Notes

| Goal                           | Best Choice                                  | Reason                                                                  |
| ------------------------------ | -------------------------------------------- | ----------------------------------------------------------------------- |
| Highest predictive performance | RandomForest / ExtraTrees / GradientBoosting | All achieved perfect scores on the test set                             |
| Fastest training               | LogisticRegression                           | ~7× faster than RandomForest and ~84× faster than GradientBoosting      |
| Fastest inference              | LogisticRegression                           | Lowest prediction latency                                               |
| Smallest model                 | LogisticRegression                           | Only 3.8 KB versus MB-sized tree models                                 |
| Best overall deployment choice | LogisticRegression                           | Nearly identical performance with dramatically lower computational cost |

### URL-Only Features

#### Benchmark Results

| Model              | Accuracy | Precision | Recall |     F1 | Train Time (s) | Predict Time (s) | File Size (MB) |
| ------------------ | -------: | --------: | -----: | -----: | -------------: | ---------------: | -------------: |
| GradientBoosting   |   0.9999 |    0.9998 | 1.0000 | 0.9999 |        15.9099 |           0.0379 |         0.0988 |
| RandomForest       |   0.9999 |    0.9998 | 1.0000 | 0.9999 |         2.4339 |           0.0416 |         3.0588 |
| ExtraTrees         |   0.9998 |    0.9997 | 1.0000 | 0.9999 |         2.0384 |           0.0518 |        10.8240 |
| LogisticRegression |   0.9998 |    0.9996 | 1.0000 | 0.9998 |         0.2980 |           0.0170 |         0.0026 |

### Observations

Compared to the full-feature benchmark:

* Performance barely changed, suggesting the URL-based features alone carry nearly all predictive power.
* Logistic Regression remains the fastest and smallest model by a large margin.
* Gradient Boosting achieved the highest scores but required ~53× more training time than Logistic Regression.
* Random Forest offers almost identical performance to Gradient Boosting while training ~6.5× faster.
* Extra Trees trained slightly faster than Random Forest but produced the largest model.

### Ranking for Deployment

| Rank | Model              | Reason                                                                                  |
| ---- | ------------------ | --------------------------------------------------------------------------------------- |
| 1    | LogisticRegression | 99.98%+ performance with the fastest training/inference and a model size of only 2.6 KB |
| 2    | RandomForest       | Matches Gradient Boosting's metrics while training much faster                          |
| 3    | GradientBoosting   | Marginally best metrics but significantly slower to train                               |
| 4    | ExtraTrees         | Similar performance but largest model size and slower inference                         |

The most interesting result here is that **removing non-URL features had almost no impact on performance**. That suggests URL-derived features alone are highly informative and may be sufficient for a lightweight phishing detector. This is useful if we want a model that can classify URLs immediately without fetching webpage content, DNS data, titles, or other external information.

## Credits and Citations

### Phishing and Legitimate URL combo `datasets/dataset.csv`

Prasad, A., & Chandra, S. (2023). PhiUSIIL: A diverse security profile empowered phishing URL detection framework based on similarity index and incremental learning. Computers & Security, 103545. doi: https://doi.org/10.1016/j.cose.2023.103545

### Legitimate URLs list from Tranco `datasets/legitimate.csv`

Victor Le Pochat, Tom Van Goethem, Samaneh Tajalizadehkhoob, Maciej Korczyński, and Wouter Joosen. 2019. "Tranco: A Research-Oriented Top Sites Ranking Hardened Against Manipulation," Proceedings of the 26th Annual Network and Distributed System Security Symposium (NDSS 2019). https://doi.org/10.14722/ndss.2019.23386

### Phishing URLs from PhishTank `datasets/phishing.csv`

https://phishtank.org/
