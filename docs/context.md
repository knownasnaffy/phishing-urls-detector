# Decision Context

A record of things we decided not to do and why, to avoid revisiting them.

## Datasets

**Do not use Tranco domains as training samples for legitimate URLs.**
The Tranco list contains bare domains (e.g. `google.com`) with no scheme, path, or query string. Synthesizing URLs from them (e.g. prepending `http://`) produces artificially clean samples that don't reflect real legitimate traffic. The list also includes infrastructure and CDN domains (e.g. `gstatic.com`, `data.microsoft.com`) that are not user-facing sites. Using them would create a feature distribution mismatch at inference time.

**Do not use Common Crawl as a legitimate URL source.**
Common Crawl crawls the entire web without filtering -- it contains both legitimate and malicious URLs. Using it as a clean source would require cross-referencing against a blocklist (PhishTank, URLhaus, etc.), which is significant extra work with no guarantee of completeness.

**Do not use the old `datasets/dataset.csv` feature columns directly.**
The files under `src/` and the old feature CSV were built against the PhiUSIIL pre-extracted dataset. That pipeline is abandoned. The raw `URL` and `label` columns from `datasets/dataset.csv` are still used as a data source, but nothing else.

## Features

**Do not include `URLSimilarityIndex`.**
This feature requires comparing the URL against a reference corpus at inference time. That violates the URL-only constraint and makes inference dependent on an external lookup.

**Do not use the raw `TLD` string as a feature.**
High-cardinality categorical encoding of TLD would inflate the feature space and cause issues with unseen TLDs at inference time. Replaced by `TLDLegitimateProb`, a continuous score derived from TLD frequency in the training split.

**Do not implement `DomainSimilarityIndex` (for now).**
Computing meaningful similarity against a large reference corpus (e.g. Tranco 1M) requires either an approximate nearest-neighbor index (MinHash LSH via `datasketch`) or a brute-force edit distance pass -- both are heavyweight. The signal it provides (typosquatting detection) is partially covered by `TopDomainRank`, `DomainLength`, and `TLDLegitimateProb`. Can be revisited if benchmarks show a meaningful accuracy gap.

**`URLCharProb` renamed to `URLCharFreqScore`.**
The `Prob` suffix implied a true probability output, but the value is a frequency-derived score based on character distribution in legitimate training URLs. `FreqScore` more accurately reflects what it measures.

**`URLShannonEntropy` split into `DomainShannonEntropy` + `PathShannonEntropy`. `URLCharFreqScore` scoped to hostname only.**
Whole-URL entropy and char freq penalise URLs with random-looking paths (UUIDs, session IDs, slugs) even when the domain is clean — e.g. `chatgpt.com/c/saljkdjkd-asd-asddsa-dsad` was misclassified as phishing. Legitimate apps (ChatGPT, Notion, GitHub, Dropbox) routinely generate such paths. Splitting entropy lets the model learn that high path entropy on a known domain is normal, while high domain entropy is suspicious. Scoping `URLCharFreqScore` to hostname ensures the char frequency signal reflects domain-level patterns, not path noise. The char frequency table in `datasets/char_probs.csv` is now built from legitimate hostnames rather than full URLs.

**Do not add `HasAtSymbol`.**
The `@` trick in URLs is very old; modern browsers display a warning for it and modern phishing campaigns rarely use it. Low occurrence rate means it provides negligible signal.

**Do not add `HasDoubleSlash`.**
Intended to catch redirect tricks like `http://legit.com//redirect?url=evil.com`. Weak signal in practice and not captured by existing features (slashes are in the exclusion set for `NoOfOtherSpecialCharsInURL`). Not worth the marginal gain.

**Do not add `SuspiciousKeywordScore` (login, verify, account, etc.).**
Training data bias makes this unreliable. The legitimate set comes from PhiUSIIL URLs which are not full-path crawled URLs — they mostly lack path segments like `/login` or `/account`. The phishing set contains them abundantly. The feature would be measuring "has a path" more than "is phishing", producing a heavily biased signal that would not generalize.

**Do not add `QueryParamCount`.**
Redundant with `NoOfAmpersandInURL`, which already counts `&` characters — a direct proxy for query parameter count. Legitimate analytics and tracking URLs also routinely carry 10+ parameters, reducing signal quality. No marginal value over the existing feature.

**Do not add `PathLength`.**
Path length is already partially captured by `URLLength` minus `DomainLength`. As a standalone feature it lacks discriminative power — legitimate sites can have long paths and phishing sites can have short ones.

**Use `TLDIsFreeAbuse` instead of `TLDIsCountryCode`.**
Country-code TLDs (`.in`, `.uk`, `.de`, etc.) are used legitimately by millions of websites and are not a reliable phishing signal. The real signal is a small set of Freenom free TLDs (`.tk`, `.ml`, `.ga`, `.cf`, `.gq`) that have historically been offered at no cost and are massively abused for phishing campaigns. A binary flag for exactly this set is precise and defensible.

## Architecture

**Do not add an `--extended` mode with webpage or WHOIS features.**
Phishing URLs are ephemeral -- many are taken down before or shortly after appearing in datasets like PhishTank or PhiUSIIL. Fetching webpage content or WHOIS data retroactively is not possible for a large portion of phishing samples, leading to systematic missingness on exactly the class that matters most. An extended mode would only be viable for a live real-time scanner architecture, which is a different project.

**Do not drop data to fix class imbalance.**
The dataset has roughly 100K legitimate vs 200K phishing samples. Downsampling the majority class discards real signal. Instead, all models use `class_weight='balanced'` to compensate during training.

**Deployment model is RandomForest, not LogisticRegression.**
On the combined PhiUSIIL + PhishTank dataset, RandomForest achieves F1 0.9510 vs LogisticRegression's 0.8939 — a ~6-point gap that justifies the larger model size (235 MB). LogisticRegression remains the better choice only on the old PhiUSIIL-only dataset where all models score near-perfect.

## Inference / UI Architecture

**Save Tranco rank dict as `datasets/tranco_map.pkl` during dataset preparation.**
Reloading the 1M-row CSV at every inference startup adds unnecessary latency. A joblib pickle of the `{domain: rank}` dict loads in milliseconds.

**Save feature column order as `model/features.pkl` during training.**
`train_model.py` trains on `X = df.drop(columns=["label"])`, which inherits column order from `features.csv`. Inference must reproduce this exact order. Saving `list(X.columns)` after the split and reindexing in `features.py` guarantees alignment regardless of dict insertion order.

**Shared prediction logic lives in `src/predict.py`.**
Both `src/cli.py` and `src/streamlit.py` call a single `predict(url) -> dict` function rather than duplicating model loading and feature pipeline logic. The return value includes `label`, `confidence` (predict_proba of predicted class), and `features` (the extracted feature dict for display).

**Do not expose raw `predict_proba` values directly in UIs.**
Show confidence as the probability of the predicted class (e.g. "99.1% confidence it's phishing"), not the raw `[p_legit, p_phish]` array. This is less ambiguous for users.

**Two UI entry points, both in `src/`.**
`src/cli.py` for terminal use (one URL argument, one-line output). `src/streamlit.py` for the GUI (single-column, result card with expandable feature breakdown). No separate package structure needed given the project scope.