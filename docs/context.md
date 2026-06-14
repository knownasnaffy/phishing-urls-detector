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

## Architecture

**Do not add an `--extended` mode with webpage or WHOIS features.**
Phishing URLs are ephemeral -- many are taken down before or shortly after appearing in datasets like PhishTank or PhiUSIIL. Fetching webpage content or WHOIS data retroactively is not possible for a large portion of phishing samples, leading to systematic missingness on exactly the class that matters most. An extended mode would only be viable for a live real-time scanner architecture, which is a different project.

**Do not drop data to fix class imbalance.**
The dataset has roughly 100K legitimate vs 200K phishing samples. Downsampling the majority class discards real signal. Instead, all models use `class_weight='balanced'` to compensate during training.
