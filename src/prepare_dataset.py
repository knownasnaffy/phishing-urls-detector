"""Phase 1: Extract URL-only features and produce datasets/features.csv."""

import math
import re
import os
from collections import Counter
from urllib.parse import urlparse

import joblib
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
DATASET_CSV   = "datasets/dataset.csv"
PHISHING_CSV  = "datasets/phishing.csv"
TRANCO_CSV    = "datasets/legitimate.csv"
TLD_PROBS     = "datasets/tld_probs.csv"
CHAR_PROBS    = "datasets/char_probs.csv"
FEATURES_CSV  = "datasets/features.csv"
TRANCO_MAP    = "datasets/tranco_map.pkl"

# ── Helpers ───────────────────────────────────────────────────────────────────
_IP_RE  = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
_PCT_RE = re.compile(r"%[0-9a-fA-F]{2}")
_SPECIAL_EXCL = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
                    "0123456789.-_~:/?#[]@!$&'()*+,;=%")


def _tld(hostname: str) -> str:
    parts = hostname.rstrip(".").split(".")
    return parts[-1].lower() if parts else ""


def extract_features(url: str) -> dict:
    try:
        p = urlparse(url if "://" in url else "http://" + url)
    except Exception:
        p = urlparse("")
    hostname = p.hostname or ""
    full = url

    url_len   = len(full)
    obf_chars = _PCT_RE.findall(full)
    n_obf     = len(obf_chars)
    letters   = sum(c.isalpha() for c in full)
    digits    = sum(c.isdigit() for c in full)
    specials  = sum(c not in _SPECIAL_EXCL for c in full)

    # CharContinuationRate: max run of any single char / url_len
    max_run = 1
    cur_run = 1
    for i in range(1, len(full)):
        if full[i] == full[i-1]:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 1
    ccr = max_run / url_len if url_len else 0

    tld_str = _tld(hostname)
    # subdomains = dots in hostname minus 1 (for registrable domain)
    dots_in_host = hostname.count(".")
    n_sub = max(0, dots_in_host - 1)

    return {
        "URLLength":               url_len,
        "DomainLength":            len(hostname),
        "IsDomainIP":              int(bool(_IP_RE.match(hostname))),
        "TLDLength":               len(tld_str),
        "NoOfSubDomain":           n_sub,
        "HasObfuscation":          int(n_obf > 0),
        "NoOfObfuscatedChar":      n_obf,
        "ObfuscationRatio":        n_obf / url_len if url_len else 0,
        "NoOfLettersInURL":        letters,
        "LetterRatioInURL":        letters / url_len if url_len else 0,
        "NoOfDegitsInURL":         digits,
        "DegitRatioInURL":         digits / url_len if url_len else 0,
        "NoOfEqualsInURL":         full.count("="),
        "NoOfQMarkInURL":          full.count("?"),
        "NoOfAmpersandInURL":      full.count("&"),
        "NoOfOtherSpecialCharsInURL": specials,
        "SpacialCharRatioInURL":   specials / url_len if url_len else 0,
        "IsHTTPS":                 int(p.scheme == "https"),
        "CharContinuationRate":    ccr,
        "_tld":                    tld_str,   # temp, removed before output
        "_hostname":               hostname,  # temp
    }


# ── Load sources ──────────────────────────────────────────────────────────────
print("Loading PhiUSIIL dataset.csv …")
phiusiil = pd.read_csv(DATASET_CSV, usecols=["URL", "label"])
legit_urls   = phiusiil.loc[phiusiil["label"] == 0, "URL"].dropna().tolist()
phish_phiusiil = phiusiil.loc[phiusiil["label"] == 1, "URL"].dropna().tolist()

print("Loading PhishTank phishing.csv …")
phishtank_urls = pd.read_csv(PHISHING_CSV, usecols=["url"])["url"].dropna().tolist()

# Deduplicate phishing set
phish_set = list(dict.fromkeys(phish_phiusiil + phishtank_urls))
print(f"  Legitimate: {len(legit_urls):,}  |  Phishing: {len(phish_set):,}")

# ── Lookup tables (computed from legitimate training URLs) ────────────────────
if os.path.exists(TLD_PROBS) and os.path.exists(CHAR_PROBS):
    print("Loading existing lookup tables …")
    tld_prob_map  = pd.read_csv(TLD_PROBS,  index_col="tld")["prob"].to_dict()
    char_prob_map = pd.read_csv(CHAR_PROBS, index_col="char")["prob"].to_dict()
else:
    print("Computing TLD and char probability tables from legitimate URLs …")
    tld_counter  = Counter()
    char_counter = Counter()
    for url in legit_urls:
        feats = extract_features(url)
        tld_counter[feats["_tld"]] += 1
        char_counter.update(url)

    total_tld  = sum(tld_counter.values())
    total_char = sum(char_counter.values())
    tld_prob_map  = {t: c / total_tld  for t, c in tld_counter.items()}
    char_prob_map = {c: n / total_char for c, n in char_counter.items()}

    pd.DataFrame(
        list(tld_prob_map.items()), columns=["tld", "prob"]
    ).to_csv(TLD_PROBS, index=False)
    pd.DataFrame(
        list(char_prob_map.items()), columns=["char", "prob"]
    ).to_csv(CHAR_PROBS, index=False)
    print(f"  TLD table: {len(tld_prob_map)} entries | Char table: {len(char_prob_map)} entries")

# ── Tranco rank lookup ────────────────────────────────────────────────────────
print("Loading Tranco list …")
tranco = pd.read_csv(TRANCO_CSV, header=None, names=["rank", "domain"],
                     usecols=["rank", "domain"])
tranco_map = dict(zip(tranco["domain"].str.lower(), tranco["rank"]))
joblib.dump(tranco_map, TRANCO_MAP)
print(f"  Saved {TRANCO_MAP}  ({len(tranco_map):,} entries)")


def top_domain_rank(hostname: str) -> int:
    """Return Tranco rank or 0 if not found. Try subdomain stripping."""
    h = hostname.lower()
    while h:
        if h in tranco_map:
            return int(tranco_map[h])
        parts = h.split(".", 1)
        if len(parts) < 2:
            break
        h = parts[1]
    return 0


# ── Feature extraction ────────────────────────────────────────────────────────
def build_row(url: str, label: int) -> dict:
    f = extract_features(url)
    tld = f.pop("_tld")
    hostname = f.pop("_hostname")

    f["TLDLegitimateProb"] = tld_prob_map.get(tld, 0.0)

    char_probs = [char_prob_map.get(c, 0.0) for c in url]
    f["URLCharProb"] = sum(char_probs) / len(char_probs) if char_probs else 0.0

    f["TopDomainRank"] = top_domain_rank(hostname)
    f["label"] = label
    return f


print("Extracting features …")
rows = []
total = len(legit_urls) + len(phish_set)
checkpoint = total // 20  # 5% intervals

for i, url in enumerate(legit_urls):
    rows.append(build_row(url, 0))
    if (i + 1) % checkpoint == 0:
        print(f"  {(i+1)/total*100:.0f}%", end=" ", flush=True)

for i, url in enumerate(phish_set):
    rows.append(build_row(url, 1))
    idx = len(legit_urls) + i + 1
    if idx % checkpoint == 0:
        print(f"  {idx/total*100:.0f}%", end=" ", flush=True)

print()

# ── Write output ──────────────────────────────────────────────────────────────
os.makedirs("datasets", exist_ok=True)
df = pd.DataFrame(rows)
df.to_csv(FEATURES_CSV, index=False)
print(f"Wrote {FEATURES_CSV}  ({len(df):,} rows × {len(df.columns)} columns)")
print(df["label"].value_counts().to_string())
