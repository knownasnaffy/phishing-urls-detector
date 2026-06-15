"""Inference feature extraction. Loads lookup artifacts once at import."""

import re
from urllib.parse import urlparse

import joblib
import pandas as pd

# ── Artifacts (loaded once) ───────────────────────────────────────────────────
_tld_prob_map  = pd.read_csv("datasets/tld_probs.csv",  index_col="tld")["prob"].to_dict()
_char_prob_map = pd.read_csv("datasets/char_probs.csv", index_col="char")["prob"].to_dict()
_tranco_map    = joblib.load("datasets/tranco_map.pkl")
_feature_cols  = joblib.load("model/features.pkl")

# ── Helpers ───────────────────────────────────────────────────────────────────
_IP_RE       = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
_PCT_RE      = re.compile(r"%[0-9a-fA-F]{2}")
_SPECIAL_EXCL = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
                    "0123456789.-_~:/?#[]@!$&'()*+,;=%")


def _tld(hostname: str) -> str:
    parts = hostname.rstrip(".").split(".")
    return parts[-1].lower() if parts else ""


def _top_domain_rank(hostname: str) -> int:
    h = hostname.lower()
    while h:
        if h in _tranco_map:
            return int(_tranco_map[h])
        parts = h.split(".", 1)
        if len(parts) < 2:
            break
        h = parts[1]
    return 0


# ── Public API ────────────────────────────────────────────────────────────────
def extract_features(url: str) -> dict:
    """Parse URL string and return raw feature dict (no lookup tables)."""
    try:
        p = urlparse(url if "://" in url else "http://" + url)
    except Exception:
        p = urlparse("")
    hostname = p.hostname or ""
    url_len  = len(url)

    obf_chars = _PCT_RE.findall(url)
    n_obf     = len(obf_chars)
    letters   = sum(c.isalpha() for c in url)
    digits    = sum(c.isdigit() for c in url)
    specials  = sum(c not in _SPECIAL_EXCL for c in url)

    max_run = cur_run = 1
    for i in range(1, len(url)):
        if url[i] == url[i - 1]:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 1

    tld_str = _tld(hostname)
    n_sub   = max(0, hostname.count(".") - 1)

    return {
        "URLLength":                  url_len,
        "DomainLength":               len(hostname),
        "IsDomainIP":                 int(bool(_IP_RE.match(hostname))),
        "TLDLength":                  len(tld_str),
        "NoOfSubDomain":              n_sub,
        "HasObfuscation":             int(n_obf > 0),
        "NoOfObfuscatedChar":         n_obf,
        "ObfuscationRatio":           n_obf / url_len if url_len else 0,
        "NoOfLettersInURL":           letters,
        "LetterRatioInURL":           letters / url_len if url_len else 0,
        "NoOfDegitsInURL":            digits,
        "DegitRatioInURL":            digits / url_len if url_len else 0,
        "NoOfEqualsInURL":            url.count("="),
        "NoOfQMarkInURL":             url.count("?"),
        "NoOfAmpersandInURL":         url.count("&"),
        "NoOfOtherSpecialCharsInURL": specials,
        "SpacialCharRatioInURL":      specials / url_len if url_len else 0,
        "IsHTTPS":                    int(p.scheme == "https"),
        "CharContinuationRate":       max_run / url_len if url_len else 0,
        "TLDLegitimateProb":          _tld_prob_map.get(tld_str, 0.0),
        "URLCharProb":                sum(_char_prob_map.get(c, 0.0) for c in url) / url_len if url_len else 0.0,
        "TopDomainRank":              _top_domain_rank(hostname),
    }


def url_to_feature_row(url: str) -> pd.DataFrame:
    """Return a single-row DataFrame with columns in model/features.pkl order."""
    return pd.DataFrame([extract_features(url)])[_feature_cols]
