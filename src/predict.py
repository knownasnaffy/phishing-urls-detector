"""Shared prediction helper. Loads model once at import."""

import joblib

from features import extract_features, url_to_feature_row

_model = joblib.load("model/model.pkl")


def predict(url: str) -> dict:
    """
    Returns:
        {"label": "Phishing"|"Legitimate", "confidence": float, "features": dict}
    """
    row       = url_to_feature_row(url)
    pred      = int(_model.predict(row)[0])
    proba     = _model.predict_proba(row)[0]
    confidence = float(proba[pred])
    return {
        "label":      "Phishing" if pred == 1 else "Legitimate",
        "confidence": confidence,
        "features":   extract_features(url),
    }
