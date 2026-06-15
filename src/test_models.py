"""Phase 3: Benchmark all four models on datasets/features.csv."""

import os
import time

import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES_CSV = "datasets/features.csv"
RANDOM_STATE = 42

print("Loading features.csv …")
df = pd.read_csv(FEATURES_CSV)
X = df.drop(columns=["label"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"Train: {len(y_train):,}  |  Test: {len(y_test):,}")

models = {
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, class_weight="balanced")),
    ]),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"),
    "ExtraTrees": ExtraTreesClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
}

os.makedirs("benchmarks", exist_ok=True)
os.makedirs("model/test", exist_ok=True)

rows = []
for name, model in models.items():
    print(f"\nTraining {name} …")

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    predict_time = time.perf_counter() - t0

    model_path = f"model/test/{name}.pkl"
    joblib.dump(model, model_path)
    size_mb = os.path.getsize(model_path) / (1024 * 1024)

    row = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "Train Time (s)": round(train_time, 4),
        "Predict Time (s)": round(predict_time, 4),
        "File Size (MB)": round(size_mb, 4),
    }
    rows.append(row)
    print(f"  Accuracy={row['Accuracy']:.4f}  F1={row['F1']:.4f}  "
          f"Train={train_time:.2f}s  Predict={predict_time:.4f}s  Size={size_mb:.2f}MB")

results_df = pd.DataFrame(rows).sort_values("F1", ascending=False)

timestamp = time.strftime("%Y%m%d-%H%M%S")
out_path = f"benchmarks/benchmark-{timestamp}.csv"
results_df.to_csv(out_path, index=False)

print(f"\n=== Benchmark Summary ===")
print(results_df.round(4).to_string(index=False))
print(f"\nSaved to {out_path}")
