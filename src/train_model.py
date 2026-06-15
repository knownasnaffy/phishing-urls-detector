"""Phase 2: Train LogisticRegression pipeline and save to model/model.pkl."""

import os
import time

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES_CSV = "datasets/features.csv"
MODEL_PATH   = "model/model.pkl"
RANDOM_STATE = 42

print("Loading features.csv …")
df = pd.read_csv(FEATURES_CSV)

X = df.drop(columns=["label"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"Train: {len(y_train):,}  |  Test: {len(y_test):,}")

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )),
])

print("Training …")
t0 = time.perf_counter()
model.fit(X_train, y_train)
train_time = time.perf_counter() - t0

t0 = time.perf_counter()
y_pred = model.predict(X_test)
predict_time = time.perf_counter() - t0

print(f"\nAccuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1:        {f1_score(y_test, y_pred):.4f}")
print(f"Train time: {train_time:.2f}s  |  Predict time: {predict_time:.4f}s")

os.makedirs("model", exist_ok=True)
joblib.dump(model, MODEL_PATH)
size_kb = os.path.getsize(MODEL_PATH) / 1024
print(f"\nModel saved to {MODEL_PATH}  ({size_kb:.1f} KB)")
