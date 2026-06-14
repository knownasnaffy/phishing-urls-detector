# Dataset source: https://archive-beta.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset/files?path=PhiUSIIL_Phishing_URL_Dataset.csv

import os
import time

import pandas as pd
import joblib
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import argparse

from columns import webpage_columns, identifier_column

parser = argparse.ArgumentParser()
parser.add_argument(
    "--url-only",
    action="store_true",
    help="Use only URL-based features"
)

args = parser.parse_args()

print('Reading dataset file...')
dataset = pd.read_csv('../datasets/dataset.csv', sep=',', low_memory=False)

print(dataset.describe())
print(dataset.head())
print(dataset.dtypes)

# Label is handled separately
target_column = "label"

# Automatically find all non-numeric columns
non_numeric_columns = dataset.select_dtypes(
    exclude=["number", "bool"]
).columns.tolist()

# Remove label from drop list if present
columns_to_drop = [
    col for col in non_numeric_columns
    if col != target_column
]

if args.url_only:
    print("Using URL-only features")

    columns_to_drop.extend(webpage_columns)


dataset["HyphenCount"] = dataset["URL"].str.count("-")
X = dataset.drop(columns=columns_to_drop)
X = X.drop(columns=[target_column])
y = dataset[target_column]

print(f"\nDataset shape: {dataset.shape}")

print('\nIs Null: \n')
print(dataset.isna().sum().sum())

print('\nValue counts:')
print(y.value_counts())
print('\nValue counts (normalized):')
print(y.value_counts(normalize=True))

print('Extracting samples...')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining samples: {len(y_train)}")
print(f"Testing samples: {len(y_test)}")

models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),

    "ExtraTrees": ExtraTreesClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),

    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=100,
        random_state=42
    ),

    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            random_state=42,
            class_weight="balanced"
        ))
    ])
}

results = {}
benchmark_results = []

for name, model in models.items():
    print(f"\n====================\n{name}\n====================")
    print("\nTraining...")

    start_train = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start_train

    start_predict = time.perf_counter()
    y_pred = model.predict(X_test)
    predict_time = time.perf_counter() - start_predict

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results[name] = accuracy

    print(f"\nAccuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "feature": X.columns,
            "importance": model.feature_importances_
        })

        importance_df = importance_df.sort_values(
            by="importance",
            ascending=False
        )

        print("\nTop Features:")
        print(importance_df.head(20))

    elif name == "LogisticRegression":
        coeffs = model.named_steps["classifier"].coef_[0]

        coef_df = pd.DataFrame({
            "feature": X.columns,
            "coefficient": coeffs
        })

        coef_df["abs_coef"] = coef_df["coefficient"].abs()

        coef_df = coef_df.sort_values(
            by="abs_coef",
            ascending=False
        )

        print("\nTop Coefficients:")
        print(coef_df.head(20))

    print('\nConfusion matrix:')
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    model_path = f"model/test/{name}.pkl"

    os.makedirs("model/test", exist_ok=True)
    joblib.dump(model, model_path)

    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)

    benchmark_results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Train Time (s)": train_time,
        "Predict Time (s)": predict_time,
        "File Size (MB)": file_size_mb
    })

print("\n=== Summary ===")

for name, accuracy in sorted(
    results.items(),
    key=lambda x: x[1],
    reverse=True
):
    print(f"{name:<20} {accuracy:.4f}")

summary_df = pd.DataFrame(benchmark_results)

summary_df = summary_df.sort_values(
    by="F1",
    ascending=False
)

os.makedirs("benchmarks", exist_ok=True)

timestamp = time.strftime("%Y%m%d-%H%M%S")

benchmark_path = (
    f"benchmarks/benchmark-{timestamp}"
    f"{'-url-only' if args.url_only else ''}.csv"
)

summary_df.to_csv(
    benchmark_path,
    index=False
)

print("\n=== Benchmark Summary ===")
print(summary_df.round(4))

print(f"\nBenchmark saved to: {benchmark_path}")
