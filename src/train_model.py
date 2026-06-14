# Dataset source: https://archive-beta.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset/files?path=PhiUSIIL_Phishing_URL_Dataset.csv

import os
import time

import pandas as pd
import joblib
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

from columns import webpage_columns, identifier_column, unknown_columns

parser = argparse.ArgumentParser()
parser.add_argument(
    "--url-only",
    action="store_true",
    help="Use only URL-based features"
)
parser.add_argument(
    "--debug",
    action="store_true",
    help="Print debug information"
)
parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Don't write to filesystem"
)

args = parser.parse_args()

def dprint(any):
    if args.debug:
        print(any)


print('Reading dataset file...')
dataset = pd.read_csv('../datasets/dataset.csv', sep=',', low_memory=False)

dprint(dataset.describe())
dprint(dataset.head())
dprint(dataset.dtypes)

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

if identifier_column not in columns_to_drop:
    columns_to_drop.extend([identifier_column])

columns_to_drop.extend(unknown_columns)

if args.url_only:
    print("Using URL-only features")

    columns_to_drop.extend(webpage_columns)


dataset["HyphenCount"] = dataset["URL"].str.count("-")
X = dataset.drop(columns=columns_to_drop)
X = X.drop(columns=[target_column])
y = dataset[target_column]

dprint(f"\nDataset shape: {dataset.shape}")

dprint('\nIs Null: \n')
dprint(dataset.isna().sum().sum())

dprint('\nValue counts:')
dprint(y.value_counts())
dprint('\nValue counts (normalized):')
dprint(y.value_counts(normalize=True))

print('\nExtracting samples...')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

dprint(f"\nTraining samples: {len(y_train)}")
dprint(f"Testing samples: {len(y_test)}")

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=2000,
        random_state=42,
        class_weight=None
    ))
])

results = {}
benchmark_results = []

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

dprint(classification_report(y_test, y_pred))

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

dprint("\nTop Coefficients:")
dprint(coef_df.head(20))

dprint('\nConfusion matrix:')
cm = confusion_matrix(y_test, y_pred)
dprint(cm)

file_size_mb = 'N/A'

if not args.dry_run:
    model_path = "model/model.pkl"

    os.makedirs("model", exist_ok=True)
    joblib.dump(model, model_path)

    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)

benchmark_results.append({
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1": f1,
    "Train Time (s)": train_time,
    "Predict Time (s)": predict_time,
    "File Size (MB)": file_size_mb
})

summary_df = pd.DataFrame(benchmark_results)

summary_df = summary_df.sort_values(
    by="F1",
    ascending=False
)

os.makedirs("benchmarks", exist_ok=True)

timestamp = time.strftime("%Y%m%d-%H%M%S")

benchmark_path = (
    f"benchmarks/benchmark-{timestamp}-training"
    f"{'-url-only' if args.url_only else ''}.csv"
)

print("\nSummary")
print(summary_df.round(4))

if not args.dry_run:
    summary_df.to_csv(
        benchmark_path,
        index=False
    )
    dprint(f"\nBenchmark saved to: {benchmark_path}")

