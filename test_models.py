# Dataset source: https://archive-beta.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset/files?path=PhiUSIIL_Phishing_URL_Dataset.csv

import pandas as pd
import joblib
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

print('Reading dataset file...')
dataset = pd.read_csv('./dataset.csv', sep=',', low_memory=False)

print(dataset.describe())
print(dataset.head())
print(dataset.dtypes)

dataset["HyphenCount"] = dataset["URL"].str.count("-")
X = dataset.drop(['FILENAME', 'Domain', 'URL', 'TLD', 'Title', 'label'], axis=1)
y = dataset['label']

print(f"\nDataset shape: {dataset.shape}")

print('\nIs Null: \n')
print(dataset.isna().sum().sum())

print('\nValue counts:')
print(y.value_counts())
print('\nValue counts (normalized):')
print(y.value_counts(normalize=True))

print('Extracting samples...')
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

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
    )
}

results = {}

for name, model in models.items():
    print(f"\n====================\n{name}\n====================")
    print("\nTraining...")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    results[name] = accuracy

    print(f"\nAccuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
        })

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False
        )

    print('\nFeatures:')
    print(importance_df)

    print('\nConfusion matrix:')
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    joblib.dump(model, f"model/test/{name}.pkl")

print("\n=== Summary ===")

for name, accuracy in sorted(
    results.items(),
    key=lambda x: x[1],
    reverse=True
):
    print(f"{name:<20} {accuracy:.4f}")

