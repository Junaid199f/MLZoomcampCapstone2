"""Training script exported from notebooks/eda_and_training.ipynb."""

import json
import joblib
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("EEG-Eye-State.csv")
MODEL_PATH = Path("models/model.pkl")
META_PATH = Path("models/metadata.json")


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def split_data(df: pd.DataFrame):
    X = df.drop(columns=["eyeDetection"])
    y = df["eyeDetection"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate_models(X_train, X_val, y_train, y_val):
    models = {
        "log_reg": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000)),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        "gradient_boost": GradientBoostingClassifier(random_state=42),
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_val)[:, 1]
        else:
            proba = pred
        results.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_val, pred),
                "roc_auc": roc_auc_score(y_val, proba),
            }
        )

    return results


def tune_random_forest(X_train, y_train):
    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [None, 8, 16],
        "min_samples_split": [2, 5],
    }
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid = GridSearchCV(rf, param_grid, scoring="roc_auc", cv=3, n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid


def final_evaluation(model, X_test, y_test):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "classification_report": classification_report(y_test, pred, digits=4),
    }
    return metrics


def save_artifacts(model, feature_names, best_params):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    meta = {
        "features": list(feature_names),
        "target": "eyeDetection",
        "best_params": best_params,
    }
    with META_PATH.open("w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    df = load_data(DATA_PATH)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    baseline_results = evaluate_models(X_train, X_val, y_train, y_val)
    print("Baseline models:")
    for res in baseline_results:
        print(res)

    grid = tune_random_forest(X_train, y_train)
    print("Best params:", grid.best_params_)
    print("Best CV ROC AUC:", grid.best_score_)

    best_model = grid.best_estimator_
    best_model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))

    metrics = final_evaluation(best_model, X_test, y_test)
    print("Final metrics:")
    print("Accuracy:", metrics["accuracy"])
    print("ROC AUC:", metrics["roc_auc"])
    print(metrics["classification_report"])

    save_artifacts(best_model, X_train.columns, grid.best_params_)
    print(f"Saved model to {MODEL_PATH}")
