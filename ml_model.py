

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

from create_dataset import DATASET_PATH, save_dataset
from reminder_engine import CATEGORIES, CONDITIONS, LOCATIONS

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "decision_tree_model.pkl"
TREE_IMAGE_PATH = MODELS_DIR / "decision_tree_plot.png"

NO_EXPIRY_PLACEHOLDER = 999      # used when no expiry date exists
TARGET_COLUMN = "reminder_status"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw columns into numbers a Decision Tree can use.

    * Numbers stay numbers (days_stored, days_to_expiry).
    * Text columns become 0/1 "one-hot" columns, e.g. category_Dairy = 1.
    * Missing expiry -> has_expiry_date = 0 and days_to_expiry = 999.
    The column order is always the same, so training and prediction match.
    """
    features = pd.DataFrame(index=df.index)
    features["days_stored"] = pd.to_numeric(df["days_stored"], errors="coerce").fillna(0.0)
    features["has_expiry_date"] = pd.to_numeric(df["has_expiry_date"], errors="coerce").fillna(0).astype(int)
    features["days_to_expiry"] = (pd.to_numeric(df["days_to_expiry"], errors="coerce")
                                  .fillna(NO_EXPIRY_PLACEHOLDER))
    for value in CATEGORIES:
        features[f"category_{value}"] = (df["food_category"] == value).astype(int)
    for value in LOCATIONS:
        features[f"location_{value}"] = (df["storage_location"] == value).astype(int)
    for value in CONDITIONS:
        features[f"condition_{value}"] = (df["food_condition"] == value).astype(int)
    return features


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    """Read the demo dataset; create it first if it does not exist."""
    if not Path(path).exists():
        save_dataset(Path(path))
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        raise RuntimeError(f"Could not read the dataset at {path}: {exc}") from exc


def train_and_save(model_path: Path = MODEL_PATH, max_depth: int = 12) -> dict:
    """Train the Decision Tree, evaluate it and save everything to a .pkl file."""
    df = load_dataset()
    X = build_features(df)
    y = df[TARGET_COLUMN]

    # 1. Split: 80% for learning, 20% kept hidden for testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # 2. Train
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    # 3. Evaluate on the unseen test set (plus a "always guess the most common" baseline)
    predictions = model.predict(X_test)
    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    labels = list(model.classes_)
    bundle = {
        "model": model,
        "feature_columns": list(X.columns),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "baseline_accuracy": float(accuracy_score(y_test, baseline.predict(X_test))),
        "report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
        "report_text": classification_report(y_test, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
        "labels": labels,
        "feature_importances": dict(zip(X.columns, model.feature_importances_.tolist())),
        "tree_text": export_text(model, feature_names=list(X.columns), max_depth=3),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "max_depth": max_depth,
    }

    # 4. Save the model (the models folder is created automatically)
    try:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as file:
            pickle.dump(bundle, file)
        _save_tree_plot(model, list(X.columns), TREE_IMAGE_PATH)
    except OSError as exc:
        raise RuntimeError(f"Could not save the model: {exc}") from exc
    return bundle


def _save_tree_plot(model, feature_names, image_path: Path) -> None:
    """Save a picture of the top levels of the tree."""
    fig, ax = plt.subplots(figsize=(16, 7))
    plot_tree(model, max_depth=3, feature_names=feature_names,
              class_names=list(model.classes_), filled=True, rounded=True,
              fontsize=7, ax=ax)
    fig.savefig(image_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def load_model(model_path: Path = MODEL_PATH) -> dict | None:
    """Load the saved bundle. Returns None if the file is missing or unreadable."""
    if not Path(model_path).exists():
        return None
    try:
        with open(model_path, "rb") as file:
            return pickle.load(file)
    except Exception:  # damaged file or different scikit-learn version
        return None


def predict_status(bundle: dict, category: str, storage_location: str, condition: str,
                   days_stored: float, days_to_expiry: int | None) -> str:
    """Predict the reminder status for ONE example."""
    row = pd.DataFrame([{
        "food_category": category,
        "storage_location": storage_location,
        "food_condition": condition,
        "days_stored": days_stored,
        "has_expiry_date": 0 if days_to_expiry is None else 1,
        "days_to_expiry": np.nan if days_to_expiry is None else days_to_expiry,
    }])
    features = build_features(row)[bundle["feature_columns"]]
    return str(bundle["model"].predict(features)[0])


if __name__ == "__main__":
    result = train_and_save()
    print("=" * 60)
    print("DECISION TREE DEMONSTRATION (synthetic data)")
    print("=" * 60)
    print(f"Training rows : {result['n_train']}")
    print(f"Test rows     : {result['n_test']}")
    print(f"Baseline accuracy (always guess most common): {result['baseline_accuracy']:.2%}")
    print(f"Decision Tree accuracy on test data         : {result['accuracy']:.2%}")
    print("\nClassification report:\n")
    print(result["report_text"])
    example = predict_status(result, "Dairy", "Refrigerator", "Opened", 6.5, 1)
    print("Prediction example: Dairy, Refrigerator, Opened, 6.5 days stored, 1 day to date")
    print(f"  -> predicted reminder status: {example}")
    print("\nNOTE: the labels came from our own rules, so this model only imitates them.")
    print("It cannot tell whether real food is fresh or safe.")
    print(f"\nModel saved to: {MODEL_PATH}")
