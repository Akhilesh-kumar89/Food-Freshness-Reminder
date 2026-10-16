from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from reminder_engine import CATEGORIES, CONDITIONS, LOCATIONS, evaluate_reminder, get_guidance_days

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "food_demo_dataset.csv"

REFERENCE_NOW = datetime(2025, 1, 15, 12, 0)


def generate_demo_dataset(n_rows: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Build the synthetic demonstration dataset."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_rows):
        category = str(rng.choice(CATEGORIES))
        location = str(rng.choice(LOCATIONS))
        condition = str(rng.choice(CONDITIONS, p=[0.45, 0.20, 0.15, 0.12, 0.08]))

        # Sample a storage time around the configured limit (some below, some above)
        limit_days = get_guidance_days(category, location, condition)
        max_days = limit_days * 1.6 if limit_days is not None else 30.0
        days_stored = round(float(rng.uniform(0, max_days)), 3)

        has_expiry = bool(rng.random() < 0.6)
        days_to_expiry = int(rng.integers(-5, 16)) if has_expiry else None

        storage_dt = REFERENCE_NOW - timedelta(days=days_stored)
        expiry_date = (REFERENCE_NOW.date() + timedelta(days=days_to_expiry)) if has_expiry else None

        # The label comes from the transparent rule engine
        result = evaluate_reminder(category, location, storage_dt, expiry_date,
                                   condition, now=REFERENCE_NOW)
        rows.append({
            "food_category": category,
            "storage_location": location,
            "food_condition": condition,
            "days_stored": days_stored,
            "has_expiry_date": int(has_expiry),
            "days_to_expiry": days_to_expiry,
            "reminder_status": result["reminder_status"],
        })
    return pd.DataFrame(rows)


def save_dataset(path: Path = DATASET_PATH, n_rows: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Create the dataset and save it as CSV (the data folder is created automatically)."""
    df = generate_demo_dataset(n_rows, seed)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"Could not save the dataset to {path}: {exc}")
    return df


if __name__ == "__main__":
    data = save_dataset()
    print(f"Saved {len(data)} demonstration rows to: {DATASET_PATH}")
    print("\nRows per reminder status (target column):")
    print(data["reminder_status"].value_counts().to_string())
    print("\nREMINDER: this is SYNTHETIC demonstration data, not real food-safety data.")
