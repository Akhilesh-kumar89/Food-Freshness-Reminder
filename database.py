

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from reminder_engine import CATEGORIES, CONDITIONS, DEFAULT_CONDITION, LOCATIONS

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "food_inventory.db"

STORAGE_FORMAT = "%Y-%m-%d %H:%M"
EXPIRY_FORMAT = "%Y-%m-%d"

COLUMNS = ["id", "food_name", "category", "storage_location", "storage_date",
           "expiry_date", "food_condition", "notes", "created_at"]


class DatabaseError(Exception):
    """Raised when something goes wrong while talking to SQLite."""


def get_connection(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    """Open a connection (the data folder is created automatically)."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(db_path))


def init_db(db_path: Path | str = DB_PATH) -> None:
    """Create the table if it does not exist yet. Safe to call many times."""
    try:
        with closing(get_connection(db_path)) as conn, conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS food_items (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    food_name        TEXT NOT NULL,
                    category         TEXT NOT NULL,
                    storage_location TEXT NOT NULL,
                    storage_date     TEXT NOT NULL,
                    expiry_date      TEXT,
                    food_condition   TEXT NOT NULL DEFAULT 'Fresh / Normal',
                    notes            TEXT,
                    created_at       TEXT NOT NULL,
                    -- Stops the exact same item being saved twice
                    UNIQUE (food_name COLLATE NOCASE, category, storage_location, storage_date)
                )
                """
            )
    except sqlite3.Error as exc:
        raise DatabaseError(f"Could not create the database: {exc}") from exc


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def clean_name(name: str | None) -> str:
    """Trim spaces and collapse repeated spaces."""
    return " ".join((name or "").split())


def validate_food_input(food_name, category, storage_location, storage_dt,
                        expiry_date=None, food_condition=DEFAULT_CONDITION,
                        notes="", now: datetime | None = None) -> list[str]:
    """Return a list of error messages (an empty list means the input is valid)."""
    now = now or datetime.now()
    errors: list[str] = []

    name = clean_name(food_name)
    if not name:
        errors.append("Food name is required.")
    elif len(name) > 100:
        errors.append("Food name must be 100 characters or fewer.")
    elif not any(ch.isalnum() for ch in name):
        errors.append("Food name must contain letters or numbers.")

    if category not in CATEGORIES:
        errors.append("Please choose a valid food category.")
    if storage_location not in LOCATIONS:
        errors.append("Please choose a valid storage location.")
    if food_condition not in CONDITIONS:
        errors.append("Please choose a valid food condition.")

    if not isinstance(storage_dt, datetime):
        errors.append("Storage date and time are required and must be valid.")
    elif storage_dt > now:
        errors.append("Storage date and time cannot be in the future.")

    if expiry_date is not None:
        if not isinstance(expiry_date, date) or isinstance(expiry_date, datetime):
            errors.append("Expiry date is not a valid date.")
        elif isinstance(storage_dt, datetime) and expiry_date < storage_dt.date():
            errors.append("Expiry date is before the storage date. Please check both dates.")

    if notes and len(notes) > 300:
        errors.append("Notes must be 300 characters or fewer.")
    return errors


# ---------------------------------------------------------------------------
# Create / Read / Update / Delete
# ---------------------------------------------------------------------------
def _expiry_to_text(expiry_date: date | None) -> str | None:
    return expiry_date.strftime(EXPIRY_FORMAT) if expiry_date else None


def add_food(food_name, category, storage_location, storage_dt, expiry_date=None,
             food_condition=DEFAULT_CONDITION, notes="",
             db_path: Path | str = DB_PATH, now: datetime | None = None) -> int:
    """Validate and save a new item. Returns the new id.

    Raises ValueError for invalid input / duplicates and DatabaseError for SQLite problems.
    """
    errors = validate_food_input(food_name, category, storage_location, storage_dt,
                                 expiry_date, food_condition, notes, now)
    if errors:
        raise ValueError("\n".join(errors))
    init_db(db_path)
    try:
        with closing(get_connection(db_path)) as conn, conn:
            cursor = conn.execute(
                """INSERT INTO food_items
                   (food_name, category, storage_location, storage_date, expiry_date,
                    food_condition, notes, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (clean_name(food_name), category, storage_location,
                 storage_dt.strftime(STORAGE_FORMAT), _expiry_to_text(expiry_date),
                 food_condition, (notes or "").strip(),
                 datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            )
            return int(cursor.lastrowid)
    except sqlite3.IntegrityError as exc:
        raise ValueError("This exact food item (same name, category, location and "
                         "storage time) is already saved.") from exc
    except sqlite3.Error as exc:
        raise DatabaseError(f"Could not save the food item: {exc}") from exc


def get_all_foods(db_path: Path | str = DB_PATH) -> pd.DataFrame:
    """Return every record as a DataFrame (empty DataFrame with columns if none)."""
    init_db(db_path)
    try:
        with closing(get_connection(db_path)) as conn:
            df = pd.read_sql_query(
                f"SELECT {', '.join(COLUMNS)} FROM food_items ORDER BY storage_date DESC, id DESC",
                conn,
            )
    except (sqlite3.Error, pd.errors.DatabaseError) as exc:
        raise DatabaseError(f"Could not read food records: {exc}") from exc
    return df


def get_food(food_id: int, db_path: Path | str = DB_PATH) -> dict | None:
    """Return one record as a dictionary, or None when it does not exist."""
    init_db(db_path)
    try:
        with closing(get_connection(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                f"SELECT {', '.join(COLUMNS)} FROM food_items WHERE id = ?", (int(food_id),)
            ).fetchone()
    except sqlite3.Error as exc:
        raise DatabaseError(f"Could not read the food record: {exc}") from exc
    return dict(row) if row else None


def update_food(food_id, food_name, category, storage_location, storage_dt,
                expiry_date=None, food_condition=DEFAULT_CONDITION, notes="",
                db_path: Path | str = DB_PATH, now: datetime | None = None) -> bool:
    """Validate and update an item. Returns True if a record was changed."""
    errors = validate_food_input(food_name, category, storage_location, storage_dt,
                                 expiry_date, food_condition, notes, now)
    if errors:
        raise ValueError("\n".join(errors))
    init_db(db_path)
    try:
        with closing(get_connection(db_path)) as conn, conn:
            cursor = conn.execute(
                """UPDATE food_items
                   SET food_name = ?, category = ?, storage_location = ?, storage_date = ?,
                       expiry_date = ?, food_condition = ?, notes = ?
                   WHERE id = ?""",
                (clean_name(food_name), category, storage_location,
                 storage_dt.strftime(STORAGE_FORMAT), _expiry_to_text(expiry_date),
                 food_condition, (notes or "").strip(), int(food_id)),
            )
            return cursor.rowcount > 0
    except sqlite3.IntegrityError as exc:
        raise ValueError("Another record with the same name, category, location and "
                         "storage time already exists.") from exc
    except sqlite3.Error as exc:
        raise DatabaseError(f"Could not update the food item: {exc}") from exc


def delete_food(food_id, db_path: Path | str = DB_PATH) -> bool:
    """Delete one record. Returns True if something was deleted."""
    init_db(db_path)
    try:
        with closing(get_connection(db_path)) as conn, conn:
            cursor = conn.execute("DELETE FROM food_items WHERE id = ?", (int(food_id),))
            return cursor.rowcount > 0
    except sqlite3.Error as exc:
        raise DatabaseError(f"Could not delete the food item: {exc}") from exc
