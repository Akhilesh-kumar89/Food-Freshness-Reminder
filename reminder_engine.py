
from __future__ import annotations

from datetime import date, datetime, time, timedelta

import pandas as pd

# ---------------------------------------------------------------------------
# 1. Allowed values (used by the form, the database and the ML demo)
# ---------------------------------------------------------------------------
CATEGORIES = [
    "Dairy",
    "Meat & Poultry",
    "Seafood",
    "Cooked Leftovers",
    "Eggs",
    "Fruits",
    "Vegetables",
    "Bakery",
    "Beverages",
    "Packaged / Pantry",
    "Other",
]
LOCATIONS = ["Refrigerator", "Freezer", "Room temperature"]

DEFAULT_CONDITION = "Fresh / Normal"
CONDITION_THAWED = "Thawed"
CONDITION_UNUSUAL = "Looks or smells unusual"
CONDITIONS = [
    DEFAULT_CONDITION,
    "Opened",
    "Cooked / Prepared",
    CONDITION_THAWED,
    CONDITION_UNUSUAL,
]

# ---------------------------------------------------------------------------
# 2. Reminder statuses
# ---------------------------------------------------------------------------
STATUS_NONE = "No Reminder Due"
STATUS_SOON = "Reminder Due Soon"
STATUS_CHECK = "Check Food"
STATUS_PASSED = "Date Passed"

STATUS_ORDER = [STATUS_PASSED, STATUS_CHECK, STATUS_SOON, STATUS_NONE]
PRIORITY = {STATUS_NONE: 1, STATUS_SOON: 2, STATUS_CHECK: 3, STATUS_PASSED: 4}
STATUS_ICONS = {
    STATUS_NONE: "🟢",
    STATUS_SOON: "🟡",
    STATUS_CHECK: "🟠",
    STATUS_PASSED: "🔴",
}
STATUS_COLORS = {
    STATUS_NONE: "#2E7D32",
    STATUS_SOON: "#F9A825",
    STATUS_CHECK: "#EF6C00",
    STATUS_PASSED: "#C62828",
}
# Statuses that need the user's attention now / soon
ATTENTION_STATUSES = {STATUS_CHECK, STATUS_PASSED}

# ---------------------------------------------------------------------------
# 3. Configured guidance (EXAMPLE values, in days). None = no guidance set.
# ---------------------------------------------------------------------------
FRIDGE_GUIDANCE_DAYS = {
    "Dairy": 7,
    "Meat & Poultry": 2,
    "Seafood": 2,
    "Cooked Leftovers": 4,
    "Eggs": 21,
    "Fruits": 7,
    "Vegetables": 7,
    "Bakery": 7,
    "Beverages": 7,
    "Packaged / Pantry": 7,   # applies after opening
    "Other": None,
}

FREEZER_GUIDANCE_DAYS = {
    "Dairy": 90,
    "Meat & Poultry": 180,
    "Seafood": 90,
    "Cooked Leftovers": 90,
    "Eggs": 180,
    "Fruits": 240,
    "Vegetables": 240,
    "Bakery": 90,
    "Beverages": 90,
    "Packaged / Pantry": 180,
    "Other": None,
}

ROOM_PERISHABLE = {"Dairy", "Meat & Poultry", "Seafood", "Cooked Leftovers", "Eggs"}
ROOM_PERISHABLE_DAYS = 2 / 24
ROOM_GUIDANCE_DAYS = {
    "Fruits": 5,
    "Vegetables": 4,
    "Bakery": 4,
    "Beverages": None,
    "Packaged / Pantry": None,
    "Other": None,
}
THAWED_FRIDGE_LIMIT_DAYS = 2        # thawed food has a shorter limit

DUE_SOON_FRACTION = 0.75            # remind when 75% of the guidance time is used
EXPIRY_WARNING_DAYS = 2             # remind when the expiry date is <= 2 days away

DISCLAIMER = (
    "Reminders are based on dates and general example guidance. They are NOT proof "
    "that food is safe or unsafe. When in doubt, follow the package label and "
    "official food-safety guidance (for example USDA FoodKeeper or FSSAI)."
)


# ---------------------------------------------------------------------------
# 4. Small helper functions
# ---------------------------------------------------------------------------
def parse_storage_datetime(value) -> datetime:
    """Convert a stored text value into a datetime. Raises ValueError if invalid."""
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unrecognised storage date: {value!r}")


def parse_expiry_date(value) -> date | None:
    """Convert stored text into a date. Missing values give None."""
    if value is None or (not isinstance(value, (str, date)) and pd.isna(value)):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    return datetime.strptime(text, "%Y-%m-%d").date()


def format_duration(hours: float) -> str:
    """Turn hours into friendly text such as '3 days 4 h' or '45 min'."""
    if hours < 1:
        return f"{int(round(hours * 60))} min"
    if hours < 24:
        return f"{int(hours)} h"
    days = int(hours // 24)
    rest = int(round(hours - days * 24))
    unit = "day" if days == 1 else "days"
    return f"{days} {unit} {rest} h" if rest else f"{days} {unit}"


def _format_limit(limit_days: float) -> str:
    if limit_days < 1:
        return f"{int(round(limit_days * 24))} hours"
    n = int(limit_days)
    return f"{n} day" if n == 1 else f"{n} days"


def _location_phrase(storage_location: str) -> str:
    return {"Refrigerator": "in the refrigerator", "Freezer": "in the freezer",
            "Room temperature": "at room temperature"}.get(storage_location, storage_location)


def get_guidance_days(category: str, storage_location: str,
                      condition: str = DEFAULT_CONDITION) -> float | None:
    """Return the configured guidance (in days) or None when none is configured."""
    if storage_location == "Refrigerator":
        limit = FRIDGE_GUIDANCE_DAYS.get(category)
        if limit is not None and condition == CONDITION_THAWED:
            limit = min(limit, THAWED_FRIDGE_LIMIT_DAYS)
        return limit
    if storage_location == "Freezer":
        return FREEZER_GUIDANCE_DAYS.get(category)
    if storage_location == "Room temperature":
        if category in ROOM_PERISHABLE:
            return ROOM_PERISHABLE_DAYS
        return ROOM_GUIDANCE_DAYS.get(category)
    return None


# ---------------------------------------------------------------------------
# 5. The main rule function (used by the app AND by the ML demo dataset)
# ---------------------------------------------------------------------------
def evaluate_reminder(category: str, storage_location: str, storage_dt: datetime,
                      expiry_date: date | None = None,
                      condition: str = DEFAULT_CONDITION,
                      now: datetime | None = None) -> dict:
    """Apply the transparent rules to ONE food item and return a result dictionary."""
    now = now or datetime.now()
    # Never allow a future storage time to create negative storage durations.
    elapsed_hours = max((now - storage_dt).total_seconds() / 3600.0, 0.0)
    days_stored = elapsed_hours / 24.0

    findings: list[tuple[int, str]] = []   # (priority, message)
    notes: list[str] = []                  # messages about missing information

    # Rule 1: the user says the food looks or smells unusual
    if condition == CONDITION_UNUSUAL:
        findings.append((PRIORITY[STATUS_CHECK],
                         "You marked this food as looking or smelling unusual. Do not taste it "
                         "to test it. Follow food-safety guidance; when in doubt, throw it out."))

    # Rule 2: the expiry date entered by the user
    days_to_expiry = None
    expiry_check_dt = None
    if expiry_date is None:
        notes.append("No expiry date was entered, so only storage-time guidance is used.")
    else:
        days_to_expiry = (expiry_date - now.date()).days
        expiry_check_dt = datetime.combine(expiry_date, time(23, 59))
        if days_to_expiry < 0:
            findings.append((PRIORITY[STATUS_PASSED],
                             f"The date on the package passed {abs(days_to_expiry)} day(s) ago. "
                             "A date alone does not prove food is safe or unsafe, so check the "
                             "label (use-by vs best-before) and follow food-safety guidance."))
        elif days_to_expiry <= EXPIRY_WARNING_DAYS:
            when = "today" if days_to_expiry == 0 else f"in {days_to_expiry} day(s)"
            findings.append((PRIORITY[STATUS_SOON],
                             f"The date on the package is {when}. Plan to check this food."))

    # Rule 3: configured storage-time guidance
    limit_days = get_guidance_days(category, storage_location, condition)
    guidance_check_dt = None
    if limit_days is None:
        notes.append("No storage-time guidance is configured for this category and location. "
                     "Check the package label or official food-safety guidance.")
    else:
        limit_hours = limit_days * 24.0
        guidance_check_dt = storage_dt + timedelta(hours=limit_hours)
        stored_text = format_duration(elapsed_hours)
        limit_text = _format_limit(limit_days)
        if elapsed_hours >= limit_hours:
            msg = (f"Stored for {stored_text}. This reaches the example guidance of {limit_text} "
                   f"for {category.lower()} kept {_location_phrase(storage_location)}. "
                   "Check the food and follow food-safety guidance.")
            if storage_location == "Freezer":
                msg += " Food kept constantly frozen stays safe; this reminder is about quality."
            if storage_location == "Room temperature" and category in ROOM_PERISHABLE:
                msg += " Perishable food should not stay at room temperature for more than about 2 hours."
            findings.append((PRIORITY[STATUS_CHECK], msg))
        elif elapsed_hours >= DUE_SOON_FRACTION * limit_hours:
            findings.append((PRIORITY[STATUS_SOON],
                             f"Stored for {stored_text}; the example guidance is {limit_text}. "
                             "A check will be due soon."))
        if condition == CONDITION_THAWED and storage_location == "Refrigerator":
            notes.append("Thawed food uses a shorter example limit.")

    # Decide the final status = the most serious finding
    if findings:
        findings.sort(key=lambda item: -item[0])
        top_priority = findings[0][0]
        status = next(s for s, p in PRIORITY.items() if p == top_priority)
        messages = [text for _, text in findings]
    else:
        status = STATUS_NONE
        messages = ["No reminder is due yet based on the dates and guidance available."]
    messages.extend(notes)

    # "Check by" = the earliest of the expiry date and the guidance time
    check_by_text = "Not available"
    options = [dt for dt in (guidance_check_dt, expiry_check_dt) if dt is not None]
    if options:
        earliest = min(options)
        if earliest is guidance_check_dt and limit_days is not None and limit_days < 1:
            check_by_text = earliest.strftime("%d %b %Y, %I:%M %p")
        else:
            check_by_text = earliest.strftime("%d %b %Y")

    return {
        "reminder_status": status,
        "status_icon": STATUS_ICONS[status],
        "priority": PRIORITY[status],
        "days_stored": round(days_stored, 1),
        "time_stored": format_duration(elapsed_hours),
        "days_to_expiry": days_to_expiry,
        "check_by": check_by_text,
        "guidance_days": limit_days,
        "reminder_message": " ".join(messages),
    }


_RESULT_COLUMNS = ["reminder_status", "status_icon", "priority", "days_stored",
                   "time_stored", "days_to_expiry", "check_by", "guidance_days",
                   "reminder_message"]


def apply_reminders(df: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    """Add reminder columns to a DataFrame of food records."""
    now = now or datetime.now()
    if df.empty:
        out = df.copy()
        for col in _RESULT_COLUMNS:
            out[col] = pd.Series(dtype="object")
        return out

    results = []
    for _, row in df.iterrows():
        try:
            result = evaluate_reminder(
                category=row["category"],
                storage_location=row["storage_location"],
                storage_dt=parse_storage_datetime(row["storage_date"]),
                expiry_date=parse_expiry_date(row["expiry_date"]),
                condition=row.get("food_condition", DEFAULT_CONDITION),
                now=now,
            )
        except (ValueError, TypeError, KeyError):
            # A damaged record should not crash the whole app.
            result = {
                "reminder_status": STATUS_CHECK, "status_icon": STATUS_ICONS[STATUS_CHECK],
                "priority": PRIORITY[STATUS_CHECK], "days_stored": None,
                "time_stored": "Unknown", "days_to_expiry": None,
                "check_by": "Not available", "guidance_days": None,
                "reminder_message": "The dates of this record could not be read. "
                                    "Please edit the record and check the food manually.",
            }
        results.append(result)
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)


# ---------------------------------------------------------------------------
# 6. Guidance table and waste-reduction tips
# ---------------------------------------------------------------------------
def guidance_table() -> pd.DataFrame:
    """A readable table of the configured guidance (shown in the app)."""
    def show(days):
        return "Not configured" if days is None else _format_limit(days)

    rows = []
    for category in CATEGORIES:
        rows.append({
            "Category": category,
            "Refrigerator": show(get_guidance_days(category, "Refrigerator")),
            "Freezer (quality)": show(get_guidance_days(category, "Freezer")),
            "Room temperature": show(get_guidance_days(category, "Room temperature")),
        })
    return pd.DataFrame(rows)


GENERAL_TIPS = [
    ("🗂️ Organise by storage date", "Keep older items at the front (first in, first out) so they are seen first."),
    ("🏷️ Label containers", "Write the food name and storage date on every container or bag."),
    ("🥡 Use suitable containers", "Airtight, clean containers help keep food from drying out or picking up smells."),
    ("🍽️ Plan meals around reminders", "Check the Reminders page before shopping or cooking and plan meals around items that need attention."),
    ("🧊 Freeze early", "If you know you will not use food in time, freeze it while it is still fresh, in portion-sized packs."),
    ("📚 Follow reliable guidance", "Use the package label and official sources such as USDA FoodKeeper or FSSAI for storage advice."),
    ("🛒 Buy only what you need", "A short weekly plan and shopping list reduces over-buying."),
]

SAFETY_TIP = ("Reducing waste must never mean eating questionable food. If food looks, smells "
              "or feels unusual, or you are not sure how long it was stored, follow food-safety "
              "guidance instead of risking illness.")


def get_contextual_tips(df_with_reminders: pd.DataFrame) -> list[str]:
    """Short tips based on the user's current inventory."""
    tips = []
    if df_with_reminders.empty:
        return ["Add your first food item to receive personalised reminders."]
    statuses = df_with_reminders["reminder_status"]
    if (statuses == STATUS_PASSED).any() or (statuses == STATUS_CHECK).any():
        tips.append("Some items need checking. Review them on the Reminders page and follow "
                    "food-safety guidance for each one.")
    if (statuses == STATUS_SOON).any():
        tips.append("Some reminders are coming up. If those items have been stored properly and "
                    "show no signs of spoilage, you can plan meals around them.")
    if df_with_reminders["expiry_date"].isna().sum() > 0:
        tips.append("Some items have no expiry date. Add one from the package label for better reminders.")
    if not tips:
        tips.append("Everything looks organised. Keep labelling containers with storage dates.")
    return tips
