

from __future__ import annotations

import html
from datetime import date, datetime, time, timedelta

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import database as db
import ml_model
import reminder_engine as engine

# ---------------------------------------------------------------------------
# Page setup and styling
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Food Freshness Reminder", page_icon="🥗", layout="wide")

CUSTOM_CSS = """
<style>
.block-container {padding-top: 1.6rem; max-width: 1200px;}
.app-banner {
    background: linear-gradient(120deg, #1B5E20 0%, #43A047 60%, #81C784 100%);
    color: #ffffff; padding: 1.2rem 1.6rem; border-radius: 16px; margin-bottom: 1.2rem;
    box-shadow: 0 4px 14px rgba(27, 94, 32, 0.25);
}
.app-banner h1 {margin: 0; font-size: 1.9rem; color: #ffffff; padding: 0;}
.app-banner p {margin: 0.3rem 0 0 0; opacity: 0.95; font-size: 1rem;}
.reminder-card {
    background: rgba(128, 128, 128, 0.07); border-left: 6px solid #2E7D32;
    border-radius: 10px; padding: 0.8rem 1rem; margin-bottom: 0.7rem;
}
.rc-title {font-size: 1.05rem; font-weight: 700;}
.rc-meta {font-size: 0.85rem; opacity: 0.75; margin: 0.15rem 0 0.35rem 0;}
.rc-msg {font-size: 0.92rem;}
.badge {
    color: #ffffff; padding: 0.1rem 0.6rem; border-radius: 999px; font-size: 0.75rem;
    font-weight: 600; margin-left: 0.5rem; vertical-align: middle; white-space: nowrap;
}
.tip-card {
    background: rgba(46, 125, 50, 0.08); border: 1px solid rgba(46, 125, 50, 0.25);
    border-radius: 12px; padding: 0.8rem 1rem; margin-bottom: 0.7rem; min-height: 5.2rem;
}
.tip-card b {display: block; margin-bottom: 0.2rem;}
div[data-testid="stMetric"] {
    background: rgba(46, 125, 50, 0.08); border: 1px solid rgba(46, 125, 50, 0.22);
    padding: 0.8rem 1rem; border-radius: 12px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PAGES = [
    "🏠 Home Dashboard",
    "➕ Add Food",
    "📋 Food Inventory",
    "⏰ Reminders",
    "🤖 ML Demonstration",
    "💡 Food Waste Reduction Tips",
]
DISPLAY_DATETIME = "%d %b %Y, %I:%M %p"
DISPLAY_DATE = "%d %b %Y"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def show_banner(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="app-banner"><h1>{html.escape(title)}</h1>'
                f'<p>{html.escape(subtitle)}</p></div>', unsafe_allow_html=True)


def flash(kind: str, message: str) -> None:
    """Remember a message so it can still be shown after the page reruns."""
    st.session_state["flash"] = (kind, message)


def show_flash() -> None:
    item = st.session_state.pop("flash", None)
    if item:
        kind, message = item
        getattr(st, kind)(message)


def show_table(data) -> None:
    """Show a table full-width (works on older and newer Streamlit versions)."""
    try:
        st.dataframe(data, width="stretch", hide_index=True)
    except Exception:
        st.dataframe(data, use_container_width=True, hide_index=True)


def load_inventory() -> pd.DataFrame:
    """Read all food records and add reminder columns. Stops the page on DB errors."""
    try:
        return engine.apply_reminders(db.get_all_foods())
    except db.DatabaseError as exc:
        st.error(f"Database problem: {exc}")
        st.stop()


def show_validation_errors(exc: ValueError) -> None:
    for message in str(exc).split("\n"):
        st.error(message)


def render_reminder_card(row: pd.Series) -> None:
    status = row["reminder_status"]
    color = engine.STATUS_COLORS[status]
    expiry = row["expiry_date"]
    expiry_text = "no expiry date" if pd.isna(expiry) else f"date on package {expiry}"
    meta = (f'{row["category"]} · {row["storage_location"]} · stored {row["time_stored"]} · '
            f'check by {row["check_by"]} · {expiry_text}')
    card = (f'<div class="reminder-card" style="border-left-color:{color}">'
            f'<div class="rc-title">{row["status_icon"]} {html.escape(str(row["food_name"]))}'
            f'<span class="badge" style="background:{color}">{html.escape(status)}</span></div>'
            f'<div class="rc-meta">{html.escape(meta)}</div>'
            f'<div class="rc-msg">{html.escape(str(row["reminder_message"]))}</div></div>')
    st.markdown(card, unsafe_allow_html=True)


def category_chart(df: pd.DataFrame):
    """Horizontal bar chart: number of items per category."""
    counts = df["category"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(6, max(2.6, 0.45 * len(counts) + 1)))
    bars = ax.barh(counts.index, counts.values, color="#43A047")
    ax.bar_label(bars, padding=3)
    ax.set_xlabel("Number of items")
    ax.set_title("Food inventory by category", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, max(counts.values) + 1)
    fig.tight_layout()
    return fig


def format_display_table(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare a friendly table with readable dates and clear column names."""
    out = pd.DataFrame({
        "ID": df["id"],
        "Food": df["food_name"],
        "Category": df["category"],
        "Location": df["storage_location"],
        "Condition": df["food_condition"],
        "Storage date": df["storage_date"].map(
            lambda v: engine.parse_storage_datetime(v).strftime(DISPLAY_DATETIME)),
        "Expiry date": df["expiry_date"].map(
            lambda v: "Not provided" if pd.isna(v)
            else engine.parse_expiry_date(v).strftime(DISPLAY_DATE)),
        "Days stored": df["days_stored"],
        "Check by": df["check_by"],
        "Status": df["status_icon"] + " " + df["reminder_status"],
    })
    return out


def color_status_cell(value: str) -> str:
    for status, color in engine.STATUS_COLORS.items():
        if str(value).endswith(status):
            return f"background-color: {color}22; color: {color}; font-weight: 600;"
    return ""


def style_table(table: pd.DataFrame):
    styler = table.style
    apply_map = getattr(styler, "map", None) or styler.applymap
    return apply_map(color_status_cell, subset=["Status"])


def add_demo_items() -> int:
    """Insert example items (with different ages) so the app can be demonstrated quickly."""
    today = date.today()

    def ago(days: int, hour: int) -> datetime:
        return datetime.combine(today - timedelta(days=days), time(hour, 0))

    samples = [
        ("Milk", "Dairy", "Refrigerator", ago(6, 9), today + timedelta(days=2), "Opened"),
        ("Cooked Rice", "Cooked Leftovers", "Refrigerator", ago(5, 20), None, "Cooked / Prepared"),
        ("Spinach", "Vegetables", "Refrigerator", ago(2, 9), None, "Fresh / Normal"),
        ("Chicken Curry", "Cooked Leftovers", "Freezer", ago(20, 19), None, "Cooked / Prepared"),
        ("Yogurt", "Dairy", "Refrigerator", ago(4, 9), today - timedelta(days=1), "Opened"),
        ("Bread Loaf", "Bakery", "Room temperature", ago(2, 9), today + timedelta(days=3), "Fresh / Normal"),
        ("Paneer", "Dairy", "Room temperature", ago(1, 20), None, "Opened"),
        ("Apples", "Fruits", "Refrigerator", ago(3, 9), None, "Fresh / Normal"),
    ]
    added = 0
    for name, category, location, stored, expiry, condition in samples:
        try:
            db.add_food(name, category, location, stored, expiry, condition, "Demo item")
            added += 1
        except ValueError:
            pass  # already added earlier
    return added


# ---------------------------------------------------------------------------
# Page 1: Home Dashboard
# ---------------------------------------------------------------------------
def page_dashboard() -> None:
    show_banner("🥗 Food Freshness Reminder",
                "Smart food waste reduction: track what you store and get timely reminders.")
    show_flash()
    df = load_inventory()

    if df.empty:
        st.info("No food items yet. Add your first item from the **Add Food** page, "
                "or load a few demo items to explore the app.")
        if st.button("Load demo items"):
            flash("success", f"Added {add_demo_items()} demo items.")
            st.rerun()
        return

    attention = df[df["reminder_status"].isin(engine.ATTENTION_STATUSES)]
    upcoming = df[df["reminder_status"] == engine.STATUS_SOON]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total food items", len(df))
    c2.metric("Items needing attention", len(attention))
    c3.metric("Upcoming reminders", len(upcoming))
    c4.metric("Categories in use", df["category"].nunique())

    st.caption(engine.DISCLAIMER)
    left, right = st.columns([1, 1])
    with left:
        fig = category_chart(df)
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.subheader("Needs attention")
        if attention.empty:
            st.success("No items need checking right now.")
        else:
            for _, row in attention.sort_values("priority", ascending=False).head(4).iterrows():
                render_reminder_card(row)
            if len(attention) > 4:
                st.caption(f"…and {len(attention) - 4} more on the Reminders page.")

    st.subheader("Upcoming reminder dates")
    if upcoming.empty:
        st.write("No reminders are due soon.")
    else:
        table = format_display_table(upcoming)[["Food", "Category", "Location", "Check by", "Status"]]
        show_table(style_table(table))


# ---------------------------------------------------------------------------
# Page 2: Add Food
# ---------------------------------------------------------------------------
def page_add_food() -> None:
    show_banner("➕ Add Food", "Enter what you stored, where, and when.")
    show_flash()
    with st.form("add_food_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        name = c1.text_input("Food name *", max_chars=100, placeholder="e.g. Cooked rice")
        category = c2.selectbox("Food category *", engine.CATEGORIES)
        c3, c4 = st.columns(2)
        location = c3.selectbox("Storage location *", engine.LOCATIONS)
        condition = c4.selectbox("Food condition *", engine.CONDITIONS)
        c5, c6, c7 = st.columns(3)
        storage_day = c5.date_input("Storage date *", value=date.today(), format="DD/MM/YYYY")
        storage_time = c6.time_input("Storage time *", step=300)
        expiry = c7.date_input("Expiry date (optional)", value=None, format="DD/MM/YYYY",
                               help="Use the date printed on the package, if there is one.")
        notes = st.text_area("Notes (optional)", max_chars=300,
                             placeholder="e.g. Opened on Sunday, kept in a sealed box")
        submitted = st.form_submit_button("Save food item", type="primary")

    if submitted:
        storage_dt = datetime.combine(storage_day, storage_time)
        try:
            db.add_food(name, category, location, storage_dt, expiry, condition, notes)
        except ValueError as exc:
            show_validation_errors(exc)
        except db.DatabaseError as exc:
            st.error(f"Database problem: {exc}")
        else:
            result = engine.evaluate_reminder(category, location, storage_dt, expiry, condition)
            st.success(f"Saved '{db.clean_name(name)}'.")
            st.markdown(f"**Reminder right now:** {result['status_icon']} {result['reminder_status']}")
            st.caption(result["reminder_message"])

    st.caption("Fields marked * are required. Storage time cannot be in the future.")


# ---------------------------------------------------------------------------
# Page 3: Food Inventory (search, filter, edit, delete)
# ---------------------------------------------------------------------------
def render_edit_section(df: pd.DataFrame) -> None:
    labels = {int(r["id"]): f'ID {int(r["id"])} – {r["food_name"]} ({r["category"]})'
              for _, r in df.iterrows()}
    with st.expander("✏️ Edit a record"):
        food_id = st.selectbox("Choose a record to edit", list(labels), format_func=labels.get,
                               key="edit_select")
        record = db.get_food(food_id)
        if record is None:
            st.warning("That record no longer exists.")
            return
        stored = engine.parse_storage_datetime(record["storage_date"])
        old_expiry = engine.parse_expiry_date(record["expiry_date"])
        with st.form(f"edit_form_{food_id}"):
            c1, c2 = st.columns(2)
            name = c1.text_input("Food name *", value=record["food_name"])
            category = c2.selectbox("Category *", engine.CATEGORIES,
                                    index=engine.CATEGORIES.index(record["category"]))
            c3, c4 = st.columns(2)
            location = c3.selectbox("Storage location *", engine.LOCATIONS,
                                    index=engine.LOCATIONS.index(record["storage_location"]))
            condition = c4.selectbox("Condition *", engine.CONDITIONS,
                                     index=engine.CONDITIONS.index(record["food_condition"]))
            c5, c6, c7 = st.columns(3)
            day = c5.date_input("Storage date *", value=stored.date(), format="DD/MM/YYYY")
            clock = c6.time_input("Storage time *", value=stored.time(), step=300)
            expiry = c7.date_input("Expiry date (optional)", value=old_expiry, format="DD/MM/YYYY")
            notes = st.text_area("Notes", value=record["notes"] or "", max_chars=300)
            save = st.form_submit_button("Update record", type="primary")
        if save:
            try:
                changed = db.update_food(food_id, name, category, location,
                                         datetime.combine(day, clock), expiry, condition, notes)
            except ValueError as exc:
                show_validation_errors(exc)
            except db.DatabaseError as exc:
                st.error(f"Database problem: {exc}")
            else:
                flash("success" if changed else "warning",
                      "Record updated." if changed else "Nothing was updated.")
                st.rerun()


def render_delete_section(df: pd.DataFrame) -> None:
    labels = {int(r["id"]): f'ID {int(r["id"])} – {r["food_name"]} ({r["category"]})'
              for _, r in df.iterrows()}
    with st.expander("🗑️ Delete a record"):
        food_id = st.selectbox("Choose a record to delete", list(labels), format_func=labels.get,
                               key="delete_select")
        confirm = st.checkbox("Yes, I want to delete this record", key="delete_confirm")
        if st.button("Delete record", disabled=not confirm):
            try:
                deleted = db.delete_food(food_id)
            except db.DatabaseError as exc:
                st.error(f"Database problem: {exc}")
                return
            flash("success" if deleted else "warning",
                  "Record deleted." if deleted else "That record was already removed.")
            st.session_state["delete_confirm"] = False
            st.rerun()


def page_inventory() -> None:
    show_banner("📋 Food Inventory", "Search, filter, edit and delete your stored food.")
    show_flash()
    df = load_inventory()
    if df.empty:
        st.info("Your inventory is empty. Add food on the **Add Food** page.")
        return

    f1, f2, f3, f4 = st.columns([2, 1.3, 1.3, 1.3])
    search = f1.text_input("🔎 Search by food name", placeholder="e.g. milk")
    category = f2.selectbox("Category", ["All"] + engine.CATEGORIES)
    location = f3.selectbox("Storage location", ["All"] + engine.LOCATIONS)
    status = f4.selectbox("Reminder status", ["All"] + engine.STATUS_ORDER)
    only_attention = st.checkbox("Show only foods that need attention (Check Food / Date Passed)")

    filtered = df
    if search.strip():
        filtered = filtered[filtered["food_name"].str.contains(search.strip(), case=False, na=False, regex=False)]
    if category != "All":
        filtered = filtered[filtered["category"] == category]
    if location != "All":
        filtered = filtered[filtered["storage_location"] == location]
    if status != "All":
        filtered = filtered[filtered["reminder_status"] == status]
    if only_attention:
        filtered = filtered[filtered["reminder_status"].isin(engine.ATTENTION_STATUSES)]

    st.write(f"Showing **{len(filtered)}** of **{len(df)}** items")
    if filtered.empty:
        st.warning("No items match your search or filters.")
    else:
        show_table(style_table(format_display_table(filtered)))
        st.caption("Statuses are reminders based on dates and example guidance, not proof of food safety.")

    render_edit_section(df)
    render_delete_section(df)


# ---------------------------------------------------------------------------
# Page 4: Reminders
# ---------------------------------------------------------------------------
def page_reminders() -> None:
    show_banner("⏰ Reminders", "Date-based reminders to help you check food on time.")
    show_flash()
    df = load_inventory()
    st.warning(engine.DISCLAIMER)
    if df.empty:
        st.info("Add food items to see reminders.")
        return

    counts = df["reminder_status"].value_counts()
    cols = st.columns(4)
    for col, status in zip(cols, engine.STATUS_ORDER):
        col.metric(f"{engine.STATUS_ICONS[status]} {status}", int(counts.get(status, 0)))

    for status in engine.STATUS_ORDER:
        group = df[df["reminder_status"] == status]
        if group.empty:
            continue
        st.subheader(f"{engine.STATUS_ICONS[status]} {status} ({len(group)})")
        for _, row in group.iterrows():
            render_reminder_card(row)

    with st.expander("📖 View the configured guidance used for reminders"):
        st.caption("Example values for this educational project. Verify with official sources "
                   "such as USDA FoodKeeper or FSSAI. Frozen-food numbers are about quality, "
                   "not safety. Where it says 'Not configured', the app has no guidance and "
                   "tells you to check the label or official guidance.")
        show_table(engine.guidance_table())


# ---------------------------------------------------------------------------
# Page 5: ML Demonstration
# ---------------------------------------------------------------------------
def page_ml() -> None:
    show_banner("🤖 ML Demonstration", "A small Decision Tree classification example for learning.")
    st.warning(
        "**Educational demo only.** The data is **synthetic demonstration data**, and its labels "
        "(reminder statuses) were produced by this app's own rules. The model only learns to "
        "imitate those rules. It **cannot** tell whether real food is fresh, safe or unsafe. "
        "The real reminders in this app always come from transparent rules, not from this model.")

    with st.expander("What is happening here? (simple explanation)", expanded=True):
        st.markdown(
            "- **Task:** predict the reminder status (No Reminder Due / Reminder Due Soon / "
            "Check Food / Date Passed).\n"
            "- **Features (inputs):** category, storage location, condition, days stored, "
            "whether an expiry date exists, and days until that date.\n"
            "- **Target (output):** `reminder_status`.\n"
            "- **Train-test split:** 80% of rows teach the tree; 20% are hidden and used to test it.\n"
            "- **Decision Tree:** asks a series of yes/no questions (like *days stored > 5?*) "
            "until it reaches an answer.\n"
            "- **Why not use ML for real reminders?** Because the labels are not real-world "
            "truth. Transparent rules are easier to explain and verify.")

    bundle = ml_model.load_model()
    if bundle is None:
        st.error("The trained model file is missing or could not be read "
                 "(this can happen after a scikit-learn update).")
        if st.button("Train the model now", type="primary"):
            with st.spinner("Training…"):
                try:
                    ml_model.train_and_save()
                except RuntimeError as exc:
                    st.error(str(exc))
                    return
            st.rerun()
        st.caption("Or run `python ml_model.py` in the terminal.")
        return

    st.subheader("1. Demonstration dataset (first rows)")
    try:
        dataset = ml_model.load_dataset()
        show_table(dataset.head(10))
        st.caption(f"{len(dataset)} synthetic rows in data/food_demo_dataset.csv")
    except RuntimeError as exc:
        st.error(str(exc))

    st.subheader("2. Results on the hidden test data")
    m1, m2, m3 = st.columns(3)
    m1.metric("Decision Tree accuracy", f"{bundle['accuracy']:.1%}")
    m2.metric("Baseline (always guess most common)", f"{bundle['baseline_accuracy']:.1%}")
    m3.metric("Train / test rows", f"{bundle['n_train']} / {bundle['n_test']}")
    st.caption("High accuracy here only means the tree copies our rules well on synthetic data. "
               "It is not evidence about real food.")

    report = pd.DataFrame(bundle["report"]).T.round(2)
    report = report.drop(index=["accuracy"], errors="ignore")
    st.markdown("**Classification report** (precision = how often a prediction is right, "
                "recall = how many real cases were found)")
    st.dataframe(report)

    st.markdown("**Most important features**")
    importances = (pd.Series(bundle["feature_importances"]).sort_values(ascending=False).head(8)
                   .sort_values())
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.barh(importances.index, importances.values, color="#43A047")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Importance")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    if ml_model.TREE_IMAGE_PATH.exists():
        with st.expander("Picture of the top levels of the tree"):
            st.image(str(ml_model.TREE_IMAGE_PATH))

    st.subheader("3. Try a prediction")
    p1, p2, p3 = st.columns(3)
    category = p1.selectbox("Category", engine.CATEGORIES, key="ml_cat")
    location = p2.selectbox("Storage location", engine.LOCATIONS, key="ml_loc")
    condition = p3.selectbox("Condition", engine.CONDITIONS, key="ml_cond")
    p4, p5 = st.columns(2)
    days_stored = p4.number_input("Days stored", min_value=0.0, max_value=400.0, value=3.0, step=0.5)
    has_expiry = p5.checkbox("The package has a date", value=True)
    days_to_expiry = None
    if has_expiry:
        days_to_expiry = int(p5.number_input("Days until that date (negative = already passed)",
                                             min_value=-60, max_value=365, value=5, step=1))

    if st.button("Predict reminder status", type="primary"):
        predicted = ml_model.predict_status(bundle, category, location, condition,
                                            days_stored, days_to_expiry)
        now = datetime.now()
        rule_result = engine.evaluate_reminder(
            category, location, now - timedelta(days=days_stored),
            None if days_to_expiry is None else now.date() + timedelta(days=days_to_expiry),
            condition, now=now)
        r1, r2 = st.columns(2)
        r1.info(f"🤖 Decision Tree predicts: **{predicted}**")
        r2.success(f"📏 Transparent rules say: **{rule_result['reminder_status']}**")
        if predicted != rule_result["reminder_status"]:
            st.caption("They differ: the tree is an approximation. This is why the app uses the rules.")
        st.caption("Neither result says whether the food is safe. Follow food-safety guidance.")


# ---------------------------------------------------------------------------
# Page 6: Food Waste Reduction Tips
# ---------------------------------------------------------------------------
def page_tips() -> None:
    show_banner("💡 Food Waste Reduction Tips", "Simple habits that help you waste less food.")
    df = load_inventory()
    st.subheader("For your inventory")
    for tip in engine.get_contextual_tips(df):
        st.write(f"• {tip}")

    st.subheader("General habits")
    cols = st.columns(2)
    for index, (title, text) in enumerate(engine.GENERAL_TIPS):
        cols[index % 2].markdown(
            f'<div class="tip-card"><b>{html.escape(title)}</b>{html.escape(text)}</div>',
            unsafe_allow_html=True)

    st.warning(engine.SAFETY_TIP)


# ---------------------------------------------------------------------------
# Main: sidebar navigation
# ---------------------------------------------------------------------------
def main() -> None:
    try:
        db.init_db()
    except db.DatabaseError as exc:
        st.error(f"Could not start the database: {exc}")
        st.stop()

    st.sidebar.title("🥗 Food Freshness Reminder")
    page = st.sidebar.radio("Go to", PAGES, label_visibility="collapsed")
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Today: {date.today().strftime('%A, %d %B %Y')}")
    st.sidebar.caption("Reminders are not food-safety verdicts. When unsure, follow the package "
                       "label and official food-safety guidance.")

    {
        PAGES[0]: page_dashboard,
        PAGES[1]: page_add_food,
        PAGES[2]: page_inventory,
        PAGES[3]: page_reminders,
        PAGES[4]: page_ml,
        PAGES[5]: page_tips,
    }[page]()


main()
