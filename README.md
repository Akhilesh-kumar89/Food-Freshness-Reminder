# 🥗 Food Freshness Reminder – Smart Food Waste Reduction System

A beginner-friendly Python project that helps you track food in your refrigerator, freezer or pantry,
shows **date-based reminders**, and demonstrates a small **Decision Tree** machine-learning example.

> ⚠️ **Important:** Reminders are NOT food-safety verdicts. The app never says food is "safe" or
> "unsafe" based on a date. When safety cannot be determined, follow the package label and official
> guidance (for example USDA FoodKeeper or FSSAI).

---

## 1. Problem Statement
Households throw away a lot of food because people forget what they stored and when. At the same time,
guessing about old food can be risky. People need a simple way to remember storage dates and to be
reminded to *check* food at sensible times.

## 2. Proposed Solution
A local web app that stores food records in SQLite, calculates how long each item has been stored, and
generates reminders from **transparent rules** (storage time, user-entered expiry date, configured example
guidance, and the user's own "looks or smells unusual" flag). A separate, clearly labelled **educational**
ML page shows a basic classification workflow.

## 3. Features
| Feature | Where |
|---|---|
| Add food (name, category, date + time, location, condition, expiry, notes) with validation | ➕ Add Food |
| Table with days stored, check-by date and reminder status | 📋 Food Inventory |
| Search by name; filter by category, location, status; "needs attention" filter | 📋 Food Inventory |
| Edit and delete records (delete asks for confirmation) | 📋 Food Inventory |
| Reminder statuses: **Reminder Due Soon, Check Food, Date Passed, No Reminder Due** | ⏰ Reminders |
| Clear messages when expiry date or guidance is missing | ⏰ Reminders |
| Dashboard: totals, items needing attention, upcoming reminders, category chart | 🏠 Home Dashboard |
| Educational Decision Tree with metrics and live prediction | 🤖 ML Demonstration |
| Waste-reduction tips (without encouraging risky eating) | 💡 Tips |
| "Load demo items" button for quick presentations | 🏠 Home Dashboard |

## 4. Technologies Used
Python · Streamlit · SQLite (`sqlite3`) · Pandas · NumPy · Scikit-learn (DecisionTreeClassifier) · Matplotlib

## 5. Machine Learning Methodology
**Purpose:** teach a basic classification workflow (data → features → split → train → evaluate → predict).

**Task:** predict the reminder status from category, location, condition, days stored, whether a package
date exists and days until that date.

**Data:** `data/food_demo_dataset.csv` – 2,000 **synthetic demonstration rows** made by `create_dataset.py`.

**Target:** `reminder_status`. The labels come from **our own rule engine**, *not* from real-world food-safety
measurements. So the tree learns to **imitate our rules**.

**Steps (in `ml_model.py`):**
1. Load the CSV. 2. Build features (numbers stay numbers; text becomes 0/1 columns).
3. Split 80% train / 20% test (`train_test_split`, stratified). 4. Train `DecisionTreeClassifier(max_depth=12)`.
5. Evaluate with accuracy, a "most-common-class" baseline and a classification report. 6. Save to `models/decision_tree_model.pkl`.

**Typical results:** about 91% test accuracy (baseline about 39%). Shallower trees score lower (e.g. depth 4 ≈ 75%),
deeper trees score higher on this synthetic data. You can try different `max_depth` values and explain
under-fitting vs over-fitting in your presentation.

**Limitations:** synthetic data + rule-made labels means the model **cannot** tell whether real food is fresh
or safe. The app's real reminders always come from `reminder_engine.py`.

## 6. Folder Structure and Purpose of Each File
```
food-freshness-reminder/
├── app.py                  Streamlit interface (6 pages, styling, navigation)
├── database.py             SQLite: create table, add / read / update / delete, validation
├── reminder_engine.py      Transparent reminder rules + configured example guidance + tips
├── ml_model.py             Train / evaluate / save / load / predict with the Decision Tree
├── create_dataset.py       Builds the synthetic demonstration dataset (CSV)
├── requirements.txt        Python packages to install
├── README.md               This documentation
├── .streamlit/config.toml  Green theme for the app
├── data/
│   ├── food_demo_dataset.csv     Demo dataset (synthetic)
│   └── food_inventory.db         Created automatically the first time you run the app
└── models/
    ├── decision_tree_model.pkl   Saved model (created by ml_model.py)
    └── decision_tree_plot.png    Picture of the tree (created by ml_model.py)
```

**How the files work together**
```
app.py ──► database.py          save / read / update / delete records (SQLite)
app.py ──► reminder_engine.py   dates + guidance ──► reminder status
app.py ──► ml_model.py          loads the saved Decision Tree for the ML page
ml_model.py ──► create_dataset.py ──► reminder_engine.py   (demo labels come from the rules)
```

**Important functions**
* `database.validate_food_input()` – checks name, category, location, condition, dates (no future storage time, expiry not before storage).
* `database.add_food / get_all_foods / update_food / delete_food` – the CRUD functions. Duplicates are blocked by a UNIQUE rule.
* `reminder_engine.evaluate_reminder()` – the heart of the project: applies the rules to one item.
* `reminder_engine.apply_reminders()` – runs the rules for every row of the inventory table.
* `ml_model.build_features()` – converts raw columns to numbers. `train_and_save()` trains and saves. `predict_status()` predicts one example.

**How reminders are decided (simple version)**
1. User marked "Looks or smells unusual" → **Check Food**.
2. Package date already passed → **Date Passed**; within 2 days → **Reminder Due Soon**.
3. Stored time ≥ configured guidance → **Check Food**; ≥ 75% of it → **Reminder Due Soon**.
4. The highest-priority result wins. Nothing triggered → **No Reminder Due**.
5. Missing expiry date or missing guidance → a clear message is shown.

The guidance numbers (for example "cooked leftovers: 4 days in the refrigerator") are **example values** in
`reminder_engine.py` and can be changed. Please verify them with official sources.

## 7. Installation (Windows + VS Code)

Open **VS Code → Terminal → New Terminal** (PowerShell or Command Prompt).

**Step 1 – Check Python (3.10 or newer recommended)**
```
python --version
```
If this fails, try `py --version`, or install Python from python.org and tick **"Add Python to PATH"**.

**Step 2 – Go to the project folder** (after unzipping it)
```
cd path\to\food-freshness-reminder
code .
```
(To start from scratch instead: `mkdir food-freshness-reminder`, `cd food-freshness-reminder`, and copy the files in.)

**Step 3 – Create a virtual environment**
```
python -m venv venv
```

**Step 4 – Activate it**
```
venv\Scripts\activate
```
If PowerShell blocks it, run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or use Command Prompt.
You should see `(venv)` at the start of the line.

**Step 5 – Install the packages**
```
pip install -r requirements.txt
```

## 8. Execution Order
```
python create_dataset.py     # creates data/food_demo_dataset.csv (already included, safe to re-run)
python ml_model.py           # trains and saves models/decision_tree_model.pkl (re-run it on your PC!)
streamlit run app.py         # opens the app in your browser (http://localhost:8501)
```
> Re-run `python ml_model.py` on your own computer so the saved model matches your scikit-learn version.
> The app also has a **"Train the model now"** button if the model file is missing or unreadable.

## 9. Sample Workflow
1. Open **Home Dashboard** → click **Load demo items** (8 sample foods with different ages).
2. See the counts, the chart and the "Needs attention" cards.
3. Open **Reminders** → read the status and message for each item.
4. Open **Add Food** → add "Cooked rice", Cooked Leftovers, Refrigerator, 5 days ago → status **Check Food**.
5. Open **Food Inventory** → search "rice", filter by category, edit or delete a record.
6. Open **ML Demonstration** → read the warning, see accuracy, try a prediction and compare it with the rules.
7. Close and restart the app → your records are still there (SQLite file).

## 10. Test Cases

| # | Test | Steps | Expected result |
|---|---|---|---|
| 1 | Add a new food item | Add Food → name "Milk", Dairy, Refrigerator, yesterday, expiry in 5 days → Save | Green "Saved" message with the current reminder; item appears in Inventory with Days stored ≈ 1 |
| 2 | Add without expiry date | Leave expiry empty → Save | Saved. Inventory shows "Not provided"; the reminder message says no expiry date was entered |
| 3 | Invalid date | (a) Expiry date earlier than storage date. (b) Type a non-existing date such as 31/02/2026 in a date box | (a) Red error "Expiry date is before the storage date". (b) Streamlit's date box refuses the invalid date and nothing is saved |
| 4 | Future storage date | Choose tomorrow as storage date (or today with a later time) | Red error "Storage date and time cannot be in the future"; nothing saved |
| 5 | View reminders | Load demo items → Reminders page | Items grouped by status (Date Passed, Check Food, Reminder Due Soon, No Reminder Due) with explanations and the warning banner |
| 6 | Search for food | Inventory → search "milk" (any capitalisation) | Only items whose name contains "milk" are listed; a no-match search shows "No items match…" |
| 7 | Filter by category | Inventory → Category = Dairy | Only Dairy items. Also try Location and Status filters and the "needs attention" checkbox |
| 8 | Delete a record | Inventory → Delete a record → choose item → tick confirm → Delete | Success message; the item disappears and dashboard totals update |
| 9 | Persistence | Stop the app (Ctrl+C), run `streamlit run app.py` again | All saved records are still there |
| 10 | Duplicate protection | Save exactly the same item (same name, category, location, date and time) twice | Second save is rejected with "already saved" |
| 11 | Empty / missing name | Leave name blank | Red error "Food name is required" |
| 12 | Missing model | Delete `models/decision_tree_model.pkl`, open ML page | Message with a "Train the model now" button; the rest of the app keeps working |

## 11. Limitations
* Guidance values are simple examples, not a complete or official food-safety database.
* Reminders cannot know real storage temperatures, hygiene or how the food was handled.
* The ML model uses synthetic data and rule-made labels; it only imitates the rules.
* Single-user, local app; no notifications are pushed (you must open the app).
* Time is the computer's local time; no time-zone handling.

## 12. Future Improvements
* Email / phone notifications, barcode scanning for package dates.
* A larger, editable guidance table (importable from official sources) and per-item storage temperature.
* Real data collected with expert-verified labels (before any real ML claims).
* User accounts, shopping list, recipe ideas for items that need attention (when appropriate).
* Unit tests with `pytest`.

## 13. Presentation Tips (Hinglish)
* **Pehle problem batao:** log khana store karke bhool jaate hain → waste hota hai.
* **Solution:** app dates track karta hai aur *check karne ka reminder* deta hai — "safe/unsafe" ka verdict nahi deta.
* **ML honestly explain karo:** "Yeh synthetic data hai aur labels hamare rules se bane hain; model sirf rules ko copy karna seekhta hai."
* Demo flow: demo items load karo → Reminders dikhao → ek item add karo → ML page par prediction vs rules compare karo.
