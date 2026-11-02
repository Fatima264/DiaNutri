# DiaNutri AI Assistant

DiaNutri is a small content-based restaurant recommendation project for
diabetic users. It cleans the supplied Lebanon restaurant CSV, asks five
questions in a chat-style Streamlit page, applies hard safety filters, and uses
K-nearest neighbors (KNN) to rank the remaining meals.

> Important: the project uses claims written in the dataset. It does not prove
> that a meal has zero sugar or that its ingredient list is complete. A real
> health product must confirm information with each restaurant and a qualified
> clinician or dietitian.

## Features

- **Chat history**: Every conversation is saved automatically. Click **New Search**
  to start a fresh chat (like ChatGPT/Manus), or click any saved chat in the
  sidebar to resume it.
- **Diabetic level question**: The first question asks for the user's diabetic
  level and tailors results and advice accordingly.
- **All areas option**: Choose "All" in the area question to see results from
  every area in the dataset.
- **Allergy flexibility**: Choose "None" for no allergies, or select specific
  allergens and add custom ones.

## 1. What was found in the original dataset

- 177 meal rows and 51 restaurants.
- 16 original columns.
- No empty cells and no exact duplicate rows.
- Four rows contained `Beirut ` with a trailing space. Cleaning changes all of
	  them to `Beirut`.
- The dataset has no ready-made columns for meal time, food preference, allergy
  flags, or measured sugar grams.
- Only 31 of the 177 rows contain an explicit textual claim such as
  `sugar-free`, `sugar-free option`, `no added sugar`, `no white sugar`, or
  `no refined sugar`. The app filters to these 31 before doing any ranking.
- `Certified Diabetic` contains 10 `Yes` rows and 167 `No` rows. It is displayed
	  to the user but is not treated as a medical guarantee.
- The dataset includes a `Diabetic Level` column with four values:
  `Pre-diabetic / At Risk`, `Type 2 Early / Pre-diabetic`,
  `Type 2 Controlled / Managed`, and `Type 1 / Insulin-Dependent`.

## 2. Why this uses filters plus KNN

KNN by itself is not the best complete solution. It measures similarity, so it
can say one meal is *near* another meal; it cannot guarantee a budget limit or
an allergy exclusion.

The project therefore uses this pipeline:

1. Keep only rows matching the user's diabetic level.
2. Keep only rows with a sugar-free/no-added-sugar claim in the CSV.
3. Read the lowest meal price in the complete dataset, then keep only meals
   whose complete listed price range is between that minimum and the user's
   maximum budget.
4. Keep only rows in the selected area (or all areas if "All" is chosen).
5. Remove rows whose ingredients contain a selected allergen.
6. Convert area, meal category, food tags, and text tokens into a numeric feature
   matrix with `DictVectorizer`.
7. Use `NearestNeighbors(metric="cosine")` to rank the safe candidates.
8. If exact meal and food matches exist, show only exact matches. Otherwise,
   label the returned rows as nearest safe alternatives.
9. Group the ranked meals by restaurant and show up to two meals per restaurant.

This is a **content-based recommender**, not a supervised classifier. It does not
need a target label and it does not split the 177 rows into training/testing
sets merely to predict a label that does not exist.

## 3. Project structure

```text
DiaNutri/
├── app.py                      Streamlit chat interface
├── chat_store.py               saves and loads chats as JSON files
├── recommender.py              cleaning, filters, and KNN ranking
├── 01_data_cleaning_and_knn.ipynb
├── assets/
│   └── style.css               page styling
├── data/
│   ├── lebanon_diabetic_restaurants.csv
│   └── cleaned_restaurants.csv
├── saved_chats/                created at runtime; personal data, not committed
├── tests/
│   ├── test_recommender.py
│   ├── test_chat_store.py
│   └── test_app.py
├── .github/workflows/tests.yml CI: runs the tests on every push
├── requirements.txt            runtime dependencies
├── requirements-dev.txt        notebook and development dependencies
└── README.md
```

### What each file does

- `01_data_cleaning_and_knn.ipynb`: the educational notebook. Run it from top
  to bottom to inspect, clean, explore, save, and test the dataset.
- `recommender.py`: reusable data-cleaning, feature-engineering, allergy-filter,
  diabetic-level filter, and KNN functions.
- `app.py`: the chat-style website built with Streamlit. It asks the questions
  and shows results, and loads its styling from `assets/style.css`.
- `chat_store.py`: the only module that reads or writes saved chats.
- `data/lebanon_diabetic_restaurants.csv`: untouched copy of the supplied CSV.
- `data/cleaned_restaurants.csv`: generated cleaned/enriched CSV.
- `tests/test_recommender.py`: checks the important cleaning and safety rules.

## 4. Set up the project in VS Code on Windows

Open the `DiaNutri` folder in VS Code. Then open **Terminal → New Terminal**
and run these commands in PowerShell:

```powershell
py -3.13 --version
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The first command must display `Python 3.13.x`. After activating `.venv`, the
`python` command automatically uses the Python 3.13 interpreter stored inside
that virtual environment. You can confirm it with:

```powershell
python --version
```

If PowerShell blocks activation, run this once in that terminal and activate
again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

To open the notebook, also install the development packages:

```powershell
python -m pip install -r requirements-dev.txt
```

In VS Code:

1. Install the official **Python** extension.
2. Install the official **Jupyter** extension.
3. Press `Ctrl+Shift+P`.
4. Choose **Python: Select Interpreter**.
5. Select the interpreter whose path contains `.venv`.
6. Open `01_data_cleaning_and_knn.ipynb`.
7. Click **Select Kernel** and choose the same `.venv` interpreter.
8. Click **Run All**.

The notebook will recreate `data/cleaned_restaurants.csv`.

## 5. Run the web assistant

Keep the virtual environment active and run:

```powershell
python -m streamlit run app.py
```

The terminal displays a local address, normally `http://localhost:8501`. Open
it in your browser if it does not open automatically.

The assistant asks:

1. What is your diabetic level?
2. Which area are you looking in?
3. What meal do you want?
4. Which food do you prefer?
5. What is your maximum budget? The choices begin at the dataset's real minimum
   meal price; the selected value becomes the upper end of the budget interval.
6. Do you have any food allergies?

Every question provides choices and an `Other` option. Allergies allow multiple
choices because a person can have more than one allergy.

## 6. Detailed data-cleaning explanation

The `clean_dataset()` function performs these steps:

### Step A — Standardize column names

Spaces, parentheses, and capital letters are converted to snake case:

```text
Restaurant Name  → restaurant_name
Max Price (USD)  → max_price_usd
```

Snake-case names are easier and safer to use in Python.

### Step B — Validate required columns

The function stops with a clear error if a column required by the app is
missing. This prevents a confusing error later in the model.

### Step C — Clean text

All text columns are stripped, repeated spaces are collapsed, and the trailing
spaces in area values are removed. This fixes `Beirut` versus `Beirut `.

### Step D — Convert numbers

Price and calorie fields are converted with `pd.to_numeric(...,
errors="coerce")`. Invalid numeric text becomes missing and is handled by the
validation step.

### Step E — Remove invalid data

The code removes:

- rows missing a restaurant, location, area, meal, ingredient list, or price;
- repeated restaurant/location/meal combinations;
- negative prices;
- rows where minimum price is greater than maximum price;
- impossible calorie ranges.

### Step F — Validate websites

A website must start with `http://` or `https://`. An invalid website becomes
missing, but its whole meal row is not deleted.

### Step G — Engineer missing features

The raw dataset does not directly answer the app's questions. The code derives:

- `meal_category`: Breakfast, Lunch / Dinner, Dessert, or Meal Plan;
- `food_tags`: Chicken / Turkey, Fish / Seafood, Beef / Lamb,
  Vegetarian / Vegan, and/or Mixed / Customizable;
- `sugar_status`: explicit claim, option available, no added/refined sugar, or
  claim not stated;
- `contains_*` Boolean flags for gluten, dairy, eggs, peanuts, tree nuts,
  sesame, fish, shellfish, and soy;
- average price and average calories.

The derived fields are based on keyword rules and must be manually reviewed
when the dataset grows.

## 7. How the KNN features work

Each meal becomes a feature dictionary. Important choices receive larger
weights:

```python
{
    "bias": 1.0,
    "area=beirut": 3.0,
    "meal=lunch / dinner": 3.0,
    "food=chicken / turkey": 2.5,
    "token=quinoa": 0.35,
}
```

The user's answers become a similar dictionary. `DictVectorizer` converts the
dictionaries to columns, and KNN calculates cosine distance. A distance near
zero means the meal is more similar to the request.

The model deliberately does **not** encode allergies as soft similarity
features. Allergy rows are removed before KNN.

## 8. Test the project

Run:

```powershell
python -m unittest discover -s tests -v
```

The tests verify that:

- trailing spaces are cleaned;
- the required engineered columns exist;
- all 31 sugar-claim rows are detected;
- meal prices stay between the complete dataset's minimum and the user's
  maximum budget, even when no matching meal is available;
- tree-nut meals are removed for a tree-nut allergy;
- a known Beirut chicken query returns exact preference matches.

The suite also checks the chat storage module (atomic saves, sorting, corrupt
files, path-safe ids) and runs the full Streamlit flow with Streamlit's
`AppTest`. Those tests write to a temporary folder, never to `saved_chats/`.
The same command runs in GitHub Actions on every push (`.github/workflows/tests.yml`).

## 9. Example query

Use these answers for the first test:

```text
Diabetic level: Type 2 Early / Pre-diabetic
Area: Beirut
Meal: Lunch / Dinner
Food: Chicken / Turkey
Maximum budget: $15
Allergies: None
```

With the supplied CSV, the safe-filter stage finds eligible rows and KNN places
the closest chicken meals first.

## 10. When you add more data

1. Add one row per meal, not one row per restaurant.
2. Use the same spelling for repeated restaurant details.
3. Keep price and calories numeric.
4. Write full ingredients, especially all allergens and sauces.
5. Add verified fields when possible: `total_sugar_g`, `added_sugar_g`, serving
   size, carbohydrate grams, source URL, and last verified date.
6. Do not infer medical suitability only from the restaurant name.
7. Run the notebook and tests again after every dataset update.

For a real public system, `total_sugar_g`, complete allergen declarations, and
source verification are more important than changing the machine-learning
algorithm.

## 11. Chat history

Every conversation is automatically saved in the `saved_chats/` folder as a
JSON file. The sidebar shows all previous chats. You can:

- Click **➕ New Search** to start a fresh conversation (the old one stays saved).
- Click any previous chat to resume and view its results.
- Delete old chats with the trash icon.

### Where the data is stored and who can see it

- **Location:** `saved_chats/` inside the project folder, one file per chat,
  named `chat_<id>.json`. Set the environment variable `DIANUTRI_CHATS_DIR` to
  store them somewhere else.
- **Contents:** diabetic level, area, meal preference, budget, allergies, the
  conversation, and the recommended restaurants. This is health-related
  personal data.
- **Storage:** plain JSON on the disk of the machine that runs the app. There
  is no database, no encryption, and no user accounts. Anyone who can read the
  folder can read every chat.
- **Not for public deployment as-is:** every visitor of a hosted copy shares
  the same folder and sees the same sidebar list. Before publishing the app,
  add user accounts or per-visitor storage, or turn saving off.
- **Git:** `saved_chats/` is listed in `.gitignore`. Do not remove that line.

This works like ChatGPT or Manus: each search is a separate tab/session,
preserved for future reference.

## 12. What to say when presenting the project

> I created a hybrid content-based recommendation assistant called DiaNutri.
> First, I cleaned and validated 177 meal records from 51 restaurants. Then I engineered meal,
> preference, sugar-claim, and allergen features from text. The system applies
> strict filters for diabetic level, area, the interval from the dataset's
> minimum price to the user's maximum budget, sugar claims, and allergies.
> Finally, an unsupervised KNN model ranks the remaining meals by cosine
> similarity and groups them by restaurant. I kept health constraints outside
> the similarity model so they cannot be relaxed by KNN. The app also saves
> every conversation so users can revisit previous searches.
