"""Cleaning and KNN recommendation logic for the DiaNutri project.

The recommender uses two stages:
1. Exact filters for area, budget, sugar claims, allergies, and diabetic level.
2. K-nearest-neighbors ranking for meal and food preferences.

This module intentionally keeps health-related exclusions outside KNN. A
similarity model must never decide that an allergen is "close enough".
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.neighbors import NearestNeighbors


ALLERGEN_PATTERNS: dict[str, str] = {
    "Gluten": (
        r"\b(?:wheat|flour|semolina|bread|baguette|wrap|bun|bulgur|pita|"
        r"crackers|farro|barley|rye|couscous)\b"
    ),
    "Dairy": r"\b(?:milk|cheese|labneh|yogurt|cream|whey|dairy)\b",
    "Eggs": r"\beggs?\b",
    "Peanuts": r"\bpeanuts?\b",
    "Tree nuts": r"\b(?:almonds?|walnuts?|pistachios?|cashews?|hazelnuts?|nuts?)\b",
    "Sesame": r"\b(?:sesame|tahini)\b",
    "Fish": r"\b(?:salmon|tuna|fish)\b",
    "Shellfish": r"\b(?:shrimp|prawn|crab|lobster|shellfish)\b",
    "Soy": r"\b(?:soy|tofu|edamame)\b",
}

MEAL_CHOICES = ["Any meal", "Breakfast", "Lunch / Dinner", "Dessert", "Meal Plan"]
FOOD_CHOICES = [
    "Any food",
    "Chicken / Turkey",
    "Fish / Seafood",
    "Beef / Lamb",
    "Vegetarian / Vegan",
    "Mixed / Customizable",
]

DIABETIC_LEVEL_CHOICES = [
    "Pre-diabetic / At Risk",
    "Type 2 Early / Pre-diabetic",
    "Type 2 Controlled / Managed",
    "Type 1 / Insulin-Dependent",
]

# Mapping from user-selected diabetic level to the matching dataset values
DIABETIC_LEVEL_MATCH = {
    "Pre-diabetic / At Risk": {
        "Pre-diabetic / At Risk",
        "Type 2 Early / Pre-diabetic",
    },
    "Type 2 Early / Pre-diabetic": {
        "Type 2 Early / Pre-diabetic",
    },
    "Type 2 Controlled / Managed": {
        "Type 2 Controlled / Managed",
    },
    "Type 1 / Insulin-Dependent": {
        "Type 1 / Insulin-Dependent",
    },
}

# Results interpretation per diabetic level
DIABETIC_RESULTS_ADVICE: dict[str, dict[str, str]] = {
    "Pre-diabetic / At Risk": {
        "focus": (
            "These results focus on meals with no added sugar and balanced "
            "carbohydrates to help prevent progression to diabetes."
        ),
        "tip": (
            "Tip: Choose meals labeled as 'sugar-free' or 'no added sugar'. "
            "Monitor portion sizes and pair carbohydrates with protein."
        ),
    },
    "Type 2 Early / Pre-diabetic": {
        "focus": (
            "These results are tailored for early Type 2 diabetes, "
            "prioritizing low-glycemic options and controlled sugar content."
        ),
        "tip": (
            "Tip: Prefer meals with explicitly sugar-free options. "
            "Consider meals with high fiber and lean protein."
        ),
    },
    "Type 2 Controlled / Managed": {
        "focus": (
            "These results support managed Type 2 diabetes with a variety "
            "of sugar-conscious meal options."
        ),
        "tip": (
            "Tip: You have more flexibility with food choices. "
            "Still monitor total carbohydrates and watch for hidden sugars."
        ),
    },
    "Type 1 / Insulin-Dependent": {
        "focus": (
            "These results are filtered for insulin-dependent diabetes, "
            "focusing on meals with predictable carbohydrate content."
        ),
        "tip": (
            "Tip: Check meal descriptions for carbohydrate estimates "
            "and consult with your dietitian to match insulin dosing."
        ),
    },
}


def _snake_case(value: str) -> str:
    """Convert a column title such as 'Max Price (USD)' to 'max_price_usd'."""
    value = re.sub(r"[^A-Za-z0-9]+", "_", value.strip())
    return value.strip("_").lower()


def _combined_text(row: pd.Series) -> str:
    """Return searchable text from the meal fields in lowercase."""
    fields = [
        "meal_name",
        "meal_description",
        "ingredients",
        "diabetic_notes",
    ]
    return " ".join(str(row.get(field, "")) for field in fields).lower()


def infer_meal_category(row: pd.Series) -> str:
    """Create the meal-time category that is missing from the source CSV."""
    name = str(row.get("meal_name", "")).lower()
    source_type = str(row.get("meal_type_program_item", "")).lower()

    if source_type == "program" or "plan" in name:
        return "Meal Plan"
    if re.search(r"\b(cake|sweet|dessert)\b", name):
        return "Dessert"
    if re.search(r"\b(pancake|acai|labneh)\b", name):
        return "Breakfast"
    return "Lunch / Dinner"


def infer_food_tags(row: pd.Series) -> str:
    """Infer one or more broad food preferences from the text fields."""
    text = _combined_text(row)
    tags: list[str] = []

    if re.search(r"\b(chicken|turkey|tawook)\b", text):
        tags.append("Chicken / Turkey")
    if re.search(r"\b(salmon|tuna|fish|shrimp|prawn|crab|lobster|seafood)\b", text):
        tags.append("Fish / Seafood")
    if re.search(r"\b(beef|lamb|kafta)\b", text):
        tags.append("Beef / Lamb")

    animal_tags = len(tags)
    explicitly_plant_based = bool(re.search(r"\b(vegan|vegetarian|plant-based)\b", text))
    if animal_tags == 0 or explicitly_plant_based:
        tags.append("Vegetarian / Vegan")

    is_flexible = bool(
        re.search(r"\b(custom|choice|choose|build your own|mixed|proteins|meal plan)\b", text)
    )
    if animal_tags > 1 or is_flexible:
        tags.append("Mixed / Customizable")

    # dict.fromkeys removes repeats while preserving their order.
    return " | ".join(dict.fromkeys(tags))


def infer_sugar_status(row: pd.Series) -> str:
    """Classify only the sugar claim visible in this dataset.

    This is evidence extraction, not a medical or laboratory verification.
    """
    text = _combined_text(row)

    if re.search(r"sugar[- ]free.{0,25}option|option.{0,25}sugar[- ]free", text):
        return "Sugar-free option available"
    if re.search(r"\bsugar[- ]free\b", text):
        return "Explicitly sugar-free"
    if re.search(
        r"\b(no added sugar|without added sugar|no white sugar|no refined sugars?)\b",
        text,
    ):
        return "No added/refined sugar stated"
    return "Sugar claim not stated"


def clean_dataset(
    input_path: str | Path,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load, validate, clean, enrich, and optionally save the restaurant CSV."""
    df = pd.read_csv(input_path)
    df.columns = [_snake_case(column) for column in df.columns]

    required_columns = {
        "restaurant_name",
        "location",
        "area",
        "website",
        "meal_name",
        "meal_description",
        "ingredients",
        "min_price_usd",
        "max_price_usd",
        "diabetic_notes",
        "meal_type_program_item",
    }
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    text_columns = df.select_dtypes(include=["object", "string"]).columns
    for column in text_columns:
        df[column] = (
            df[column]
            .astype("string")
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    numeric_columns = [
        "min_price_usd",
        "max_price_usd",
        "min_calories",
        "max_calories",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    # Remove unusable rows, repeated meal rows, and impossible numeric ranges.
    df = df.dropna(
        subset=[
            "restaurant_name",
            "location",
            "area",
            "meal_name",
            "ingredients",
            "min_price_usd",
            "max_price_usd",
        ]
    )
    df = df.drop_duplicates(
        subset=["restaurant_name", "location", "meal_name"], keep="first"
    )
    df = df[
        (df["min_price_usd"] >= 0)
        & (df["max_price_usd"] >= df["min_price_usd"])
    ]
    if {"min_calories", "max_calories"}.issubset(df.columns):
        df = df[
            (df["min_calories"] >= 0)
            & (df["max_calories"] >= df["min_calories"])
        ]

    # Validate websites without deleting a row when only its URL is imperfect.
    website_is_valid = df["website"].str.match(r"^https?://", na=False)
    df.loc[~website_is_valid, "website"] = pd.NA

    df["meal_category"] = df.apply(infer_meal_category, axis=1)
    df["food_tags"] = df.apply(infer_food_tags, axis=1)
    df["sugar_status"] = df.apply(infer_sugar_status, axis=1)
    df["sugar_eligible"] = df["sugar_status"].ne("Sugar claim not stated")
    df["average_price_usd"] = (
        (df["min_price_usd"] + df["max_price_usd"]) / 2
    ).round(2)

    if {"min_calories", "max_calories"}.issubset(df.columns):
        df["average_calories"] = (
            (df["min_calories"] + df["max_calories"]) / 2
        ).round(0)

    ingredient_text = df["ingredients"].fillna("").str.lower()
    for allergen, pattern in ALLERGEN_PATTERNS.items():
        column = "contains_" + _snake_case(allergen)
        df[column] = ingredient_text.str.contains(pattern, regex=True, na=False)

    df = df.sort_values(
        ["restaurant_name", "max_price_usd", "meal_name"],
        kind="stable",
    ).reset_index(drop=True)

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output, index=False)

    return df


def _normalise_answer(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).casefold()


def _area_mask(df: pd.DataFrame, requested_area: str) -> pd.Series:
    """Match a standard area exactly, or a custom place inside area/location."""
    if requested_area == "All":
        return pd.Series(True, index=df.index)
    requested = _normalise_answer(requested_area)
    areas = df["area"].fillna("").str.casefold()

    if requested in set(areas):
        return areas.eq(requested)

    safe_requested = re.escape(requested)
    locations = df["location"].fillna("").str.casefold()
    return areas.str.contains(safe_requested, regex=True) | locations.str.contains(
        safe_requested, regex=True
    )


def _diabetic_level_mask(df: pd.DataFrame, requested_level: str) -> pd.Series:
    """Filter meals by their diabetic level compatibility."""
    if requested_level == "Any":
        return pd.Series(True, index=df.index)
    allowed_levels = DIABETIC_LEVEL_MATCH.get(requested_level, set())
    if not allowed_levels:
        return pd.Series(True, index=df.index)
    df_level = df["diabetic_level"].fillna("").str.casefold()
    allowed_lower = {l.casefold() for l in allowed_levels}
    return df_level.isin(allowed_lower)


def _custom_allergy_mask(df: pd.DataFrame, allergy_terms: Iterable[str]) -> pd.Series:
    """Return True for rows containing any user-entered allergy word."""
    mask = pd.Series(False, index=df.index)
    ingredients = df["ingredients"].fillna("").str.casefold()
    for term in allergy_terms:
        clean_term = _normalise_answer(term)
        if clean_term:
            mask |= ingredients.str.contains(re.escape(clean_term), regex=True)
    return mask


_STOP_WORDS = frozenset(
    {"and", "with", "the", "from", "for", "meal", "food", "any", "other", "option"}
)


def _text_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z]{3,}", text.casefold())
        if token not in _STOP_WORDS
    }


def _item_features(row: pd.Series) -> dict[str, float]:
    features: dict[str, float] = {"bias": 1.0}
    features[f"area={_normalise_answer(str(row['area']))}"] = 3.0
    features[f"meal={_normalise_answer(str(row['meal_category']))}"] = 3.0

    for tag in str(row["food_tags"]).split(" | "):
        if tag:
            features[f"food={_normalise_answer(tag)}"] = 2.5

    searchable = " ".join(
        [
            str(row.get("meal_name", "")),
            str(row.get("meal_description", "")),
            str(row.get("ingredients", "")),
        ]
    )
    for token in _text_tokens(searchable):
        features[f"token={token}"] = 0.35
    return features


def _query_features(
    area: str,
    meal_choice: str,
    food_choice: str,
) -> dict[str, float]:
    features: dict[str, float] = {"bias": 1.0}
    if area != "All":
        features[f"area={_normalise_answer(area)}"] = 3.0

    if meal_choice not in {"Any meal", ""}:
        if meal_choice in MEAL_CHOICES:
            features[f"meal={_normalise_answer(meal_choice)}"] = 3.0
        else:
            for token in _text_tokens(meal_choice):
                features[f"token={token}"] = 1.5

    if food_choice not in {"Any food", ""}:
        if food_choice in FOOD_CHOICES:
            features[f"food={_normalise_answer(food_choice)}"] = 2.5
        else:
            for token in _text_tokens(food_choice):
                features[f"token={token}"] = 1.5
    return features


def _preference_matches(row: pd.Series, meal_choice: str, food_choice: str) -> bool:
    meal_match = True
    food_match = True

    if meal_choice not in {"Any meal", ""}:
        if meal_choice in MEAL_CHOICES:
            meal_match = row["meal_category"] == meal_choice
        else:
            meal_match = _normalise_answer(meal_choice) in _normalise_answer(
                f"{row['meal_name']} {row['meal_description']}"
            )

    if food_choice not in {"Any food", ""}:
        if food_choice in FOOD_CHOICES:
            food_match = food_choice in str(row["food_tags"]).split(" | ")
        else:
            food_match = _normalise_answer(food_choice) in _normalise_answer(
                f"{row['meal_name']} {row['ingredients']}"
            )
    return bool(meal_match and food_match)


def recommend_restaurants(
    df: pd.DataFrame,
    *,
    area: str,
    meal_choice: str,
    food_choice: str,
    max_budget: float,
    allergies: Iterable[str] = (),
    other_allergies: Iterable[str] = (),
    diabetic_level: str = "Any",
    top_restaurants: int = 10,
    meals_per_restaurant: int = 3,
) -> dict:
    """Return safe-filtered and KNN-ranked restaurants with smart fallbacks.

    The budget interval starts at the lowest meal price in the complete dataset
    and ends at the user's maximum budget. `max_price_usd <= max_budget` keeps
    every returned meal's complete listed price range inside that interval.
    Allergies, sugar claims, and diabetic level are also hard filters.
    """
    if max_budget <= 0:
        raise ValueError("Maximum budget must be greater than zero.")

    dataset_min_price = float(df["min_price_usd"].min())
    if max_budget < dataset_min_price:
        raise ValueError(
            f"Maximum budget must be at least the dataset minimum price "
            f"(${dataset_min_price:g})."
        )

    def apply_hard_filters(data, b, a):
        c = data.copy()
        audit = {"starting_meals": int(len(c))}

        c = c[_diabetic_level_mask(c, diabetic_level)]
        audit["after_diabetic_level_filter"] = int(len(c))

        c = c[c["sugar_eligible"]]
        audit["after_sugar_filter"] = int(len(c))

        c = c[
            (c["min_price_usd"] >= dataset_min_price)
            & (c["max_price_usd"] <= float(b))
        ]
        audit["dataset_min_price"] = dataset_min_price
        audit["requested_max_budget"] = float(b)
        audit["after_budget_filter"] = int(len(c))

        c = c[_area_mask(c, a)]
        audit["after_area_filter"] = int(len(c))

        selected_allergies = {
            allergy for allergy in allergies if allergy in ALLERGEN_PATTERNS
        }
        for allergy in selected_allergies:
            allergen_column = "contains_" + _snake_case(allergy)
            c = c[~c[allergen_column]]

        c = c[~_custom_allergy_mask(c, other_allergies)]
        audit["after_allergy_filter"] = int(len(c))

        return c, audit

    candidates, audit = apply_hard_filters(df, max_budget, area)
    fallback_mode = None

    if candidates.empty:
        # Fallback 1: Try expanding the area to "All"
        if area != "All":
            candidates, audit = apply_hard_filters(df, max_budget, "All")
            if not candidates.empty:
                fallback_mode = "area_expanded"

    if candidates.empty:
        return {
            "restaurants": [],
            "audit": audit,
            "message": (
                "I couldn't find meals matching your criteria between "
                f"${dataset_min_price:g} and ${max_budget:g}. "
                "Try a higher maximum budget or different preferences."
            ),
            "exact_preference_meals": 0,
            "used_preference_alternatives": False,
            "fallback_mode": "failed",
            "minimum_dataset_price": dataset_min_price,
            "maximum_budget": float(max_budget),
        }

    item_feature_dicts = [
        _item_features(row) for _, row in candidates.iterrows()
    ]
    query_feature_dict = _query_features(area, meal_choice, food_choice)

    vectorizer = DictVectorizer(sparse=True)
    matrix = vectorizer.fit_transform(item_feature_dicts + [query_feature_dict])
    item_matrix = matrix[:-1]
    query_matrix = matrix[-1]

    neighbors_to_rank = len(candidates)
    model = NearestNeighbors(
        n_neighbors=neighbors_to_rank,
        metric="cosine",
        algorithm="brute",
    )
    model.fit(item_matrix)
    distances, positions = model.kneighbors(query_matrix)

    ranked = candidates.iloc[positions[0]].copy()
    ranked["knn_distance"] = distances[0]
    ranked["preference_match"] = ranked.apply(
        lambda row: _preference_matches(row, meal_choice, food_choice), axis=1
    )
    if "certified_diabetic" in ranked.columns:
        ranked["certified_sort"] = ranked["certified_diabetic"].astype(str).str.casefold().ne("yes")
    else:
        ranked["certified_sort"] = True
    ranked = ranked.sort_values(
        ["preference_match", "knn_distance", "certified_sort", "max_price_usd"],
        ascending=[False, True, True, True],
        kind="stable",
    )

    exact_match_count = int(ranked["preference_match"].sum())
    output_rows = (
        ranked[ranked["preference_match"]].copy()
        if exact_match_count > 0
        else ranked.copy()
    )

    restaurant_results: list[dict] = []
    for restaurant_name, group in output_rows.groupby("restaurant_name", sort=False):
        selected_meals = group.head(meals_per_restaurant)
        first = selected_meals.iloc[0]
        meals = []
        for _, meal in selected_meals.iterrows():
            min_price = float(meal["min_price_usd"])
            max_price = float(meal["max_price_usd"])
            price_text = (
                f"${min_price:.0f}"
                if min_price == max_price
                else f"${min_price:.0f}–${max_price:.0f}"
            )
            meals.append(
                {
                    "meal_name": str(meal["meal_name"]),
                    "meal_description": str(meal["meal_description"]),
                    "price": price_text,
                    "sugar_status": str(meal["sugar_status"]),
                    "meal_category": str(meal["meal_category"]),
                    "food_tags": str(meal["food_tags"]),
                    "preference_match": bool(meal["preference_match"]),
                    "knn_distance": round(float(meal["knn_distance"]), 4),
                }
            )

        restaurant_results.append(
            {
                "restaurant_name": str(restaurant_name),
                "location": str(first["location"]),
                "website": None if pd.isna(first["website"]) else str(first["website"]),
                "certified_diabetic": str(first.get("certified_diabetic", "Not stated")),
                "meals": meals,
            }
        )
        if len(restaurant_results) >= top_restaurants:
            break

    message = "I found these great matches for you!"
    if fallback_mode == "area_expanded":
        message = f"I couldn't find matches exactly in {area}, so I found these safe alternatives in nearby areas for you."
    return {
        "restaurants": restaurant_results,
        "audit": audit,
        "message": message,
        "exact_preference_meals": exact_match_count,
        "used_preference_alternatives": exact_match_count == 0,
        "fallback_mode": fallback_mode,
        "minimum_dataset_price": dataset_min_price,
        "maximum_budget": float(max_budget),
    }


if __name__ == "__main__":
    project_directory = Path(__file__).resolve().parent
    raw_path = project_directory / "data" / "lebanon_diabetic_restaurants.csv"
    cleaned_path = project_directory / "data" / "cleaned_restaurants.csv"
    cleaned = clean_dataset(raw_path, cleaned_path)
    print(f"Cleaned {len(cleaned)} rows and saved: {cleaned_path}")
