"""Small regression tests for cleaning and recommendation rules."""

from pathlib import Path
import sys
import unittest


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIRECTORY))

from recommender import clean_dataset, recommend_restaurants  # noqa: E402


class RecommenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = clean_dataset(
            PROJECT_DIRECTORY / "data" / "lebanon_diabetic_restaurants.csv"
        )

    def test_cleaning_fixes_beirut_whitespace(self):
        self.assertNotIn("Beirut ", set(self.data["area"]))
        self.assertIn("Beirut", set(self.data["area"]))

    def test_cleaning_derives_required_features(self):
        expected = {
            "meal_category",
            "food_tags",
            "sugar_status",
            "sugar_eligible",
            "contains_gluten",
            "contains_tree_nuts",
        }
        self.assertTrue(expected.issubset(self.data.columns))
        self.assertEqual(int(self.data["sugar_eligible"].sum()), 31)

    def test_budget_is_a_hard_filter(self):
        dataset_min_price = float(self.data["min_price_usd"].min())
        result = recommend_restaurants(
            self.data,
            area="Beirut",
            meal_choice="Any meal",
            food_choice="Any food",
            max_budget=10,
            allergies=[],
        )
        for restaurant in result["restaurants"]:
            for meal in restaurant["meals"]:
                price_parts = meal["price"].replace("$", "").split("–")
                minimum_price = float(price_parts[0])
                maximum_price = float(price_parts[-1])
                self.assertGreaterEqual(minimum_price, dataset_min_price)
                self.assertLessEqual(maximum_price, 10)
        self.assertEqual(result["minimum_dataset_price"], dataset_min_price)
        self.assertEqual(result["maximum_budget"], 10)

    def test_budget_is_never_expanded_when_no_match_exists(self):
        result = recommend_restaurants(
            self.data,
            area="All",
            meal_choice="Any meal",
            food_choice="Any food",
            max_budget=6,
            allergies=[],
            diabetic_level="Type 1 / Insulin-Dependent",
        )
        self.assertEqual(result["restaurants"], [])
        self.assertEqual(result["fallback_mode"], "failed")
        self.assertEqual(result["maximum_budget"], 6)

    def test_tree_nut_allergy_excludes_tree_nut_ingredients(self):
        result = recommend_restaurants(
            self.data,
            area="Beirut",
            meal_choice="Any meal",
            food_choice="Any food",
            max_budget=35,
            allergies=["Tree nuts"],
        )
        returned_names = {
            meal["meal_name"]
            for restaurant in result["restaurants"]
            for meal in restaurant["meals"]
        }
        # Verify that even with fallback, safety filters are respected
        for restaurant in result["restaurants"]:
            for meal in restaurant["meals"]:
                self.assertNotIn("Power Quickie", returned_names)
                self.assertNotIn("Low-Calorie Dessert", returned_names)

    def test_known_chicken_query_has_exact_matches(self):
        result = recommend_restaurants(
            self.data,
            area="Beirut",
            meal_choice="Lunch / Dinner",
            food_choice="Chicken / Turkey",
            max_budget=15,
            allergies=[],
        )
        self.assertGreater(result["exact_preference_meals"], 0)
        self.assertFalse(result["used_preference_alternatives"])
        for restaurant in result["restaurants"]:
            for meal in restaurant["meals"]:
                self.assertTrue(meal["preference_match"])


if __name__ == "__main__":
    unittest.main()
