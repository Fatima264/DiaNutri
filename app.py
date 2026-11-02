"""DiaNutri Streamlit chat assistant: a guided, five-question restaurant search."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from chat_store import delete_chat, list_chats, load_chat, new_chat_id, save_chat
from recommender import (
    ALLERGEN_PATTERNS,
    DIABETIC_LEVEL_CHOICES,
    MEAL_CHOICES,
    clean_dataset,
    recommend_restaurants,
)


BASE_DIRECTORY = Path(__file__).resolve().parent
RAW_DATA_PATH = BASE_DIRECTORY / "data" / "lebanon_diabetic_restaurants.csv"
STYLE_PATH = BASE_DIRECTORY / "assets" / "style.css"


st.set_page_config(
    page_title="DiaNutri AI Assistant",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def _load_stylesheet() -> str:
    """Read the stylesheet once per server process."""
    return STYLE_PATH.read_text(encoding="utf-8")


def inject_styles() -> None:
    st.markdown(f"<style>{_load_stylesheet()}</style>", unsafe_allow_html=True)


@st.cache_data
def load_data() -> pd.DataFrame:
    """Clean the dataset once and reuse it across Streamlit reruns."""
    return clean_dataset(RAW_DATA_PATH)


# --- Session and chat persistence ---
def init_session(chat_id: str | None = None) -> None:
    """Open a saved chat by id, or start a new one when no saved chat matches."""
    if chat_id:
        data = load_chat(chat_id)
        if data:
            st.session_state.step = data.get("step", 0)
            st.session_state.answers = data.get("answers", {})
            st.session_state.history = [tuple(item) for item in data.get("history", [])]
            st.session_state.results = data.get("results")
            st.session_state.chat_id = chat_id
            st.session_state.diabetic_level = data.get("diabetic_level", "Any")
            st.session_state.created_at = data.get("created_at", datetime.now().isoformat())
            return

    st.session_state.step = 0
    st.session_state.answers = {}
    st.session_state.history = []
    st.session_state.results = None
    st.session_state.chat_id = new_chat_id()
    st.session_state.diabetic_level = "Any"
    st.session_state.created_at = datetime.now().isoformat()


def persist_current_chat() -> None:
    """Save the open chat. Chats with no answers yet are not written to disk."""
    chat_id = st.session_state.get("chat_id")
    if not chat_id or not st.session_state.get("history"):
        return
    save_chat(
        chat_id,
        {
            "chat_id": chat_id,
            "step": st.session_state.step,
            "answers": st.session_state.answers,
            "history": list(st.session_state.history),
            "results": st.session_state.results,
            "diabetic_level": st.session_state.get("diabetic_level", "Any"),
            "created_at": st.session_state.created_at,
        },
    )


def record_answer(question: str, display_answer: str, key: str, value) -> None:
    """Store an answer, add it to the transcript, and advance to the next question."""
    st.session_state.answers[key] = value
    st.session_state.history.append((question, display_answer))
    st.session_state.step += 1
    persist_current_chat()
    st.rerun()


def chat_title(answers: dict) -> str:
    """Build a short sidebar label such as 'Beirut · Lunch / Dinner · Type 2'."""
    parts = [answers.get("area", ""), answers.get("meal_choice", ""), answers.get("diabetic_level", "")]
    return " · ".join(part for part in parts if part) or "New Search"


# --- Rendering ---
def show_previous_messages() -> None:
    """Replay completed questions and answers as chat bubbles."""
    for question, answer in st.session_state.history:
        with st.chat_message("assistant"):
            st.write(question)
        with st.chat_message("user"):
            st.write(answer)


def render_meal(meal_number: int, meal: dict) -> str:
    # Single-line HTML: indented lines can be rendered as Markdown code blocks.
    preference_text = (
        "Exact preference match" if meal["preference_match"] else "Nearest safe alternative"
    )
    return (
        '<div class="meal-box">'
        f"<strong>Meal {meal_number}: {meal['meal_name']}</strong><br>"
        f"{meal['meal_description']}<br>"
        f"<strong>Price:</strong> {meal['price']}<br>"
        f'<span class="small-note">{meal["sugar_status"]} · {preference_text}</span>'
        "</div>"
    )


def render_restaurant(restaurant: dict) -> str:
    website = restaurant["website"]
    website_html = (
        f'<a href="{website}" target="_blank">Open website</a>'
        if website
        else "Website not available"
    )
    meals_html = "".join(
        render_meal(number, meal) for number, meal in enumerate(restaurant["meals"], start=1)
    )
    return (
        '<div class="result-card">'
        f"<h3>{restaurant['restaurant_name']}</h3>"
        f"<strong>Location:</strong> {restaurant['location']}<br>"
        f"<strong>Web:</strong> {website_html}<br>"
        f'<span class="small-note">Certified diabetic field in dataset: {restaurant["certified_diabetic"]}</span>'
        f"{meals_html}"
        "</div>"
    )


def show_results(result: dict) -> None:
    """Show restaurants and at most two ranked meals for each one."""
    with st.chat_message("assistant"):
        if not result["restaurants"]:
            st.error(result["message"])
            return

        if result.get("fallback_mode"):
            st.warning(result["message"])
        else:
            st.success(result["message"])

        for restaurant in result["restaurants"]:
            st.markdown(render_restaurant(restaurant), unsafe_allow_html=True)


# --- Sidebar ---
def render_sidebar() -> None:
    with st.sidebar:
        st.markdown('<div class="sidebar-title">🥗 DiaNutri Chats</div>', unsafe_allow_html=True)

        if st.button("➕ New Chat", width="stretch", key="new_search_btn"):
            persist_current_chat()
            init_session()
            st.rerun()

        st.markdown("---")
        st.markdown("**Previous Chats**")

        saved_chats = list_chats()
        current_chat_id = st.session_state.get("chat_id", "")
        if not saved_chats:
            st.caption("No saved chats yet.")
            return

        for chat in saved_chats:
            chat_id = chat["chat_id"]
            is_active = chat_id == current_chat_id
            open_column, delete_column = st.columns([4, 1])
            with open_column:
                if st.button(
                    chat_title(chat["answers"]),
                    key=f"load_{chat_id}",
                    width="stretch",
                    type="primary" if is_active else "secondary",
                ):
                    init_session(chat_id)
                    st.rerun()
            with delete_column:
                if st.button("🗑", key=f"del_{chat_id}", help="Delete this chat"):
                    delete_chat(chat_id)
                    if is_active:
                        init_session()
                    st.rerun()


# --- Questions ---
def ask_diabetic_level() -> None:
    question = "What is your diabetic level?"
    with st.chat_message("assistant"):
        st.write(question)
        level = st.selectbox(
            "Choose your diabetic level",
            ["Choose...", "Any"] + DIABETIC_LEVEL_CHOICES,
            key="diabetic_level_select",
        )
        if st.button("Continue", key="level_continue"):
            if level == "Choose...":
                st.error("Please choose a diabetic level.")
            else:
                display_level = "Any (show all levels)" if level == "Any" else level
                st.session_state.diabetic_level = level
                record_answer(question, display_level, "diabetic_level", level)


def ask_area(df: pd.DataFrame) -> None:
    question = "Which area are you looking in?"
    with st.chat_message("assistant"):
        st.write(question)
        area_options = sorted(df["area"].dropna().unique().tolist()) + ["Other"]
        area = st.selectbox(
            "Choose an area (or 'All' for all areas)",
            ["Choose...", "All"] + area_options,
            key="area_select",
        )
        custom_area = ""
        if area == "Other":
            custom_area = st.text_input(
                "Type a city, neighborhood, or area (example: Hamra)",
                key="custom_area",
            )
        if st.button("Continue", key="area_continue"):
            final_area = custom_area.strip() if area == "Other" else area
            if area == "Choose..." or not final_area:
                st.error("Please choose an area or type one under Other.")
            else:
                display_area = "All (all areas)" if final_area == "All" else final_area
                record_answer(question, display_area, "area", final_area)


def ask_meal() -> None:
    question = "What meal do you want?"
    with st.chat_message("assistant"):
        st.write(question)
        meal = st.selectbox(
            "Choose a meal category",
            ["Choose..."] + MEAL_CHOICES + ["Other"],
            key="meal_select",
        )
        custom_meal = ""
        if meal == "Other":
            custom_meal = st.text_input(
                "Type the meal you want (example: salad)", key="custom_meal"
            )
        if st.button("Continue", key="meal_continue"):
            final_meal = custom_meal.strip() if meal == "Other" else meal
            if meal == "Choose..." or not final_meal:
                st.error("Please choose a meal or type one under Other.")
            else:
                record_answer(question, final_meal, "meal_choice", final_meal)


def budget_choices(df: pd.DataFrame) -> tuple[float, float, list[float]]:
    """Return the dataset's min and max price plus the standard budget steps inside that range."""
    dataset_min = float(df["min_price_usd"].min())
    dataset_max = float(df["max_price_usd"].max())
    standard = [10.0, 15.0, 20.0, 25.0, 30.0, 35.0]
    values = sorted({dataset_min, dataset_max, *(v for v in standard if dataset_min <= v <= dataset_max)})
    return dataset_min, dataset_max, values


def ask_budget(df: pd.DataFrame) -> None:
    question = "What is your maximum budget?"
    with st.chat_message("assistant"):
        st.write(question)
        dataset_min, dataset_max, values = budget_choices(df)
        labels = [f"${value:g}" for value in values]
        budget = st.selectbox(
            f"Choose your maximum price in USD (minimum: ${dataset_min:g})",
            ["Choose..."] + labels + ["Other"],
            key="budget_select",
        )
        st.caption(
            f"Meals will be searched from the dataset's minimum price "
            f"(${dataset_min:g}) up to your selected budget."
        )
        custom_budget = None
        if budget == "Other":
            custom_budget = st.number_input(
                "Type your maximum budget in USD",
                min_value=dataset_min,
                max_value=max(500.0, dataset_max),
                value=max(dataset_min, min(20.0, dataset_max)),
                step=1.0,
                key="custom_budget",
            )
        if st.button("Continue", key="budget_continue"):
            if budget == "Choose...":
                st.error("Please choose a budget.")
            else:
                final_budget = float(custom_budget) if budget == "Other" else float(budget.replace("$", ""))
                record_answer(
                    question,
                    f"${dataset_min:g}–${final_budget:g}",
                    "max_budget",
                    final_budget,
                )


def ask_allergies(df: pd.DataFrame) -> None:
    question = "Do you have any food allergies?"
    with st.chat_message("assistant"):
        st.write(question)
        allergy_options = ["None"] + list(ALLERGEN_PATTERNS) + ["Other"]
        allergies = st.multiselect("Choose all that apply", allergy_options, key="allergy_select")
        custom_allergies = ""
        if "Other" in allergies:
            custom_allergies = st.text_input(
                "Type other allergies separated by commas",
                placeholder="Example: strawberries, avocado",
                key="custom_allergies",
            )

        if st.button("Show recommendations", key="allergy_continue"):
            if not allergies:
                st.error("Please choose at least one answer.")
            elif "None" in allergies and len(allergies) > 1:
                st.error("Choose either 'None' or specific allergies, not both.")
            elif "Other" in allergies and not custom_allergies.strip():
                st.error("Please type the allergy name(s) under Other.")
            else:
                selected = [item for item in allergies if item not in {"None", "Other"}]
                other_terms = [item.strip() for item in custom_allergies.split(",") if item.strip()]
                display = ", ".join(allergies)
                if other_terms:
                    display = display.replace("Other", ", ".join(other_terms))

                answers = st.session_state.answers
                answers["allergies"] = selected
                answers["other_allergies"] = other_terms
                st.session_state.history.append((question, display))
                st.session_state.results = recommend_restaurants(
                    df,
                    area=answers["area"],
                    meal_choice=answers["meal_choice"],
                    food_choice="Any food",
                    max_budget=answers["max_budget"],
                    allergies=selected,
                    other_allergies=other_terms,
                    diabetic_level=answers.get("diabetic_level", "Any"),
                )
                st.session_state.step = 5
                persist_current_chat()
                st.rerun()


# --- Main page ---
inject_styles()
df = load_data()

if "step" not in st.session_state:
    init_session()

render_sidebar()

st.markdown('<h1 class="main-title">🥗 DiaNutri</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">A diabetes aware restaurant recommendation assistant</p>',
    unsafe_allow_html=True,
)

top_left, top_right = st.columns([3, 1])
with top_left:
    st.caption("Answer five short questions to receive personalized restaurant suggestions.")
with top_right:
    if st.button("Start over", width="stretch"):
        persist_current_chat()
        init_session()
        st.rerun()

show_previous_messages()

step = st.session_state.step
if step == 0:
    ask_diabetic_level()
elif step == 1:
    ask_area(df)
elif step == 2:
    ask_meal()
elif step == 3:
    ask_budget(df)
elif step == 4:
    ask_allergies(df)
elif st.session_state.results:
    show_results(st.session_state.results)
