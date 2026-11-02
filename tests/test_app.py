"""Streamlit flow tests using Streamlit's AppTest runner.

Chats are written to a temporary folder through DIANUTRI_CHATS_DIR, so the real
saved_chats/ folder is never touched.

AppTest keeps widget nodes from before a st.rerun(), so each test performs at
most one interaction per run. Multi-step states are seeded in session_state.
"""

from pathlib import Path
import os
import sys
import tempfile
import unittest


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIRECTORY))

from streamlit.testing.v1 import AppTest  # noqa: E402

import chat_store  # noqa: E402

APP_PATH = PROJECT_DIRECTORY / "app.py"
TIMEOUT = 120
CHAT_ID = "chat_0123456789"
OLD_CREATED_AT = "2026-01-01T10:00:00"

ANSWERS = {
    "diabetic_level": "Type 2 Early / Pre-diabetic",
    "area": "Beirut",
    "meal_choice": "Lunch / Dinner",
    "max_budget": 15.0,
}
HISTORY = [
    ["What is your diabetic level?", "Type 2 Early / Pre-diabetic"],
    ["Which area are you looking in?", "Beirut"],
    ["What meal do you want?", "Lunch / Dinner"],
    ["What is your maximum budget?", "$5–$15"],
]


def _button(at, label):
    matches = [b for b in at.button if b.label == label]
    if not matches:
        raise AssertionError(f"No button labelled {label!r}")
    return matches[0]


class ChatFlowTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self._tmp.name)
        self._previous_env = os.environ.get("DIANUTRI_CHATS_DIR")
        os.environ["DIANUTRI_CHATS_DIR"] = str(self.folder)

    def tearDown(self):
        if self._previous_env is None:
            os.environ.pop("DIANUTRI_CHATS_DIR", None)
        else:
            os.environ["DIANUTRI_CHATS_DIR"] = self._previous_env
        self._tmp.cleanup()

    def _app(self):
        return AppTest.from_file(str(APP_PATH), default_timeout=TIMEOUT)

    def _seed_session(self, at, step, history, results=None):
        at.session_state["step"] = step
        at.session_state["answers"] = dict(ANSWERS)
        at.session_state["history"] = [tuple(item) for item in history]
        at.session_state["results"] = results
        at.session_state["chat_id"] = CHAT_ID
        at.session_state["diabetic_level"] = ANSWERS["diabetic_level"]
        at.session_state["created_at"] = OLD_CREATED_AT

    def _saved_files(self):
        return sorted(self.folder.glob("chat_*.json"))

    def _write_saved_chat(self, step=1, history=None):
        chat_store.save_chat(
            CHAT_ID,
            {
                "chat_id": CHAT_ID,
                "step": step,
                "answers": dict(ANSWERS),
                "history": history if history is not None else HISTORY[:1],
                "results": None,
                "diabetic_level": ANSWERS["diabetic_level"],
                "created_at": OLD_CREATED_AT,
            },
            self.folder,
        )

    def test_first_screen_renders_without_exceptions(self):
        at = self._app().run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual([s.key for s in at.selectbox], ["diabetic_level_select"])

    def test_first_question_advances_to_area(self):
        at = self._app().run()
        at.selectbox(key="diabetic_level_select").set_value("Type 2 Early / Pre-diabetic")
        _button(at, "Continue").click().run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(at.session_state["step"], 1)
        self.assertEqual(len(self._saved_files()), 1)

    def test_new_chat_on_blank_session_writes_nothing(self):
        at = self._app().run()
        _button(at, "➕ New Chat").click().run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(self._saved_files(), [])

    def test_allergy_step_returns_results_and_saves_chat(self):
        at = self._app()
        at.run()
        self._seed_session(at, step=4, history=HISTORY)
        at.run()  # render the allergy question for the seeded state
        at.multiselect(key="allergy_select").set_value(["None"])
        _button(at, "Show recommendations").click().run()

        self.assertEqual(len(at.exception), 0)
        self.assertEqual(at.session_state["step"], 5)
        saved = chat_store.load_chat(CHAT_ID, self.folder)
        self.assertEqual(saved["step"], 5)
        self.assertIsNotNone(saved["results"])
        self.assertEqual(saved["answers"]["area"], "Beirut")

    def test_loading_a_saved_chat_restores_its_creation_time(self):
        self._write_saved_chat()
        at = self._app().run()
        at.button(key=f"load_{CHAT_ID}").click().run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(at.session_state["chat_id"], CHAT_ID)
        self.assertEqual(at.session_state["created_at"], OLD_CREATED_AT)

    def test_saving_keeps_original_creation_time(self):
        at = self._app()
        at.run()
        self._seed_session(at, step=1, history=HISTORY[:1])
        _button(at, "Start over").click().run()

        saved = chat_store.load_chat(CHAT_ID, self.folder)
        self.assertIsNotNone(saved)
        self.assertEqual(saved["created_at"], OLD_CREATED_AT)

    def test_deleting_a_chat_removes_its_file(self):
        self._write_saved_chat()
        at = self._app().run()
        at.button(key=f"del_{CHAT_ID}").click().run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(self._saved_files(), [])


if __name__ == "__main__":
    unittest.main()
