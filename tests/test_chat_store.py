"""Tests for local chat storage (chat_store.py)."""

from pathlib import Path
import json
import sys
import tempfile
import unittest


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIRECTORY))

import chat_store  # noqa: E402


class ChatStoreTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _chat(self, chat_id, created_at, area="Beirut"):
        return {
            "chat_id": chat_id,
            "step": 1,
            "answers": {"area": area},
            "history": [["Q?", "A"]],
            "results": None,
            "diabetic_level": "Any",
            "created_at": created_at,
        }

    def test_save_then_load_round_trip(self):
        data = self._chat("chat_0123456789", "2026-01-01T10:00:00")
        chat_store.save_chat("chat_0123456789", data, self.folder)
        self.assertEqual(chat_store.load_chat("chat_0123456789", self.folder), data)

    def test_save_leaves_no_temp_files(self):
        chat_store.save_chat("chat_0123456789", self._chat("chat_0123456789", "t"), self.folder)
        self.assertEqual([p.name for p in self.folder.iterdir()], ["chat_0123456789.json"])

    def test_list_is_sorted_newest_first_by_creation_time(self):
        # Ids are random, so sorting by filename would not give time order.
        chat_store.save_chat("chat_ffffffffff", self._chat("chat_ffffffffff", "2026-01-01T09:00:00"), self.folder)
        chat_store.save_chat("chat_0000000000", self._chat("chat_0000000000", "2026-03-01T09:00:00"), self.folder)
        chat_store.save_chat("chat_aaaaaaaaaa", self._chat("chat_aaaaaaaaaa", "2026-02-01T09:00:00"), self.folder)
        order = [chat["chat_id"] for chat in chat_store.list_chats(self.folder)]
        self.assertEqual(order, ["chat_0000000000", "chat_aaaaaaaaaa", "chat_ffffffffff"])

    def test_list_skips_corrupt_files(self):
        chat_store.save_chat("chat_0123456789", self._chat("chat_0123456789", "t"), self.folder)
        (self.folder / "chat_badbadbad0.json").write_text("{not json", encoding="utf-8")
        self.assertEqual([c["chat_id"] for c in chat_store.list_chats(self.folder)], ["chat_0123456789"])

    def test_load_missing_or_corrupt_returns_none(self):
        self.assertIsNone(chat_store.load_chat("chat_1111111111", self.folder))
        (self.folder / "chat_2222222222.json").write_text("oops", encoding="utf-8")
        self.assertIsNone(chat_store.load_chat("chat_2222222222", self.folder))

    def test_delete_is_idempotent(self):
        chat_store.save_chat("chat_0123456789", self._chat("chat_0123456789", "t"), self.folder)
        chat_store.delete_chat("chat_0123456789", self.folder)
        chat_store.delete_chat("chat_0123456789", self.folder)
        self.assertEqual(list(self.folder.glob("*.json")), [])

    def test_rejects_path_traversal_ids(self):
        for bad_id in ["../secrets", "chat_../../x", "chat_ABCDEFGHIJ", "chat_123"]:
            with self.subTest(chat_id=bad_id):
                with self.assertRaises(ValueError):
                    chat_store.save_chat(bad_id, {}, self.folder)
                self.assertIsNone(chat_store.load_chat(bad_id, self.folder))
                chat_store.delete_chat(bad_id, self.folder)  # must not raise

    def test_directory_override_via_environment(self):
        import os

        previous = os.environ.get("DIANUTRI_CHATS_DIR")
        os.environ["DIANUTRI_CHATS_DIR"] = str(self.folder)
        try:
            self.assertEqual(chat_store.chats_dir(), self.folder)
        finally:
            if previous is None:
                os.environ.pop("DIANUTRI_CHATS_DIR", None)
            else:
                os.environ["DIANUTRI_CHATS_DIR"] = previous

    def test_saved_file_is_utf8_json(self):
        data = self._chat("chat_0123456789", "t", area="Hamra · مرحبا")
        chat_store.save_chat("chat_0123456789", data, self.folder)
        raw = (self.folder / "chat_0123456789.json").read_text(encoding="utf-8")
        self.assertEqual(json.loads(raw)["answers"]["area"], "Hamra · مرحبا")


if __name__ == "__main__":
    unittest.main()
