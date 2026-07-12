import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from scripts.evaluate_finetune import evaluate, read_manifest


class EvaluateFinetuneTest(unittest.TestCase):
    def test_read_manifest_rejects_mismatched_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "wave").write_text("a /a.wav\n", encoding="utf-8")
            (root / "text").write_text("b k\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                read_manifest(root)

    def test_evaluate_writes_corpus_per(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "test"
            manifest.mkdir()
            (manifest / "wave").write_text("a /a.wav\nb /b.wav\n", encoding="utf-8")
            (manifest / "text").write_text("a k a\nb n a\n", encoding="utf-8")
            recognizer = Mock()
            recognizer.recognize.side_effect = ["k a", "m a"]

            with patch("scripts.evaluate_finetune.read_recognizer", return_value=recognizer):
                metrics = evaluate("model", "kor", manifest, root / "result", -1)

            self.assertEqual(metrics["edit_distance"], 1)
            self.assertEqual(metrics["reference_phones"], 4)
            self.assertEqual(metrics["phone_error_rate"], 0.25)
            stored = json.loads((root / "result" / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(stored["phone_error_rate"], 0.25)


if __name__ == "__main__":
    unittest.main()
