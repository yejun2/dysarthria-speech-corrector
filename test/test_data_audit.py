import tempfile
import unittest
import wave
from pathlib import Path

from scripts.audit_finetune_data import audit, utterance_id


def write_wav(path: Path, sample_rate: int = 16000, channels: int = 1, seconds: float = 0.1) -> None:
    frames = int(sample_rate * seconds)
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(b"\x00\x00" * frames * channels)


class DataAuditTest(unittest.TestCase):
    def test_valid_nested_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            leaf = root / "speaker 1" / "session"
            leaf.mkdir(parents=True)
            write_wav(leaf / "sample.wav")
            (leaf / "sample.txt").write_text("k ɯ ɾ ʌ m\n", encoding="utf-8")

            report = audit(root, root)

            self.assertEqual(report["summary"]["paired_files"], 1)
            self.assertEqual(report["summary"]["errors"], 0)
            self.assertEqual(report["records"][0]["utterance_id"], "speaker_1__session__sample")
            self.assertEqual(report["token_frequencies"]["ɾ"], 1)

    def test_missing_pair_and_non_ipa_label_are_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_wav(root / "paired.wav")
            write_wav(root / "missing.wav")
            (root / "paired.txt").write_text("o/ 안녕하세요?\n", encoding="utf-8")

            report = audit(root, root)
            codes = {issue["code"] for issue in report["issues"]}

            self.assertIn("missing_label", codes)
            self.assertIn("hangul_in_label", codes)
            self.assertIn("kspon_annotation", codes)

    def test_separate_mirrored_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio_root = root / "audio"
            label_root = root / "labels"
            (audio_root / "a").mkdir(parents=True)
            (label_root / "a").mkdir(parents=True)
            write_wav(audio_root / "a" / "x.wav")
            (label_root / "a" / "x.txt").write_text("a n\n", encoding="utf-8")

            report = audit(audio_root, label_root)

            self.assertEqual(report["summary"]["paired_files"], 1)
            self.assertEqual(report["summary"]["errors"], 0)

    def test_utterance_id_uses_relative_directories(self):
        self.assertNotEqual(utterance_id("speaker_a/same"), utterance_id("speaker_b/same"))


if __name__ == "__main__":
    unittest.main()
