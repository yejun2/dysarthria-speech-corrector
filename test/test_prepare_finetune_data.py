import tempfile
import unittest
import wave
from pathlib import Path

from scripts.prepare_finetune_data import apply_phone_mapping, prepare, split_items, split_items_fixed, Item


def write_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\x00\x00" * 1600)


class PrepareFinetuneDataTest(unittest.TestCase):
    def test_phone_mapping_can_replace_and_drop_tokens(self):
        self.assertEqual(apply_phone_mapping(["g", "x", "a"], {"g": "k", "x": None}), ("k", "a"))

    def test_split_is_reproducible_and_nonempty(self):
        items = [Item(str(i), str(i), "w", "t", ("a",)) for i in range(10)]
        first = split_items(items, 0.8, 0.1, 42)
        second = split_items(items, 0.8, 0.1, 42)
        self.assertEqual(first, second)
        self.assertEqual({name: len(values) for name, values in first.items()}, {"train": 8, "validate": 1, "test": 1})

    def test_fixed_split_uses_exact_counts_without_overlap(self):
        items = [Item(str(i), str(i), "w", "t", ("a",)) for i in range(20)]
        splits = split_items_fixed(items, validate_count=5, test_count=2, seed=42)
        self.assertEqual({name: len(values) for name, values in splits.items()}, {"train": 13, "validate": 5, "test": 2})
        id_sets = [{item.utterance_id for item in values} for values in splits.values()]
        self.assertEqual(len(set.union(*id_sets)), 20)
        self.assertEqual(sum(len(values) for values in id_sets), 20)

    def test_prepare_builds_all_manifests_with_real_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            output = root / "work"
            leaf = data_root / "KsponSpeech_01" / "preprocessed" / "ipa"
            leaf.mkdir(parents=True)
            for index in range(10):
                write_wav(leaf / f"KsponSpeech_{index:06d}.wav")
                (leaf / f"KsponSpeech_{index:06d}.txt").write_text("k ɯ ɾ ʌ m\n", encoding="utf-8")

            summary = prepare(data_root, output, "uni2005", "kor", None, 0.8, 0.1, 42, 16000, 10.0)

            self.assertEqual(summary["counts"], {"train": 8, "validate": 1, "test": 1})
            for split in ("train", "validate", "test"):
                self.assertTrue((output / split / "wave").is_file())
                self.assertTrue((output / split / "text").is_file())
            train_text = (output / "train" / "text").read_text(encoding="utf-8")
            self.assertIn("KsponSpeech_01__preprocessed__ipa__KsponSpeech_", train_text)

    def test_prepare_rejects_unsupported_phone(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(3):
                write_wav(root / f"{index}.wav")
                (root / f"{index}.txt").write_text("not_a_phone\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsupported phones"):
                prepare(root, root / "out", "uni2005", "kor", None, 0.8, 0.1, 42, 16000, 10.0)


if __name__ == "__main__":
    unittest.main()
