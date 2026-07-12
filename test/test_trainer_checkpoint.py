import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from allosaurus.am.trainer import Trainer


class TrainerCheckpointTest(unittest.TestCase):
    def test_saves_numbered_model_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer.__new__(Trainer)
            trainer.model = object()
            trainer.checkpoint_path = Path(directory)
            trainer.reporter = Mock()

            with patch("allosaurus.am.trainer.torch_save") as save:
                trainer.save_epoch_checkpoint(0, 0.125)

            save.assert_called_once_with(trainer.model, Path(directory) / "epoch_0001.pt")
            metadata = json.loads((Path(directory) / "epoch_0001.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata, {"epoch": 1, "validate_phone_error_rate": 0.125})
            trainer.reporter.write.assert_called_once()

    def test_epoch_numbers_are_one_based_and_zero_padded(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer.__new__(Trainer)
            trainer.model = object()
            trainer.checkpoint_path = Path(directory)
            trainer.reporter = Mock()

            with patch("allosaurus.am.trainer.torch_save") as save:
                trainer.save_epoch_checkpoint(11, 0.25)

            save.assert_called_once_with(trainer.model, Path(directory) / "epoch_0012.pt")


if __name__ == "__main__":
    unittest.main()
