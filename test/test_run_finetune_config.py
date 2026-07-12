import tempfile
import unittest
from pathlib import Path

from scripts.run_finetune import build_parser, load_config_defaults


class RunFinetuneConfigTest(unittest.TestCase):
    def test_loads_nested_yaml_config(self):
        defaults = load_config_defaults(Path("configs/experiments/kspon_v1.yaml"))
        args = build_parser(defaults).parse_args([])
        self.assertEqual(args.stage, "train")
        self.assertEqual(args.work_dir, Path("workspace/kspon_v1"))
        self.assertEqual(args.lang, "configs/inventories/kspon_korean_ipa.txt")
        self.assertEqual(args.phone_init_map, "configs/inventories/kspon_phone_init.json")
        self.assertEqual(args.expected_validate_size, 5000)
        self.assertEqual(args.wandb_project, "allosaurus-finetune")

    def test_cli_overrides_yaml_defaults(self):
        defaults = load_config_defaults(Path("configs/experiments/kspon_v1.yaml"))
        args = build_parser(defaults).parse_args(["--epoch", "2", "--wandb-mode", "offline"])
        self.assertEqual(args.epoch, 2)
        self.assertEqual(args.wandb_mode, "offline")

    def test_unknown_yaml_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text("training:\n  typo_epoch: 10\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown key"):
                load_config_defaults(path)


if __name__ == "__main__":
    unittest.main()
