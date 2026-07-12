import unittest
from argparse import Namespace
from unittest.mock import Mock, patch

from allosaurus.am.reporter import Reporter


class ReporterWandbTest(unittest.TestCase):
    def make_config(self, project="none"):
        return Namespace(
            new_model="uni2005",
            verbose=False,
            log="none",
            wandb_project=project,
            wandb_entity="team",
            wandb_run_name="run-name",
            wandb_mode="offline",
        )

    def test_wandb_disabled_does_not_import_sdk(self):
        with patch.dict("sys.modules", {"wandb": None}):
            reporter = Reporter(self.make_config("none"))
            reporter.log_metrics({"epoch": 1})
            reporter.close()

    def test_wandb_run_receives_metrics_and_finishes(self):
        run = Mock()
        wandb = Mock()
        wandb.init.return_value = run
        with patch.dict("sys.modules", {"wandb": wandb}):
            config = self.make_config("project")
            reporter = Reporter(config)
            reporter.log_metrics({"epoch/validate_per": 0.2})
            reporter.close()

        wandb.init.assert_called_once()
        self.assertEqual(wandb.init.call_args.kwargs["project"], "project")
        self.assertEqual(wandb.init.call_args.kwargs["mode"], "offline")
        run.log.assert_called_once_with({"epoch/validate_per": 0.2})
        run.finish.assert_called_once()


if __name__ == "__main__":
    unittest.main()
