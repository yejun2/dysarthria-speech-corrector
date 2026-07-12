import tempfile
import unittest
from pathlib import Path

import torch

from allosaurus.am.utils import torch_load
from allosaurus.lm.unit import Unit


class TinyModel(torch.nn.Module):
    def __init__(self, phone_size):
        super().__init__()
        self.phone_layer = torch.nn.Linear(1, phone_size)


class FakeMask:
    def __init__(self):
        self.domain_unit = Unit({"<blk>": 0, "k": 1, "p": 2})
        self.target_unit = Unit({"<blk>": 0, "k": 1, "k͈": 2})
        self.unit_map = {0: 0, 1: 1, 2: 2}


class PhoneInitializationTest(unittest.TestCase):
    def test_explicit_mapping_overrides_automatic_source(self):
        with tempfile.TemporaryDirectory() as directory:
            source = TinyModel(3)
            with torch.no_grad():
                source.phone_layer.weight[:, 0] = torch.tensor([10.0, 20.0, 30.0])
                source.phone_layer.bias[:] = torch.tensor([1.0, 2.0, 3.0])
            path = Path(directory) / "source.pt"
            torch.save(source.state_dict(), path)

            target = TinyModel(3)
            torch_load(target, path, -1, FakeMask(), {"k͈": "k"})

            self.assertEqual(target.phone_layer.weight[2, 0].item(), 20.0)
            self.assertEqual(target.phone_layer.bias[2].item(), 2.0)


if __name__ == "__main__":
    unittest.main()
