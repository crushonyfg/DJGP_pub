import math
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from tests.smoke_utils import make_tiny_dataset


class DJGPSmokeTest(unittest.TestCase):
    def test_djgp_main_runs_on_tiny_dataset(self):
        import shared.djgp_runner as DJGP_test

        with tempfile.TemporaryDirectory() as tmp:
            make_tiny_dataset(tmp, n_train=14, n_test=2, dim=3)
            argv = [
                "DJGP_test.py",
                "--folder_name",
                tmp,
                "--Q",
                "2",
                "--m1",
                "2",
                "--m2",
                "3",
                "--n",
                "8",
                "--num_steps",
                "1",
                "--MC_num",
                "1",
                "--lr",
                "0.005",
            ]
            with patch.object(sys, "argv", argv):
                result = DJGP_test.main()

        self.assertEqual(len(result), 3)
        rmse, crps, runtime = result
        for value in (rmse, crps, runtime):
            value = value.detach().cpu().item() if isinstance(value, torch.Tensor) else float(value)
            self.assertTrue(math.isfinite(value))


if __name__ == "__main__":
    unittest.main()
