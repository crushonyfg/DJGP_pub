import math
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from djgp.jumpgp_bridge import center_linear_gate
from tests.smoke_utils import make_tiny_dataset


class JumpGPSmokeTest(unittest.TestCase):
    def test_center_linear_gate_preserves_jumpgp_logits(self):
        w = torch.tensor([0.4, -1.2, 0.7], dtype=torch.float64)
        z_anchor = torch.tensor([0.25, -0.5], dtype=torch.float64)
        z = torch.tensor([[0.1, 0.2], [0.8, -0.3]], dtype=torch.float64)

        centered = center_linear_gate(w, z_anchor)
        direct_logits = w[0] + z @ w[1:]
        centered_logits = centered[0] + (z - z_anchor) @ centered[1:]

        self.assertTrue(torch.allclose(direct_logits, centered_logits))

    def test_jumpgp_wrapper_accepts_logtheta_override(self):
        from shared.utils1 import jumpgp_ld_wrapper

        x = torch.linspace(-1.0, 1.0, 12, dtype=torch.float64).unsqueeze(1)
        x = torch.cat([x, torch.zeros_like(x)], dim=1)
        y = (x[:, :1] > 0).to(torch.float64) + 0.1 * x[:, :1]
        xt = torch.tensor([[0.15, 0.0]], dtype=torch.float64)
        logtheta = torch.tensor([0.0, 0.0, 0.0, -1.15], dtype=torch.float64)

        mu, sig2, model, _ = jumpgp_ld_wrapper(x, y, xt, "CEM", False, logtheta=logtheta)

        self.assertTrue(torch.isfinite(mu).all())
        self.assertTrue(torch.isfinite(sig2).all())
        self.assertIn("w", model)
        self.assertEqual(model["logtheta"].numel(), 4)

    def test_jumpgp_main_runs_on_tiny_dataset(self):
        import shared.jumpgp_runner as JumpGP_test

        with tempfile.TemporaryDirectory() as tmp:
            make_tiny_dataset(tmp, n_train=14, n_test=3, dim=3)
            argv = [
                "JumpGP_test.py",
                "--folder_name",
                tmp,
                "--M",
                "8",
                "--device",
                "cpu",
            ]
            with patch.object(sys, "argv", argv):
                result = JumpGP_test.main()

        self.assertEqual(len(result), 3)
        rmse, crps, runtime = result
        for value in (rmse, crps, runtime):
            value = value.detach().cpu().item() if isinstance(value, torch.Tensor) else float(value)
            self.assertTrue(math.isfinite(value))


if __name__ == "__main__":
    unittest.main()
