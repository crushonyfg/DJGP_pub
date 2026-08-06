"""Import sanity check for the released package and benchmark entry points.

Run after ``pip install -r requirements.txt && pip install -e .`` (from the repo root):
    python -m pytest tests/test_imports.py -v
"""
import importlib
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

# Public model surface + data generators + benchmark entry points. Importing the two
# runners exercises the whole benchmark dependency chain; none of these performs any
# network access at import time (UCI fetching is deferred to prepare_uci_data.py).
RELEASE_MODULES = [
    "djgp.model",
    "djgp.projections.structured_metric_lmjgp",
    "djgp.evaluation.crps",
    "djgp.evaluation.calibration",
    "jumpgp",
    "data_gen.synthetic",
    "data_gen.highdata",
    "data_gen.highdata_utils",
    "run_synthetic",
    "run_uci",
    "scripts.prepare_uci_data",
]


class ReleaseImportTest(unittest.TestCase):
    def test_release_modules_import(self):
        for module_name in RELEASE_MODULES:
            with self.subTest(module=module_name):
                importlib.import_module(module_name)

    def test_public_api_smoke(self):
        from djgp.model import DJGP
        model = DJGP(seed=0)
        self.assertTrue(model.uniform_outlier)  # paper's uniform outlier is the default


if __name__ == "__main__":
    unittest.main()
