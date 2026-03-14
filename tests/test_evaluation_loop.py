"""
TDD tests for the _run_evaluation_loop helper (IMP-1).

These tests are written FIRST (red phase). Once _run_evaluation_loop is
implemented in example_utils.py, they should all pass (green phase).
"""

import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from FeDa4Fair.utils.example_utils import (
    LinearClassificationNet,
    TabularDataset,
    _run_evaluation_loop,
    test,
)


def _make_batch(n_samples: int = 8, input_size: int = 11) -> tuple:
    """Return a minimal (images, sex, mar, labels) batch as tensors."""
    x = torch.zeros(n_samples, input_size)
    sex = torch.zeros(n_samples, dtype=torch.long)
    mar = torch.zeros(n_samples, dtype=torch.long)
    labels = torch.zeros(n_samples, dtype=torch.long)
    return x, sex, mar, labels


class TestRunEvaluationLoop(unittest.TestCase):
    """Tests for the extracted _run_evaluation_loop helper."""

    def _make_net(self, input_size: int = 11) -> torch.nn.Module:
        return LinearClassificationNet(input_size=input_size, output_size=2)

    def _make_loader(self, n: int = 16, input_size: int = 11, batch_size: int = 8):
        """Build a DataLoader with tabular-style batches."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal((n, input_size)).astype("float32")
        sex = list(range(n))
        mar = list(range(n))
        y = [i % 2 for i in range(n)]
        ds = TabularDataset(x, sex, mar, y)
        return DataLoader(ds, batch_size=batch_size)

    def test_function_exists(self):
        """_run_evaluation_loop should be importable from example_utils."""
        self.assertTrue(callable(_run_evaluation_loop))

    def test_returns_correct_types(self):
        net = self._make_net()
        loader = self._make_loader()

        loss, accuracy, unfairness = _run_evaluation_loop(
            net=net,
            testloader=loader,
            device="cpu",
            sensitive_attrs={"SEX": 1, "MAR": 2},  # column indices in batch
        )

        self.assertIsInstance(loss, float)
        self.assertIsInstance(accuracy, float)
        self.assertIsInstance(unfairness, dict)

    def test_accuracy_in_valid_range(self):
        net = self._make_net()
        loader = self._make_loader()

        _, accuracy, _ = _run_evaluation_loop(
            net=net,
            testloader=loader,
            device="cpu",
            sensitive_attrs={"SEX": 1, "MAR": 2},
        )

        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_unfairness_dict_contains_dp_and_eo_per_attribute(self):
        """Each sensitive attr should produce DP and EO keys."""
        net = self._make_net()
        loader = self._make_loader()

        _, _, unfairness = _run_evaluation_loop(
            net=net,
            testloader=loader,
            device="cpu",
            sensitive_attrs={"SEX": 1, "MAR": 2},
        )

        self.assertIn("SEX_DP", unfairness)
        self.assertIn("SEX_EO", unfairness)
        self.assertIn("MAR_DP", unfairness)
        self.assertIn("MAR_EO", unfairness)

    def test_works_with_single_sensitive_attr(self):
        """Should work with just one sensitive attribute."""
        net = self._make_net()
        loader = self._make_loader()

        _, _, unfairness = _run_evaluation_loop(
            net=net,
            testloader=loader,
            device="cpu",
            sensitive_attrs={"SEX": 1},
        )

        self.assertIn("SEX_DP", unfairness)
        self.assertIn("SEX_EO", unfairness)
        self.assertNotIn("MAR_DP", unfairness)

    def test_net_is_set_to_eval_mode(self):
        """_run_evaluation_loop should call net.eval()."""
        net = self._make_net()
        loader = self._make_loader()

        _run_evaluation_loop(
            net=net,
            testloader=loader,
            device="cpu",
            sensitive_attrs={"SEX": 1},
        )

        # After the loop, the model should still be in eval mode
        self.assertFalse(net.training)

    def test_output_is_consistent_with_test_function(self):
        """_run_evaluation_loop output should match the old test() for SEX+MAR."""
        net = self._make_net()
        loader = self._make_loader(n=16, input_size=11, batch_size=16)

        # Both should produce the same numbers
        loss_old, acc_old, _ = test(net, loader, device="cpu")
        # Reset the model to the same state for fair comparison
        loss_new, acc_new, _ = _run_evaluation_loop(
            net=net,
            testloader=loader,
            device="cpu",
            sensitive_attrs={"SEX": 1, "MAR": 2},
        )

        self.assertAlmostEqual(loss_old, loss_new, places=5)
        self.assertAlmostEqual(acc_old, acc_new, places=5)


if __name__ == "__main__":
    unittest.main()
