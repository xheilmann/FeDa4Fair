import unittest

import numpy as np
import pandas as pd
import torch
from torch import nn

from FeDa4Fair.utils.example_utils import (
    ImageDataset,
    SimpleCNN,
    TabularDataset,
    get_default_image_transform,
    get_params,
    pre_process_income,
    pre_process_single_datasets,
    set_params,
    weighted_average,
)


class TestExampleUtils(unittest.TestCase):
    def test_pre_process_income(self):
        df = pd.DataFrame(
            {
                "COW": [1, 2],
                "SCHL": [10, 20],
                "AGEP": [20, 30],
                "WKHP": [40, 50],
                "OCCP": [100, 200],
                "POBP": [1, 2],
                "RELP": [0, 1],
                "SEX": [1, 2],
                "PINCP": [0, 1],
            }
        )
        processed = pre_process_income(df)
        self.assertIn("COW_1", processed.columns)
        self.assertIn("SCHL_10", processed.columns)
        self.assertEqual(processed["AGEP"].min(), 0.0)
        self.assertEqual(processed["AGEP"].max(), 1.0)

    def test_pre_process_single_datasets(self):
        df = pd.DataFrame({">50K": [0, 1], "SEX": [1, 2], "MAR": [1, 3], "RAC1P": [1, 8], "feat1": [0.1, 0.2]})
        dfs, labels, groups, groups2, groups3 = pre_process_single_datasets(df)
        self.assertEqual(len(dfs), 1)
        self.assertEqual(dfs[0].shape, (2, 4))  # 4 features after dropping >50K
        self.assertEqual(labels[0].tolist(), [[0], [1]])
        self.assertEqual(groups[0].tolist(), [[1], [0]])  # 1 stays 1, others 0

    def test_tabular_dataset(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        z = [0, 1]
        w = [1, 0]
        y = [0, 1]
        ds = TabularDataset(x, z, w, y)
        self.assertEqual(len(ds), 2)
        x_s, z_s, w_s, y_s = ds[0]
        np.testing.assert_array_equal(x_s, x[0])
        self.assertEqual(z_s, z[0])
        self.assertEqual(y_s, y[0])

        ds.shuffle()
        self.assertEqual(len(ds), 2)

    def test_weighted_average(self):
        metrics = [(10, {"accuracy": 0.8}), (20, {"accuracy": 0.9})]
        result = weighted_average(metrics)
        # (10*0.8 + 20*0.9) / 30 = (8 + 18) / 30 = 26 / 30 = 0.8666...
        self.assertAlmostEqual(float(result["accuracy"]), 0.8666666, places=5)

    def test_params_utils(self):
        model = nn.Linear(2, 2)
        params = [np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([0.5, 0.6])]
        set_params(model, params)
        extracted = get_params(model)
        np.testing.assert_array_almost_equal(extracted[0], params[0])
        np.testing.assert_array_almost_equal(extracted[1], params[1])

    def test_simple_cnn(self):
        model = SimpleCNN(num_classes=2)
        x = torch.randn(1, 3, 64, 64)
        out = model(x)
        self.assertEqual(out.shape, (1, 2))

    def test_image_dataset(self):
        hf_ds = [{"image": np.zeros((64, 64, 3)), "label": 1, "sensitive": 0}]
        ds = ImageDataset(hf_ds)
        self.assertEqual(len(ds), 1)
        img, s1, s2, label = ds[0]
        self.assertEqual(label, 1)
        self.assertEqual(s1, 0)

    def test_get_default_image_transform(self):
        transform = get_default_image_transform()
        self.assertTrue(callable(transform))


if __name__ == "__main__":
    unittest.main()
