"""Tests for train/test/train_celeba/test_celeba/test_image/CelebaDataset in example_utils.py."""

import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from FeDa4Fair.utils.example_utils import (
    CelebaDataset,
    ImageDataset,
    LinearClassificationNet,
    SimpleCNN,
    TabularDataset,
    evaluate_celeba,
    evaluate_image,
    evaluate_tabular,
    train,
    train_celeba,
)

# ---------------------------------------------------------------------------
# Minimal data helpers
# ---------------------------------------------------------------------------


def _make_tabular_loader(n: int = 16, input_size: int = 11, batch_size: int = 8) -> DataLoader:
    """Build a DataLoader whose batches match the (images, sex, mar, labels) convention in train/test."""
    x = np.random.default_rng(0).standard_normal((n, input_size)).astype(np.float32)
    sex = list(range(n))
    mar = list(range(n))
    y = [i % 2 for i in range(n)]
    ds = TabularDataset(x, sex, mar, y)
    return DataLoader(ds, batch_size=batch_size)


def _make_celeba_loader(n: int = 8, batch_size: int = 4) -> DataLoader:
    """Build a DataLoader for CelebA-style data: 64x64 RGB images, boolean labels."""
    images = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(n)]
    labels = [i % 2 == 0 for i in range(n)]
    sensitive = [i % 2 == 0 for i in range(n)]
    transform = transforms.Compose([transforms.ToTensor()])
    ds = CelebaDataset(images=images, labels=labels, sensitive_attributes=sensitive, transform=transform)
    return DataLoader(ds, batch_size=batch_size)


def _make_image_loader(n: int = 8, batch_size: int = 4) -> DataLoader:
    """Build a DataLoader for ImageDataset-style data: tensor images, int labels."""
    hf_ds = [{"image": np.zeros((64, 64, 3), dtype=np.uint8), "label": i % 2, "sensitive": i % 2} for i in range(n)]

    transform = transforms.Compose([transforms.ToTensor()])
    ds = ImageDataset(hf_ds, transform=transform)
    return DataLoader(ds, batch_size=batch_size)


# ---------------------------------------------------------------------------
# CelebaDataset tests
# ---------------------------------------------------------------------------


class TestCelebaDataset(unittest.TestCase):
    def _make_dataset(self, n: int = 10) -> CelebaDataset:
        images = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(n)]
        labels = [i % 2 == 0 for i in range(n)]
        sensitive = [i % 2 == 0 for i in range(n)]
        return CelebaDataset(images=images, labels=labels, sensitive_attributes=sensitive)

    def test_len(self):
        ds = self._make_dataset(10)
        self.assertEqual(len(ds), 10)

    def test_getitem_returns_three_elements(self):
        ds = self._make_dataset(4)
        item = ds[0]
        self.assertEqual(len(item), 3)

    def test_labels_converted_to_int(self):
        """Bool labels should be converted to 0/1 integers."""
        ds = self._make_dataset(4)
        _, _sensitive, label = ds[0]
        self.assertIsInstance(label, int)
        self.assertIn(label, [0, 1])

    def test_sensitive_attributes_converted_to_int(self):
        ds = self._make_dataset(4)
        _, sensitive, _ = ds[0]
        self.assertIsInstance(sensitive, int)
        self.assertIn(sensitive, [0, 1])

    def test_transform_is_applied(self):
        """If a transform is provided, the raw image should be transformed."""
        transform = transforms.Compose([transforms.ToTensor()])
        images = [np.zeros((32, 32, 3), dtype=np.uint8)]
        labels = [True]
        sensitive = [False]
        ds = CelebaDataset(images=images, labels=labels, sensitive_attributes=sensitive, transform=transform)
        img, _, _ = ds[0]
        self.assertIsInstance(img, torch.Tensor)

    def test_no_transform_returns_raw_image(self):
        images = [np.zeros((32, 32, 3), dtype=np.uint8)]
        labels = [True]
        sensitive = [False]
        ds = CelebaDataset(images=images, labels=labels, sensitive_attributes=sensitive)
        img, _, _ = ds[0]
        self.assertIsInstance(img, np.ndarray)


# Unit tests for the train and test functions


class TestTrainAndTest(unittest.TestCase):
    """Tests for the tabular train() and test() functions."""

    def setUp(self):
        self.net = LinearClassificationNet(input_size=11, output_size=2)
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.01)
        self.trainloader = _make_tabular_loader(n=16, input_size=11, batch_size=8)
        self.testloader = _make_tabular_loader(n=16, input_size=11, batch_size=8)

    def test_train_does_not_crash(self):
        """train() should run a forward+backward pass without error."""
        train(self.net, self.trainloader, self.optimizer, device="cpu")

    def test_train_updates_parameters(self):
        """Calling train() should change at least one parameter value."""
        params_before = [p.clone().detach() for p in self.net.parameters()]
        train(self.net, self.trainloader, self.optimizer, device="cpu")
        params_after = list(self.net.parameters())
        changed = any(
            not torch.equal(p_before, p_after) for p_before, p_after in zip(params_before, params_after, strict=False)
        )
        self.assertTrue(changed)

    def test_test_returns_correct_structure(self):
        """test() should return (loss: float, accuracy: float, unfairness_dict: dict)."""
        loss, accuracy, unfairness_dict = evaluate_tabular(self.net, self.testloader, device="cpu")
        self.assertIsInstance(loss, float)
        self.assertIsInstance(accuracy, float)
        self.assertIsInstance(unfairness_dict, dict)

    def test_test_accuracy_in_valid_range(self):
        _loss, accuracy, _ = evaluate_tabular(self.net, self.testloader, device="cpu")
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_test_unfairness_dict_contains_expected_keys(self):
        _, _, unfairness_dict = evaluate_tabular(self.net, self.testloader, device="cpu")
        self.assertIn("SEX_DP", unfairness_dict)
        self.assertIn("MAR_DP", unfairness_dict)
        self.assertIn("SEX_EO", unfairness_dict)
        self.assertIn("MAR_EO", unfairness_dict)


# ---------------------------------------------------------------------------
# train_celeba / evaluate_celeba
# ---------------------------------------------------------------------------


class TestTrainAndTestCeleba(unittest.TestCase):
    def setUp(self):
        self.net = SimpleCNN(num_classes=2)
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.01)
        self.loader = _make_celeba_loader(n=8, batch_size=4)

    def test_train_celeba_does_not_crash(self):
        train_celeba(self.net, self.loader, self.optimizer, device="cpu")

    def test_test_celeba_returns_correct_types(self):
        loss, accuracy, unfairness_dict = evaluate_celeba(self.net, self.loader, device="cpu")
        self.assertIsInstance(loss, float)
        self.assertIsInstance(accuracy, float)
        self.assertIsInstance(unfairness_dict, dict)

    def test_test_celeba_accuracy_in_valid_range(self):
        _, accuracy, _ = evaluate_celeba(self.net, self.loader, device="cpu")
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_test_celeba_unfairness_dict_has_sex_keys(self):
        _, _, unfairness_dict = evaluate_celeba(self.net, self.loader, device="cpu")
        self.assertIn("SEX_DP", unfairness_dict)
        self.assertIn("SEX_EO", unfairness_dict)


# ---------------------------------------------------------------------------
# evaluate_image
# ---------------------------------------------------------------------------


class TestTestImage(unittest.TestCase):
    def setUp(self):
        self.net = SimpleCNN(num_classes=2)
        self.loader = _make_image_loader(n=8, batch_size=4)

    def test_test_image_returns_correct_types(self):
        loss, accuracy, unfairness_dict = evaluate_image(self.net, self.loader, device="cpu")
        self.assertIsInstance(loss, float)
        self.assertIsInstance(accuracy, float)
        self.assertIsInstance(unfairness_dict, dict)

    def test_test_image_accuracy_in_valid_range(self):
        _, accuracy, _ = evaluate_image(self.net, self.loader, device="cpu")
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_test_image_unfairness_dict_has_sensitive_keys(self):
        _, _, unfairness_dict = evaluate_image(self.net, self.loader, device="cpu", sensitive_attribute_name="sensitive")
        self.assertIn("sensitive_DP", unfairness_dict)
        self.assertIn("sensitive_EO", unfairness_dict)


if __name__ == "__main__":
    unittest.main()
