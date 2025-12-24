import os
import sys

import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from FeDa4Fair.dataset.fair_dataset import FairFederatedDataset
from FeDa4Fair.utils.example_utils import ImageDataset, SimpleCNN, get_default_image_transform, test_image, train


def main():
    # Path to the mock dataset. Run `python examples/generate_mock_data.py` to generate it.
    data_dir = os.path.abspath("data/celeba_mock")

    print(f"Loading local image dataset from {data_dir}...")

    # 1. Initialize Dataset
    # We use "imagefolder" builder from Hugging Face
    fds = FairFederatedDataset(
        dataset="imagefolder",
        data_dir=data_dir,
        partitioners={"train": 2},  # Split into 2 clients
        label_name="label",
        sensitive_attributes=["sensitive"],
        fairness_metric="DP",
    )

    print("Preparing dataset...")
    fds.prepare()

    # 2. Load a partition (Client 0)
    client_partition = fds.load_partition(0, split="train")
    print(f"Client 0 size: {len(client_partition)}")

    # 3. Create PyTorch Dataset
    # We use the transformer from example_utils
    transform = get_default_image_transform()

    train_dataset = ImageDataset(client_partition, transform=transform, label_key="label", sensitive_key="sensitive")

    # Create DataLoader
    trainloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    # 4. Initialize Model
    model = SimpleCNN(num_classes=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # 5. Train
    print("Training model on Client 0 data...")
    train(model, trainloader, optimizer, device="cpu")
    print("Training finished.")

    # 6. Evaluate (using the same data for demo purposes, usually use test set)
    print("Evaluating...")
    loss, accuracy, fairness = test_image(model, trainloader, device="cpu", sensitive_attribute_name="sensitive")

    print(f"Accuracy: {accuracy}")
    print(f"Fairness: {fairness}")


if __name__ == "__main__":
    main()
