import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import os
import pandas as pd

# Add the evaluations/puffle directory to path to import models and utils
sys.path.append(os.path.abspath("evaluations/puffle"))

from Models.celeba_net import CelebaNet
from Utils.celeba import CelebaDataset
from torchvision import transforms


def train_single_client():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Configuration
    data_path = Path("datasets/celeba/cross_device_attribute/medium")
    img_dir = Path("datasets/celeba/images")
    csv_path = data_path / "train_0.csv"

    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return

    print(f"Loading dataset from {csv_path}")

    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset = CelebaDataset(
        csv_path=str(csv_path),
        image_path=str(img_dir),
        transform=transform,
        debug=True,  # debug=True means it loads images on the fly via __getitem__
    )

    # Check labels distribution
    from collections import Counter

    print(f"Label distribution: {Counter(dataset.targets)}")

    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = CelebaNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Starting training...")
    for epoch in range(1, 11):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, sens, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        print(f"Epoch {epoch:2d} | Loss: {running_loss / len(train_loader):.4f} | Acc: {100.0 * correct / total:.2f}%")

    # Final check: prediction distribution
    model.eval()
    all_preds = []
    with torch.no_grad():
        for images, _, labels in train_loader:
            outputs = model(images.to(device))
            all_preds.extend(outputs.argmax(1).cpu().tolist())

    print(f"Final Prediction Distribution: {Counter(all_preds)}")


if __name__ == "__main__":
    train_single_client()
