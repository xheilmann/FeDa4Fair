import csv
from pathlib import Path

import numpy as np
from PIL import Image


def create_mock_celeba(base_dir="data/celeba_mock"):
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    # Create metadata file
    metadata_path = base_path / "metadata.csv"

    rng = np.random.default_rng()

    with metadata_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "label", "sensitive"])  # label: Smiling, sensitive: Male

        for i in range(20):
            file_name = f"img_{i}.png"
            # Random label and sensitive attribute
            label = rng.integers(0, 2)
            sensitive = rng.integers(0, 2)

            writer.writerow([file_name, label, sensitive])

            # Create dummy image
            # Random noise image
            img_data = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
            img = Image.fromarray(img_data)
            img.save(base_path / file_name)


if __name__ == "__main__":
    create_mock_celeba()
    print("Mock CelebA dataset created.")
