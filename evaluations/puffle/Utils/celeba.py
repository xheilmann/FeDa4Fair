from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torchvision
from PIL import Image
from torch.utils.data import Dataset


class CelebaDataset(Dataset):
    """Definition of the dataset used for the Celeba Dataset."""

    def __init__(
        self,
        csv_path: str,
        image_path: str,
        transform: torchvision.transforms = None,
        debug: bool = True,
    ) -> None:
        """
        Initialization of the dataset.

        Args:
        ----
            csv_path (str): path of the csv file with all the information
             about the dataset
            image_path (str): path of the images
            transform (torchvision.transforms, optional): Transformation to apply
            to the images. Defaults to None.
            debug (bool, optional): If True, the images are loaded when needed.
            Defaults to True.

        """
        dataframe = pd.read_csv(csv_path)

        smiling_dict = {-1: 0, 1: 1}
        targets = [smiling_dict[item] for item in dataframe["Smiling"].tolist()]
        self.targets = targets
        self.sensitive_attributes = dataframe["Male"].tolist()
        self.samples = list(dataframe["image_id"])
        self.n_samples = len(dataframe)
        self.transform = transform
        self.image_path = Path(image_path)
        self.debug = debug
        self.indexes = range(len(self.samples))

        if not self.debug:
            self.images = [
                Image.open(self.image_path / sample).convert(
                    "RGB",
                )
                for sample in self.samples
            ]

    def __getitem__(self, index: int):
        """
        Returns a sample from the dataset.

        Args:
            index (int): index of the sample we want to retrieve

        Returns:
        -------
            _type_: sample we want to retrieve

        """
        img = Image.open(self.image_path / self.samples[index]).convert("RGB") if self.debug else self.images[index]

        if self.transform:
            img = self.transform(img)

        return (
            img,
            self.sensitive_attributes[index],
            self.targets[index],
        )

    def __len__(self) -> int:
        """
        This function returns the size of the dataset.

        Returns
        -------
            int: size of the dataset

        """
        return self.n_samples


class CelebaPreparedDataset(Dataset):
    """Definition of the dataset used for the Celeba Dataset."""

    def __init__(
        self,
        image_ids: list,
        image_dir: str | Path,
        labels: list,
        sensitive_attributes: list,
        second_sensitive_attributes: list | None = None,
        transform: Callable | None = None,
    ) -> None:
        """
        Initialization of the dataset.

        Args:
        ----
            image_ids (list): List of image IDs.
            image_dir (str | Path): Path to directory containing image files.
            labels (list): List of labels.
            sensitive_attributes (list): List of sensitive attributes.
            second_sensitive_attributes (list, optional): List of second sensitive attributes.
            transform (Callable | None, optional): Transformation to apply to the images. Defaults to None.

        """
        self.targets = [int(item) if item is not None else 0 for item in labels]

        # Safely map sensitive attributes if they match the dict, otherwise keep them
        self.sensitive_attributes = [int(item) if item is not None else 0 for item in sensitive_attributes]

        if second_sensitive_attributes is not None:
            self.second_sensitive_attributes = second_sensitive_attributes
        else:
            self.second_sensitive_attributes = [0] * len(image_ids)

        self.third_sensitive_attributes = [0] * len(image_ids)

        self.samples = image_ids
        self.image_dir = Path(image_dir)
        self.n_samples = len(image_ids)
        self.transform = transform
        self.indexes = range(len(self.samples))

    def __getitem__(self, index: int):
        """
        Returns a sample from the dataset.

        Args:
            index (int): index of the sample we want to retrieve

        Returns:
        -------
            _type_: sample we want to retrieve

        """
        img_id = str(self.samples[index])
        img_path = self.image_dir / f"{img_id}.png"

        if img_path.exists():
            img = Image.open(img_path).convert("RGB")
        else:
            # Fallback for missing images
            img = Image.new("RGB", (64, 64))

        if self.transform:
            img = self.transform(img)

        return (
            img,
            self.sensitive_attributes[index],
            self.second_sensitive_attributes[index],
            self.third_sensitive_attributes[index],
            self.targets[index],
        )

    def __len__(self) -> int:
        """
        This function returns the size of the dataset.

        Returns
        -------
            int: size of the dataset

        """
        return self.n_samples
