import base64
import io
import os
from typing import Callable

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

        """
        dataframe = pd.read_csv(csv_path)

        smiling_dict = {-1: 0, 1: 1}
        targets = [smiling_dict[item] for item in dataframe["Smiling"].tolist()]
        self.targets = targets
        # self.sensitive_attributes = [smiling_dict[item] for item in dataframe["Gender"].tolist()]
        self.sensitive_attributes = dataframe["Male"].tolist()
        self.samples = list(dataframe["image_id"])
        self.n_samples = len(dataframe)
        self.transform = transform
        self.image_path = image_path
        self.debug = debug
        self.indexes = range(len(self.samples))

        if not self.debug:
            self.images = [
                Image.open(os.path.join(self.image_path, sample)).convert(
                    "RGB",
                )
                for sample in self.samples
            ]

    def __getitem__(self, index: int):
        """
        Returns a sample from the dataset.

        Args:
            idx (_type_): index of the sample we want to retrieve

        Returns:
        -------
            _type_: sample we want to retrieve

        """
        if self.debug:
            img = Image.open(
                os.path.join(self.image_path, self.samples[index])
            ).convert(
                "RGB",
            )
        else:
            img = self.images[index]

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
        images_dict: dict,
        labels: list,
        sensitive_attributes: list,
        transform: Callable | None = None,
    ) -> None:
        """
        Initialization of the dataset.

        Args:
        ----
            image_ids (list): List of image IDs.
            images_dict (dict): Dictionary mapping image IDs to base64 encoded images.
            labels (list): List of labels.
            sensitive_attributes (list): List of sensitive attributes.
            transform (Callable | None, optional): Transformation to apply to the images. Defaults to None.

        """
        smiling_dict = {False: 0, True: 1}
        targets = [smiling_dict[item] for item in labels]
        self.targets = targets
        self.sensitive_attributes = [smiling_dict[item] for item in sensitive_attributes]
        self.samples = image_ids
        self.images_dict = images_dict
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
        
        if img_id in self.images_dict:
            b64_str = self.images_dict[img_id]
            img_bytes = base64.b64decode(b64_str)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        else:
            # Fallback for missing images
            img = Image.new("RGB", (64, 64))

        if self.transform:
            img = self.transform(img)

        return (
            img,
            self.sensitive_attributes[index],
            self.sensitive_attributes[index],
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
