# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Implements FairFederatedDataset as subclass of FederatedDataset."""

import dataclasses
import datetime
import inspect
import json
import warnings
from os import PathLike
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import Partitioner
from flwr_datasets.preprocessor import Divider, Preprocessor
from folktables import ACSDataSource, ACSEmployment, ACSIncome
from joblib import Parallel, delayed

from datasets import ClassLabel, Dataset, DatasetDict, concatenate_datasets, load_dataset
from FeDa4Fair.metrics.evaluation import evaluate_fairness
from FeDa4Fair.utils.data_utils import balance_data, cap_samples, drop_data, flip_data


def _clone_partitioner(obj: Any) -> Any:
    """
    Creates a new instance of the same class as obj with the same arguments.

    Assumes that arguments to __init__ are stored as attributes in obj.
    """
    cls = obj.__class__
    init_signature = inspect.signature(cls.__init__)
    arg_names = [param for param in init_signature.parameters if param != "self"]
    init_args = {arg: getattr(obj, arg) for arg in arg_names if hasattr(obj, arg)}
    return cls(**init_args)


class FairFederatedDataset(FederatedDataset):
    """
    Representation of a dataset designed for federated learning, fairness evaluation, and analytics.

    Supports downloading, loading, preprocessing, modifying, evaluating, mapping and partitioning
    the dataset across multiple clients (e.g., edge devices or simulated silos).

    If `sens_cols` are provided:
        - If two attributes are provided, intersectional fairness is evaluated.
        - If not provided, fairness is evaluated using default attributes for ACS datasets:
          "SEX", "MAR", and "RAC1P".

    Parameters
    ----------
    dataset : str, default="ACSIncome"
        The name of the dataset to load. Supports "ACSIncome", "ACSEmployment" (via folktables),
        or any Hugging Face dataset name (e.g., "celeba", "mnist").

    subset : Optional[str], default=None
        Optional dataset subset to load (e.g., a specific configuration of a HF dataset).

    preprocessor : Optional[Union[Preprocessor, dict[str, tuple[str, ...]]]], default=None
        A callable or configuration dictionary used to apply transformations on the dataset.

    partitioners : dict[str, Union[Partitioner, int]]
        Dictionary mapping dataset splits (e.g., state names or 'train') to partitioning strategies.
        Each split can use a custom `Partitioner` or an integer specifying the number of IID partitions.

    shuffle : bool, default=True
        Whether to shuffle the dataset before preprocessing and partitioning.

    seed : Optional[int], default=42
        Seed for reproducible shuffling. If `None`, a random seed is used.

    states : Optional[list[str]], default=None
        List of states to include for ACS datasets. If `None` and using ACS, defaults to all US states.
        For non-ACS datasets, this can be used to filter specific splits if applicable.

    year : Optional[str], default="2018"
        The ACS year to load (for ACS datasets).

    horizon : Optional[str], default="1-Year"
        Horizon of the ACS sample (for ACS datasets).

    sensitive_attributes : Optional[list[str]], default=None
        List of attributes used to evaluate intersectional fairness.

    fairness_level : Literal["attribute", "value", "attribute-value"], default="attribute"
        The level at which fairness is evaluated.

    fairness_metric : Literal["DP", "EO"], default="DP"
        Fairness metric to evaluate (Demographic Parity or Equalized Odds).

    fl_setting : Literal["cross-silo", "cross-device", None], default=None
        Strategy used to split the dataset into train/test:
        - "cross-silo": splits each client's dataset into train/val/test.
        - "cross-device": typically assumes existing train/test splits.

    perc_train_val_test : Optional[list[float]], default=None
        Proportions for train, validation, and test sets in the cross-silo setting.
        Defaults to [0.7, 0.15, 0.15] if None.

    path : Optional[PathLike], default=None
        Optional path where the dataset should be saved.

    modification_dict : Optional[dict[Any, dict[str, ...]]], default=None
        Optional dictionary to apply data modifications to specific splits or partitions.
        Keys can be split names (e.g., "train", "AK") or partition indices/names.

    mapping : Optional[dict[str, dict[int, int]]], default=None
        Optional remapping dictionary of categorical features or labels.

    label_name : Optional[str], default=None
        The name of the target label column. If None, it is inferred for known datasets (ACS).
        Required for generic Hugging Face datasets if not standard.

    client_names : Optional[list[str]], default=None
        Optional list of names for partitions. If provided, these names can be used as keys in
        `modification_dict` and for naming saved files.

    **load_dataset_kwargs : dict
        Additional keyword arguments passed to `datasets.load_dataset`.

    """

    def __init__(
        self,
        *,
        dataset: str = "ACSIncome",
        subset: str | None = None,
        preprocessor: Preprocessor | dict[str, tuple[str, ...]] | None = None,
        partitioners: dict[str, Partitioner | int],
        shuffle: bool = True,
        seed: int | None = 42,
        states: list[str] | None = None,
        year: str | None = "2018",
        horizon: str | None = "1-Year",
        sensitive_attributes: list[str] | None = None,
        fairness_level: Literal["attribute", "value", "attribute-value"] = "attribute",
        fairness_metric: Literal["DP", "EO"] = "DP",
        fl_setting: Literal["cross-silo", "cross-device", None] = None,
        perc_train_val_test: list[float] | None = None,
        path: PathLike | None = None,
        modification_dict: dict[Any, dict[str, Any]] | None = None,
        mapping: dict[str, dict[int, int]] | None = None,
        label_name: str | None = None,
        preloaded_data: dict[str, pd.DataFrame] | None = None,
        client_names: list[str] | None = None,
        sample_cap: int | None = None,
        **load_dataset_kwargs: Any,
    ) -> None:
        # Initialize states only if using ACS datasets or if states are explicitly provided
        self._states = states
        if dataset in ["ACSIncome", "ACSEmployment"] and self._states is None:
            self._states = self._get_default_us_states()

        # If partitioners is None, we need to defer its creation or set a default
        if partitioners is None:
            partitioners = dict.fromkeys(self._states, 1) if self._states else {"train": 1}

        super().__init__(
            dataset=dataset,
            subset=subset,
            preprocessor=preprocessor,
            partitioners=partitioners,
            shuffle=shuffle,
            seed=seed,
            **load_dataset_kwargs,
        )

        self._year = year
        self._horizon = horizon
        self._sensitive_attributes = sensitive_attributes
        self._fairness_level = fairness_level
        self._fairness_metric = fairness_metric
        self._fl_setting = fl_setting
        self._perc_train_test_split = perc_train_val_test if perc_train_val_test is not None else [0.7, 0.15, 0.15]
        self._path = path
        self._modification_dict = modification_dict
        self._mapping = mapping
        self._label = label_name
        self._preloaded_data = preloaded_data
        self._client_names = client_names
        self._total_removed_samples = 0
        self._sample_cap = sample_cap

        # Infer label for known datasets if not provided
        if self._label is None:
            if dataset == "ACSIncome":
                self._label = "PINCP"
            elif dataset == "ACSEmployment":
                self._label = "ESR"

    @property
    def label_column(self) -> str:
        """Return the name of the label column."""
        if self._label is None:
            return ""
        return str(self._label)

    def prepare(self) -> None:
        """Explicitly trigger dataset preparation."""
        if not self._dataset_prepared:
            self._prepare_dataset()

    def load_partition(self, partition_id: int, split: str | None = None) -> Dataset:
        """
        Load a partition and apply modifications if specified.
        """
        if split is None:
            split = next(iter(self.partitioners.keys()))

        partition = super().load_partition(partition_id, split)

        # Apply sample cap if specified
        if self._sample_cap is not None:
            partition_df = partition.to_pandas()
            if isinstance(partition_df, pd.DataFrame):
                partition_df = cap_samples(partition_df, self._sample_cap, self.label_column, seed=self._seed or 42)
                partition = Dataset.from_pandas(partition_df)

        return self._apply_modification_to_partition(partition, partition_id, split)

    def _apply_modification_to_partition(self, partition: Dataset, partition_id: int, split: str) -> Dataset:
        """
        Apply modifications to a single partition.
        """
        if self._modification_dict is None:
            return partition

        # Determine the key to look for in modification_dict
        mod_key = None
        if self._client_names and partition_id < len(self._client_names):
            mod_key = self._client_names[partition_id]

        # If mod_key is not found or not provided, try partition_id
        if (mod_key is None or mod_key not in self._modification_dict) and partition_id in self._modification_dict:
            mod_key = partition_id

        if mod_key is not None and mod_key in self._modification_dict:
            try:
                partition_df = partition.to_pandas()
                if isinstance(partition_df, pd.DataFrame):
                    partition_df = self._modify_data(partition_df, mod_key)
                    return Dataset.from_pandas(partition_df)
            except Exception as e:  # noqa: BLE001
                warnings.warn(
                    f"Could not apply modification to partition {partition_id} ({mod_key}): {e}", stacklevel=2
                )

        return partition

    def save_dataset(self, dataset_path: PathLike) -> None:
        """
        Save the dataset to disk as csv files with names by split and partition index/name.
        """
        if not self._dataset_prepared:
            self._prepare_dataset()

        self._warn_sensitive_attributes_saving()

        for key, value in self._partitioners.items():
            partitioner = value
            if isinstance(partitioner, int):
                continue

            num_partitions = partitioner.num_partitions
            for i in range(num_partitions):
                self._save_partition(dataset_path, key, i)

    def _warn_sensitive_attributes_saving(self) -> None:
        """Warn user if sensitive attributes are present in the data being saved."""
        if self._sensitive_attributes is not None:
            warnings.warn(
                "The data you are saving contains columns with sensitive attributes. "
                "If these should not be in the training data later, please remove them before training.",
                stacklevel=2,
            )

    def _save_partition(self, dataset_path: PathLike, key: str, partition_id: int) -> None:
        """Save a single partition to CSV."""
        partition = self.load_partition(partition_id, split=key)
        Path(str(dataset_path)).mkdir(parents=True, exist_ok=True)

        p_name = (
            self._client_names[partition_id]
            if self._client_names and partition_id < len(self._client_names)
            else str(partition_id)
        )
        file_name = f"{key}_{p_name}.csv"

        try:
            partition_df = partition.to_pandas()
            if isinstance(partition_df, pd.DataFrame):
                partition_df.to_csv(path_or_buf=f"{dataset_path}/{file_name}", index=False)
        except Exception:  # noqa: BLE001
            warnings.warn(f"Partition {file_name} could not be saved to CSV (likely non-tabular).", stacklevel=2)

    def evaluate(self, file: PathLike | None = None) -> None:
        """
        Evaluate fairness on all partitions.
        """
        if not self._dataset_prepared:
            self._prepare_dataset()
        if self._dataset is None:
            return
        titles = list(self._dataset.keys())

        # Determine sensitive columns to evaluate
        sens_cols = self._sensitive_attributes if self._sensitive_attributes else ["SEX", "MAR", "RAC1P"]

        # Only set intersectional_fairness if we have more than one attribute
        intersectional = (
            self._sensitive_attributes if (self._sensitive_attributes and len(self._sensitive_attributes) > 1) else None
        )

        evaluate_fairness(
            partitioner_dict=self.partitioners,
            max_num_partitions=None,
            fairness_metric=self._fairness_metric,
            fairness_level=self._fairness_level,
            titles=titles,
            legend=True,
            label_name=self.label_column,
            sens_columns=sens_cols,
            intersectional_fairness=intersectional,
        )

    def _split_into_train_val_test(self) -> None:
        """
        If cross-silo setting is chosen, splits the dataset into train, test and optionally validation sets.
        """
        divider_dict = {}
        partitioners_dict = {}

        # Check if keys exist before trying to split
        if self._dataset is None:
            return
        keys_to_process = list(self._dataset.keys())

        has_validation = len(self._perc_train_test_split) == 3

        for entry in keys_to_process:
            if has_validation:
                divider_dict[entry] = {
                    f"{entry}_train": self._perc_train_test_split[0],
                    f"{entry}_val": self._perc_train_test_split[1],
                    f"{entry}_test": self._perc_train_test_split[2],
                }
            else:
                divider_dict[entry] = {
                    f"{entry}_train": self._perc_train_test_split[0],
                    f"{entry}_test": self._perc_train_test_split[1],
                }

        divider = Divider(divide_config=divider_dict)
        if self._fl_setting == "cross-silo":
            for entry in keys_to_process:
                # We need to ensure the partitioner for this entry exists
                original_partitioner = self._partitioners.get(entry)
                if original_partitioner:
                    partitioners_dict[f"{entry}_train"] = original_partitioner
                    if has_validation:
                        partitioners_dict[f"{entry}_val"] = _clone_partitioner(original_partitioner)
                    partitioners_dict[f"{entry}_test"] = _clone_partitioner(original_partitioner)

            if self._dataset is not None:
                self._dataset = divider(self._dataset)
            self._partitioners = partitioners_dict
        elif self._fl_setting != "cross-device":
            # If it is None, we do nothing. If it is something else, error?
            if self._fl_setting is not None:
                msg = "This train-val-test split strategy is not supported."
                raise ValueError(msg)

        self._event = {
            "load_partition": dict.fromkeys(self._partitioners, False),
        }

    def _prepare_dataset(self) -> None:
        """
        Prepare the dataset by downloading, shuffling, preprocessing, and modifying.
        """
        # Case 1: Special handling for ACS datasets (Folktables)
        if self._dataset_name in ["ACSIncome", "ACSEmployment"]:
            self._prepare_acs_dataset()
        else:
            # Case 2: Generic Hugging Face Dataset
            self._prepare_generic_dataset()

        # Common post-processing
        if self._shuffle and self._dataset is not None:
            self._dataset = self._dataset.shuffle(seed=self._seed)

        if self._preprocessor and self._dataset is not None:
            self._dataset = self._preprocessor(self._dataset)

        if self._fl_setting is not None:
            self._split_into_train_val_test()

        self._dataset_prepared = True
        if self._dataset is not None:
            available_splits = list(self._dataset.keys())
            self._event["load_split"] = dict.fromkeys(available_splits, False)

        # Run initial evaluation if needed (and if tabular/compatible)
        try:
            # Check if evaluation is possible (requires label and sensitive attribute)
            if self._label and (self._sensitive_attributes or self._dataset_name in ["ACSIncome", "ACSEmployment"]):
                self.evaluate(self._path)
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"Could not perform initial fairness evaluation: {e}", stacklevel=2)

        if self._sensitive_attributes is not None:
            warnings.warn(
                "Your current data contains columns with sensitive attributes. "
                "If these should not be in the training data later, please remove them before training.",
                stacklevel=2,
            )

        if self._path is not None:
            self.save_dataset(self._path)

    @staticmethod
    def load_acs_raw_data(dataset_name: str, states: list[str], year: str, horizon: str) -> dict[str, pd.DataFrame]:
        """
        Load raw ACS data for the given states in parallel.
        Returns a dictionary mapping state codes to pandas DataFrames.
        """

        def _load_state_raw(state):
            data_source = ACSDataSource(survey_year=year, horizon=horizon, survey="person")
            acs_data = data_source.get_data(states=[state], download=True)
            if dataset_name == "ACSEmployment":
                features, label, _group = ACSEmployment.df_to_pandas(acs_data)
            else:
                features, label, _group = ACSIncome.df_to_pandas(acs_data)

            return state, pd.concat([features, label], axis=1)

        results = Parallel(n_jobs=-1)(delayed(_load_state_raw)(state) for state in states)
        return dict(results)

    def _prepare_acs_dataset(self) -> None:
        """Helper to prepare ACSIncome/ACSEmployment datasets."""
        self._check_partitioners_correctness()
        raw_data_dict = self._load_raw_data()
        self._dataset = DatasetDict()

        for state, state_data in raw_data_dict.items():
            data_to_use = state_data.copy()
            data_to_use = self._apply_acs_modifications(state, data_to_use)
            self._dataset[state] = self._create_and_cast_dataset(data_to_use)

    def _load_raw_data(self) -> dict[str, pd.DataFrame]:
        """Load raw ACS data from preloaded data or by downloading."""
        if self._preloaded_data is not None:
            return self._preloaded_data
        if self._states is not None and self._year is not None and self._horizon is not None:
            return self.load_acs_raw_data(self._dataset_name, self._states, self._year, self._horizon)
        return {}

    def _apply_acs_modifications(self, state: str, data: pd.DataFrame) -> pd.DataFrame:
        """Apply mappings and modifications to ACS data."""
        if self._mapping is not None:
            for key, value in self._mapping.items():
                if key in data.columns:
                    data[key] = data[key].replace(value)

        if self._modification_dict is not None and state in self._modification_dict:
            modifications = self._modification_dict[state]
            for col_name, config in modifications.items():
                if config.get("mitigate", False):
                    data, removed = balance_data(data, col_name, self.label_column)
                    self._total_removed_samples += removed
                else:
                    data = self._apply_single_modification(data, col_name, config)
        return data

    def _apply_single_modification(self, data: pd.DataFrame, col_name: str, config: dict) -> pd.DataFrame:
        """Apply a single drop or flip modification."""
        drop_rate = config.get("drop_rate", 0)
        flip_rate = config.get("flip_rate", 0)
        value1 = config.get("value")
        column2 = config.get("attribute")
        value2 = config.get("attribute_value")

        if drop_rate > 0:
            data = drop_data(data, drop_rate, col_name, value1, self.label_column, column2, value2)
        if flip_rate > 0:
            data = flip_data(data, flip_rate, col_name, value1, self.label_column, column2, value2)
        return data

    def _create_and_cast_dataset(self, data: pd.DataFrame) -> Dataset:
        """Create a Dataset from pandas DataFrame and cast label to ClassLabel."""
        try:
            unique_labels = sorted(data[self._label].unique())
            num_classes = len(unique_labels)
            label_to_id = {label: i for i, label in enumerate(unique_labels)}
            data[self._label] = data[self._label].map(label_to_id)

            ds = Dataset.from_pandas(data)
            return ds.cast_column(
                self._label,
                ClassLabel(num_classes=num_classes, names=[str(label_val) for label_val in unique_labels]),
            )
        except (ValueError, TypeError, KeyError):
            return Dataset.from_pandas(data)

    def _prepare_generic_dataset(self) -> None:
        """Helper to prepare generic Hugging Face datasets."""
        # Load dataset using Hugging Face datasets library

        # Check if we need to load all splits and merge them
        if self._load_dataset_kwargs.get("split") == "all":
            loaded_data = self._load_and_merge_all_splits()
        else:
            loaded_data = load_dataset(self._dataset_name, **self._load_dataset_kwargs)

        if isinstance(loaded_data, Dataset):
            # If a single split is returned, wrap it in a DatasetDict
            split_name = self._load_dataset_kwargs.get("split", "train")
            self._dataset = DatasetDict({str(split_name): loaded_data})
        elif isinstance(loaded_data, DatasetDict):
            self._dataset = loaded_data
        else:
            msg = f"Unsupported return type from load_dataset: {type(loaded_data)}"
            raise TypeError(msg)

        # Apply modifications if specified (assuming tabular/pandas compatible for now)
        if self._modification_dict:
            for split_name in self._dataset:
                if split_name in self._modification_dict:
                    try:
                        split_df = self._dataset[split_name].to_pandas()
                        if isinstance(split_df, pd.DataFrame):
                            split_df = self._modify_data(split_df, split_name)
                            self._dataset[split_name] = Dataset.from_pandas(split_df)
                    except Exception as e:  # noqa: BLE001
                        warnings.warn(f"Could not apply modification to split {split_name}: {e}", stacklevel=2)

    def _load_and_merge_all_splits(self) -> DatasetDict:
        """Load all splits and concatenate them into a single 'train' split."""
        # Remove \"split\" from kwargs to let load_dataset load all splits as a DatasetDict
        load_kwargs = self._load_dataset_kwargs.copy()
        load_kwargs.pop("split", None)

        loaded_data = load_dataset(self._dataset_name, **load_kwargs)

        if isinstance(loaded_data, DatasetDict):
            # Concatenate all splits
            merged_dataset = concatenate_datasets(list(loaded_data.values()))
            # Assign to a default key \"train\" as we are working on a single merged dataset
            return DatasetDict({"train": merged_dataset})
        if isinstance(loaded_data, Dataset):
            # Should not happen if split is removed, but handle just in case
            return DatasetDict({"train": loaded_data})

        msg = f"Unsupported return type from load_dataset: {type(loaded_data)}"
        raise TypeError(msg)

    def _get_default_us_states(self) -> list[str]:
        return [
            "AL",
            "AK",
            "AZ",
            "AR",
            "CA",
            "CO",
            "CT",
            "DE",
            "FL",
            "GA",
            "HI",
            "ID",
            "IL",
            "IN",
            "IA",
            "KS",
            "KY",
            "LA",
            "ME",
            "MD",
            "MA",
            "MI",
            "MN",
            "MS",
            "MO",
            "MT",
            "NE",
            "NV",
            "NH",
            "NJ",
            "NM",
            "NY",
            "NC",
            "ND",
            "OH",
            "OK",
            "OR",
            "PA",
            "RI",
            "SC",
            "SD",
            "TN",
            "TX",
            "UT",
            "VT",
            "VA",
            "WA",
            "WV",
            "WI",
            "WY",
            "PR",
        ]

    def _modify_data(self, data: pd.DataFrame, key: Any) -> pd.DataFrame:
        """
        Modifies a pandas dataframe representing a split/state
        according to the values given in _modification_dict.
        """
        if self._modification_dict is None or key not in self._modification_dict:
            return data

        modifications = self._modification_dict[key]

        # Correct logic based on original implementation:
        for col_name, config in modifications.items():
            if config.get("mitigate", False):
                data, removed = balance_data(data, col_name, self.label_column)
                self._total_removed_samples += removed
            else:
                drop_rate = config.get("drop_rate", 0)
                flip_rate = config.get("flip_rate", 0)
                value1 = config.get("value")
                column2 = config.get("attribute")
                value2 = config.get("attribute_value")

                if drop_rate > 0:
                    data = drop_data(data, drop_rate, col_name, value1, self.label_column, column2, value2)
                if flip_rate > 0:
                    data = flip_data(data, flip_rate, col_name, value1, self.label_column, column2, value2)

        return data

    def to_json(self, **json_kw: Any) -> str:
        """
        Returns the dataset as a JSON string.
        """

        def _default(o: Any) -> Any:
            if dataclasses.is_dataclass(o):
                return dataclasses.asdict(o)
            if isinstance(o, (datetime.datetime, datetime.date, datetime.time)):
                return o.isoformat()
            if isinstance(o, Path):
                return str(o)
            if hasattr(o, "__dict__"):
                return o.__dict__
            return str(o)

        return json.dumps(self, default=_default, **json_kw)
