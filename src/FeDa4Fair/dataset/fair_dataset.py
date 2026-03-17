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

import contextlib
import copy
import dataclasses
import datetime
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
from FeDa4Fair.utils.constants import (
    ACS_EMPLOYMENT,
    ACS_INCOME,
    CROSS_DEVICE,
    CROSS_SILO,
    DEFAULT_SEED,
    DEFAULT_SENSITIVE_ATTRIBUTES,
    DEFAULT_TRAIN_VAL_TEST_SPLIT,
    DEMOGRAPHIC_PARITY,
    EMPLOYMENT_LABEL,
    INCOME_LABEL,
    LEVEL_ATTRIBUTE,
)
from FeDa4Fair.utils.data_utils import balance_data, cap_samples, drop_data, flip_data

TRAIN_VAL_TEST_SPLIT_LEN = 3


def _clone_partitioner(obj: Any) -> Any:
    """
    Return a deep copy of a partitioner instance.

    Uses :func:`copy.deepcopy` so that all attributes — including those stored
    under private or differently-named names — are cloned correctly.  The
    returned instance has no ``dataset`` assigned yet and is therefore ready
    to be attached to a new split.
    """
    cloned = copy.deepcopy(obj)
    # Clear any already-assigned dataset so the clone is "fresh"
    if hasattr(cloned, "_dataset"):
        cloned._dataset = None
    elif hasattr(cloned, "dataset"):
        with contextlib.suppress(Exception):
            cloned.dataset = None
    return cloned


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

    year : Optional[int], default=2018
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
        dataset: str = ACS_INCOME,
        subset: str | None = None,
        preprocessor: Preprocessor | dict[str, tuple[str, ...]] | None = None,
        partitioners: dict[str, Partitioner | int],
        shuffle: bool = True,
        seed: int | None = DEFAULT_SEED,
        states: list[str] | None = None,
        year: str | None = "2018",
        horizon: str | None = "1-Year",
        sensitive_attributes: list[str] | None = None,
        fairness_level: Literal["attribute", "value", "attribute-value"] = LEVEL_ATTRIBUTE,
        fairness_metric: Literal["DP", "EO"] = DEMOGRAPHIC_PARITY,
        fl_setting: Literal["cross-silo", "cross-device"] | None = None,
        perc_train_val_test: list[float] | None = None,
        path: PathLike | None = None,
        modification_dict: dict[Any, dict[str, Any]] | None = None,
        mapping: dict[str, dict[int, int]] | None = None,
        label_name: str | None = None,
        preloaded_data: dict[str, pd.DataFrame] | None = None,
        client_names: list[str] | None = None,
        sample_cap: int | None = None,
        auto_evaluate: bool = False,
        auto_save: bool = True,
        **load_dataset_kwargs: Any,
    ) -> None:
        # Initialize this before super().__init__ because super().__init__ might call
        # _check_partitioners_correctness (which we override) or property partitioners
        # which depends on _dataset_prepared.
        self._dataset_prepared = False

        # Infer label for known datasets if not provided
        self._label = label_name
        if self._label is None:
            if dataset == ACS_INCOME:
                self._label = INCOME_LABEL
            elif dataset == ACS_EMPLOYMENT:
                self._label = EMPLOYMENT_LABEL

        # Initialize states only if using ACS datasets or if states are explicitly provided
        self._states = states
        if dataset in [ACS_INCOME, ACS_EMPLOYMENT] and self._states is None:
            self._states = self._get_default_us_states()

        # If partitioners is None, we need to defer its creation or set a default
        if partitioners is None:
            partitioners = dict.fromkeys(self._states, 1) if self._states else {"train": 1}

        # Handle 'split' if it's 'all' to trigger merging splits in prepare
        if load_dataset_kwargs.get("split") == "all":
            load_dataset_kwargs.pop("split")
            self._merge_splits = True
        else:
            self._merge_splits = False

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
        self._perc_train_test_split = perc_train_val_test if perc_train_val_test is not None else DEFAULT_TRAIN_VAL_TEST_SPLIT
        self._path = Path(path) if path else None
        self._modification_dict = modification_dict
        self._mapping = mapping
        self._preloaded_data = preloaded_data
        self._client_names = client_names
        self._total_removed_samples = 0
        self._sample_cap = sample_cap
        self.auto_evaluate = auto_evaluate
        self.auto_save = auto_save

        self._validate_modification_dict()

    @property
    def label_column(self) -> str:
        """Return the name of the label column."""
        if self._label is None:
            return ""
        return str(self._label)

    def _validate_modification_dict(self) -> None:
        """Validate the format of the modification_dict."""
        if self._modification_dict is None:
            return

        for key, mods in self._modification_dict.items():
            if not isinstance(key, (str, int)):
                msg = f"modification_dict keys must be strings or ints, got {type(key)}"
                raise TypeError(msg)
            if not isinstance(mods, dict):
                msg = f"modification_dict values must be dictionaries, got {type(mods)} for key {key}"
                raise TypeError(msg)
            for col_name, config in mods.items():
                if not isinstance(config, dict):
                    msg = f"modification config must be a dictionary, got {type(config)} for column {col_name}"
                    raise TypeError(msg)

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
                partition_df = cap_samples(partition_df, self._sample_cap, self.label_column, seed=self._seed or DEFAULT_SEED)
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
            except (KeyError, ValueError, TypeError) as e:
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

        base_path = Path(dataset_path)
        base_path.mkdir(parents=True, exist_ok=True)

        for key, value in self._partitioners.items():
            partitioner = value
            if isinstance(partitioner, int):
                continue

            num_partitions = partitioner.num_partitions
            for i in range(num_partitions):
                self._save_partition(base_path, key, i)

    def _warn_sensitive_attributes_saving(self) -> None:
        """Warn user if sensitive attributes are present in the data being saved."""
        if self._sensitive_attributes is not None:
            warnings.warn(
                "The data you are saving contains columns with sensitive attributes. "
                "If these should not be in the training data later, please remove them before training.",
                stacklevel=2,
            )

    def _save_partition(self, dataset_path: Path, key: str, partition_id: int) -> None:
        """Save a single partition to CSV."""
        partition = self.load_partition(partition_id, split=key)

        p_name = (
            self._client_names[partition_id]
            if self._client_names and partition_id < len(self._client_names)
            else str(partition_id)
        )
        file_name = f"{key}_{p_name}.csv"

        try:
            partition_df = partition.to_pandas()
            if isinstance(partition_df, pd.DataFrame):
                partition_df.to_csv(path_or_buf=dataset_path / file_name, index=False)
        except Exception as e: # noqa: BLE001
            warnings.warn(f"Partition {file_name} could not be saved to CSV: {e}", stacklevel=2)

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
        sens_cols = self._sensitive_attributes if self._sensitive_attributes else DEFAULT_SENSITIVE_ATTRIBUTES

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
            path=str(file) if file else "data_stats"
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

        has_validation = len(self._perc_train_test_split) == TRAIN_VAL_TEST_SPLIT_LEN

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
        if self._fl_setting == CROSS_SILO:
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
        elif self._fl_setting != CROSS_DEVICE:
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
        if self._dataset_name in [ACS_INCOME, ACS_EMPLOYMENT]:
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
        if self.auto_evaluate:
            try:
                # Check if evaluation is possible (requires label and sensitive attribute)
                if self._label and (self._sensitive_attributes or self._dataset_name in [ACS_INCOME, ACS_EMPLOYMENT]):
                    self.evaluate(self._path)
            except Exception as e:  # noqa: BLE001
                warnings.warn(f"Could not perform initial fairness evaluation: {e}", stacklevel=2)

        if self._sensitive_attributes is not None:
            warnings.warn(
                "Your current data contains columns with sensitive attributes. "
                "If these should not be in the training data later, please remove them before training.",
                stacklevel=2,
            )

        if self.auto_save and self._path is not None:
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
            if dataset_name == ACS_EMPLOYMENT:
                features, label, _group = ACSEmployment.df_to_pandas(acs_data)
            else:
                features, label, _group = ACSIncome.df_to_pandas(acs_data)

            return state, pd.concat([features, label], axis=1)

        results = Parallel(n_jobs=-1)(delayed(_load_state_raw)(state) for state in states)
        return dict(results)

    def _prepare_acs_dataset(self) -> None:
        """Helper to prepare ACSIncome/ACSEmployment datasets."""
        # Use _partitioners directly to avoid triggering recursion via partitioners property
        if self._states:
            for state in self._states:
                if state not in self._partitioners:
                    warnings.warn(f"State {state} not found in partitioners. Using default 1 partition.", stacklevel=2)
                    self._partitioners[state] = 1

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

        return self.load_acs_raw_data(self._dataset_name, self._states, self._year, self._horizon)

    def _apply_acs_modifications(self, state: str, data: pd.DataFrame) -> pd.DataFrame:
        """Apply mapping and other modifications to ACS data."""
        if self._mapping:
            for col, col_mapping in self._mapping.items():
                if col in data.columns:
                    data[col] = data[col].map(col_mapping).fillna(data[col])

        # Apply split-level modifications if any
        if self._modification_dict and state in self._modification_dict:
            data = self._modify_data(data, state)

        return data

    def _modify_data(self, data: pd.DataFrame, key: Any) -> pd.DataFrame:
        """Internal helper to apply modifications from modification_dict."""
        if self._modification_dict is None or key not in self._modification_dict:
            return data

        mods = self._modification_dict[key]
        for col, config in mods.items():
            if config.get("mitigate", False):
                data, removed = balance_data(data, col, self.label_column, seed=self._seed or DEFAULT_SEED)
                self._total_removed_samples += removed
            else:
                drop_rate = config.get("drop_rate", 0)
                flip_rate = config.get("flip_rate", 0)
                target_value = config.get("value")
                secondary_col = config.get("attribute")
                secondary_val = config.get("attribute_value")

                if drop_rate > 0:
                    data = drop_data(data, drop_rate, col, target_value, self.label_column, secondary_col, secondary_val, seed=self._seed or DEFAULT_SEED)

                if flip_rate > 0:
                    data = flip_data(data, flip_rate, col, target_value, self.label_column, secondary_col, secondary_val, seed=self._seed or DEFAULT_SEED)

        return data

    def _create_and_cast_dataset(self, data: pd.DataFrame) -> Dataset:
        """Create a Hugging Face Dataset from pandas and ensure correct types."""
        # Original implementation of _create_and_cast_dataset also handled ClassLabel casting
        # Checking if we should restore that to fix test_casts_label_to_classlabel_when_possible
        label_col = self._label
        if not label_col:
            if self._dataset_name == ACS_INCOME:
                label_col = INCOME_LABEL
            elif self._dataset_name == ACS_EMPLOYMENT:
                label_col = EMPLOYMENT_LABEL

        data_copy = data.copy()
        if label_col and label_col in data_copy.columns:
            try:
                unique_labels = sorted(data_copy[label_col].unique())
                if all(isinstance(x, (int, bool, np.integer, np.bool_)) for x in unique_labels) and len(unique_labels) <= 10:
                    # Convert to standard python types to help HF Dataset casting
                    data_copy[label_col] = data_copy[label_col].apply(lambda x: int(x) if not isinstance(x, bool) else x)
            except Exception: # noqa: BLE001
                pass

        ds = Dataset.from_pandas(data_copy)
        # Handle index level column if it exists
        if "__index_level_0__" in ds.column_names:
            ds = ds.remove_columns(["__index_level_0__"])

        if label_col and label_col in ds.column_names:
            try:
                unique_labels = sorted(data_copy[label_col].unique())
                if all(isinstance(x, (int, bool)) for x in unique_labels) and len(unique_labels) <= 10:
                    ds = ds.cast_column(label_col, ClassLabel(names=[str(x) for x in unique_labels]))
            except Exception: # noqa: BLE001
                pass

        return ds
    def _prepare_generic_dataset(self) -> None:
        """Prepare a generic Hugging Face dataset."""
        if self._preloaded_data is not None:
            self._dataset = DatasetDict()
            if isinstance(self._preloaded_data, pd.DataFrame):
                self._dataset["train"] = self._create_and_cast_dataset(self._preloaded_data)
            elif isinstance(self._preloaded_data, dict):
                for split_name, df in self._preloaded_data.items():
                    self._dataset[split_name] = self._create_and_cast_dataset(df)
            else:
                msg = f"Unexpected preloaded_data type: {type(self._preloaded_data)}"
                raise TypeError(msg)
        else:
            # Load dataset using standard HF load_dataset
            self._dataset = load_dataset(self._dataset_name, self._subset, **self._load_dataset_kwargs)

        if self._merge_splits and isinstance(self._dataset, DatasetDict):
            all_datasets = [self._dataset[split] for split in self._dataset]
            self._dataset = DatasetDict({"train": concatenate_datasets(all_datasets)})

        if not isinstance(self._dataset, DatasetDict):
            if isinstance(self._dataset, Dataset):
                self._dataset = DatasetDict({"train": self._dataset})
            else:
                msg = f"Unexpected dataset type: {type(self._dataset)}"
                raise TypeError(msg)

        # Apply modifications if specified
        if self._modification_dict:
            for split_name in list(self._dataset.keys()):
                if split_name in self._modification_dict:
                    df = self._dataset[split_name].to_pandas()
                    df = self._modify_data(df, split_name)
                    self._dataset[split_name] = self._create_and_cast_dataset(df)

    def _get_default_us_states(self) -> list[str]:
        """Return a list of all US state abbreviations."""
        return [
            "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
            "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
            "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
            "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
            "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "PR"
        ]

    def _check_partitioners_correctness(self) -> None:
        """Check if partitioners match the expected splits."""
        # Use _partitioners directly to avoid triggering recursion via partitioners property
        if self._states:
            for state in self._states:
                if state not in self._partitioners:
                    warnings.warn(f"State {state} not found in partitioners. Using default 1 partition.", stacklevel=2)
                    self._partitioners[state] = 1

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
