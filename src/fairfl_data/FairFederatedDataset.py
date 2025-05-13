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

"""This file implements FairFederatedDataset as subclass of FederatedDataset (https://flower.ai/docs/datasets/ref-api/flwr_datasets.FederatedDataset.html)"""

import dataclasses
import datetime
import json
from pathlib import Path
import inspect
import warnings
from os import PathLike
from typing import Any, Optional, Union, Literal
import pandas as pd
from datasets import Dataset, DatasetDict
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import Partitioner
from flwr_datasets.preprocessor import Preprocessor, Divider
from folktables import ACSDataSource, ACSEmployment, ACSIncome
from sklearn.linear_model import LogisticRegression

from evaluation import evaluate_fairness
from utils import drop_data, flip_data


def _clone_partitioner(obj):
    """
    Creates a new instance of the same class as obj with the same arguments.
    Assumes that arguments to __init__ are stored as attributes in obj.
    """
    cls = obj.__class__  # Get the class of obj
    init_signature = inspect.signature(cls.__init__)

    arg_names = [param for param in init_signature.parameters if param != "self"]

    init_args = {arg: getattr(obj, arg) for arg in arg_names if hasattr(obj, arg)}

    return cls(**init_args)


class FairFederatedDataset(FederatedDataset):
    """
    Subclass from flower FederatedDateset.
    Representation of a dataset designed for federated learning, fairness evaluation, and analytics.

    Supports downloading, loading, preprocessing, modifying, evaluating, mapping and partitioning the dataset across multiple clients
    (e.g., edge devices or simulated silos). Fairness can be evaluated using specified sensitive attributes
    or default ones.

    If `sens_cols` are provided:
        - If two attributes are provided, intersectional fairness is evaluated.
        - If not provided, fairness is evaluated using default attributes: "SEX", "MAR", and "RAC1P".

    Partitions are created on a per-split basis using `Partitioner` objects from
    `flwr_datasets.partitioner`.

    Parameters
    ----------
    dataset : str, default="ACSIncome"
        The name of the dataset to load ( "ACSIncome" or "ACSEmployment"). This can be extended by the users.

    subset : Optional[str], default=None
        Optional dataset subset to load (e.g., a specific demographic or region).

    preprocessor : Optional[Union[Preprocessor, dict[str, tuple[str, ...]]]], default=None
        A callable or configuration dictionary used to apply transformations on the dataset.
        Can be used to resplit, remove or engineer features.

    partitioners : dict[str, Union[Partitioner, int]]
        Dictionary mapping dataset splits (e.g., state names) to partitioning strategies.
        Each split can use a custom `Partitioner` or an integer specifying the number of IID partitions.

    shuffle : bool, default=True
        Whether to shuffle the dataset before preprocessing and partitioning.

    seed : Optional[int], default=42
        Seed for reproducible shuffling. If `None`, a random seed is used.

    states : Optional[list[str]], default=None
        List of states to include in the dataset. If `None`, all available states are included.

    year : Optional[str], default="2018"
        The ACS year to load ("2014" until "2018" are directly supported).

    horizon : Optional[str], default="1-Year"
        Horizon of the ACS sample ("1-Year" or "5-Year").

    sensitive_attributes : Optional[list[str]], default=None
        List of attributes used to evaluate intersectional fairness. If not provided, the fairness metrics are evaluated for each of ["SEX", "MAR", "RAC1P"].

    fairness_level : Literal["attribute", "value", "attribute-value"], default="attribute"
        The level at which fairness is evaluated:
        - "attribute": only worst fairness metric is returned,
        - "value": worst fairness metric as well as for which values this fairness was calculated is returned,
        - "attribute-value": all possible fairness metric values are returned.

    fairness_metric : Literal["DP", "EO"], default="DP"
        Fairness metric to evaluate:
        - "DP": Demographic Parity
        - "EO": Equalized Odds

    fl_setting : Literal["cross-silo", None], default=None
        Strategy used to split the dataset into train/test:
        - "cross-silo": splits each clients dataset (each partition of the splits) into train, validation and test set
        - None: no train-test splitting
        - NOTE: for cross-device settings we propose to split the clients into clients for training and clients for testing after the dataset is generated.

    perc_train_val_test : Optional[list[float]], default=[0.7, 0.15, 0.15]
        Proportions for train, validation, and test sets in the cross-silo setting.

    path : Optional[PathLike], default=None
        Optional path where the dataset should be saved.

    modification_dict : Optional[dict[int, dict[str, ...]]], default=None
        Optional dictionary to apply data modifications (e.g., label_name flipping, dropping datapoints)
        to specific states, (e.g. { "CT": { "MAR": { "drop_rate": 0.2, "flip_rate": 0.1, "value": 2, "attribute": "SEX", "attribute_value": 2, },
        "SEX": { "drop_rate": 0.3, "flip_rate": 0.2, "value": 2, "attribute": None, "attribute_value": None, }, }}). These entries must always be given.
        If attribute and attribute_value are given, the dropping and flipping will be applied on the intersecting group. Also, either dropping or
        flipping is possible by specifying a rate of 0 if not wanted. This always happens after the mapping specified in the mapping parameter.

    mapping : Optional[dict[str, dict[int, int]]], default=None
        Optional remapping dictionary of categorical features or labels.

    **load_dataset_kwargs : dict
        Additional keyword arguments passed to `datasets.load_dataset`. Common examples:
        - `num_proc=4` (parallel loading)
        - `trust_remote_code=True`
        Avoid passing parameters that change the return type (e.g., non-`DatasetDict`).

    Returns
    -------
    None
        Initializes and configures a federated dataset with fairness-aware capabilities.
    """

    def __init__(
        self,
        *,
        dataset: str = "ACSIncome",
        subset: Optional[str] = None,
        preprocessor: Optional[Union[Preprocessor, dict[str, tuple[str, ...]]]] = None,
        partitioners: dict[str, Union[Partitioner, int]],
        shuffle: bool = True,
        seed: Optional[int] = 42,
        states: Optional[list[str]] = None,
        year: Optional[str] = "2018",
        horizon: Optional[str] = "1-Year",
        sensitive_attributes: Optional[list[str]] = None,
        fairness_level: Literal["attribute", "value", "attribute-value"] = "attribute",
        fairness_metric: Literal["DP", "EO"] = "DP",
        fl_setting: Literal["cross-silo", "cross-device", None] = None,
        perc_train_val_test: Optional[list[float]] = [0.7, 0.15, 0.15],
        path: Optional[PathLike] = None,
        modification_dict: Optional[dict[int, dict[str, ...]]] = None,
        mapping: Optional[dict[str, dict[int, int]]] = None,
        **load_dataset_kwargs: Any,
    ) -> None:
        partitioners = self._initialize_states(states, partitioners)
        super().__init__(
            dataset=dataset,
            subset=subset,
            preprocessor=preprocessor,
            partitioners=partitioners,
            shuffle=shuffle,
            seed=seed,
            **load_dataset_kwargs,
        )
        self._check_dataset()
        self._year = year
        self._horizon = horizon
        self._sensitive_attributes = sensitive_attributes
        self._fairness_level = fairness_level
        self._fairness_metric = fairness_metric
        self._fl_setting = fl_setting
        self._perc_train_test_split = perc_train_val_test
        self._path = path
        self._modification_dict = modification_dict
        self._mapping = mapping

    def save_dataset(self, dataset_path: PathLike) -> None:
        """
        Save the dataset to disk as csv files with names by state and partition index.
        """
        if not self._dataset_prepared:
            self._prepare_dataset()
        if self._sensitive_attributes is not None:
            warnings.warn(
                "The data you are saving contains columns with sensitive attributes. If these should not be in the training data later, please remove them before training."
            )
        for key, value in self._partitioners.items():
            partitioner = value
            num_partitions = partitioner.num_partitions
            for i in range(num_partitions):
                partition = partitioner.load_partition(partition_id=i)
                partition.to_csv(path_or_buf=f"{dataset_path}/{key}_{i}.csv")

    def evaluate(self, file):
        """
        Can be called at all times and runs once during _prepare_dataset. Then all partitions will be evaluated in terms of sensitive attribute value counts and the given fairness metric.
        """
        if not self._dataset_prepared:
            self._prepare_dataset()
        titles = list(self._dataset.keys())
        evaluate_fairness(
            partitioner_dict=self.partitioners,
            max_num_partitions=None,
            fairness_metric=self._fairness_metric,
            fairness_level=self._fairness_level,
            titles=titles,
            legend=True,
            label_name=self._label,
            intersectional_fairness=self._sensitive_attributes,
        )

    def _split_into_train_val_test(self):
        """
        If cross-silo setting is chosen, splits the dataset into train, test and validation sets.
        """
        divider_dict = {}
        partitioners_dict = {}
        for entry in self._dataset.keys():
            divider_dict[entry] = {
                f"{entry}_train": self._perc_train_test_split[0],
                f"{entry}_val": self._perc_train_test_split[1],
                f"{entry}_test": self._perc_train_test_split[2],
            }

        divider = Divider(divide_config=divider_dict)
        if self._fl_setting == "cross-silo":
            for entry in self._dataset.keys():
                partitioners_dict[f"{entry}_train"] = self._partitioners[entry]
                partitioners_dict[f"{entry}_val"] = _clone_partitioner(self._partitioners[entry])
                partitioners_dict[f"{entry}_test"] = _clone_partitioner(self._partitioners[entry])
            self._dataset = divider(self._dataset)
            self._partitioners = partitioners_dict
        elif self._fl_setting != "cross-device":
            raise ValueError("This train-val-test split strategy is not supported.")
        self._event = {
            "load_partition": {split: False for split in self._partitioners},
        }

    def _prepare_dataset(self) -> None:
        """
        This is overwritten from FederatedDataset to fit to our datasets.
        Prepars the dataset (prior to partitioning) by download, shuffle,
        preprocessing, mapping, modification_dict and fl_setting.

        Run only ONCE when triggered by load_* function. (In future more control whether
        this should happen lazily or not can be added). The operations done here should
        not happen more than once.

        It is controlled by a single flag, `_dataset_prepared` that is set True at the
        end of the function.
        """
        data_source = ACSDataSource(survey_year=self._year, horizon=self._horizon, survey="person", use_archive=True)
        self._dataset = DatasetDict()
        self._check_partitioners_correctness()
        for state in self._states:
            acs_data = data_source.get_data(states=[state], download=True)
            if self._dataset_name == "ACSEmployment":
                features, label, group = ACSEmployment.df_to_pandas(acs_data)
                self._label = "ESR"
            else:
                features, label, group = ACSIncome.df_to_pandas(acs_data)
                self._label = "PINCP"
            state_data = pd.concat([features, label], axis=1)
            if self._mapping is not None:
                for key, value in self._mapping.items():
                    state_data[key] = state_data[key].replace(value)
            if self._modification_dict is not None:
                if state in self._modification_dict.keys():
                    state_data = self._modify_data(state_data, state)
            self._dataset[state] = Dataset.from_pandas(state_data)
        if not isinstance(self._dataset, DatasetDict):
            raise ValueError(
                "Probably one of the specified parameter in `load_dataset_kwargs` "
                "change the return type of the datasets.load_dataset function. "
                "Make sure to use parameter such that the return type is DatasetDict. "
                f"The return type is currently: {type(self._dataset)}."
            )

        if self._shuffle:
            # Note it shuffles all the splits. The self._dataset is DatasetDict
            # so e.g. {"train": train_data, "test": test_data}. All splits get shuffled.
            self._dataset = self._dataset.shuffle(seed=self._seed)
        if self._preprocessor:
            self._dataset = self._preprocessor(self._dataset)
        if self._fl_setting is not None:
            self._split_into_train_val_test()
        self._dataset_prepared = True
        available_splits = list(self._dataset.keys())

        self._event["load_split"] = {split: False for split in available_splits}
        self.evaluate(self._path)
        if self._sensitive_attributes is not None:
            warnings.warn(
                "Your current data contains columns with sensitive attributes. If these should not be in the training data later, please remove them before training."
            )
        if self._path is not None:
            self.save_dataset(self._path)

    def _check_dataset(self):
        """
        Checks if the dataset is supported.
        """
        if self._dataset_name not in ["ACSIncome", "ACSEmployment"]:
            raise ValueError(f"This dataset is not compatible. Please choose ACSIncome or ACSEmployment.")

    def _initialize_states(self, states, partitioners):
        """
        Initializes the US states to all states if not specified and adds the partitioners for this.
        """
        if states is None:
            self._states = [
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

        else:
            self._states = states

        if partitioners is None:
            partitioners = {key: 1 for key in self._states}

        return partitioners

    def _modify_data(self, data, state):
        """
        Modifies a pandas dataframe representing a state
        according to the values given in _modification_dict by flipping labels or dropping datapoints
        """
        modifications = self._modification_dict[state]
        for key, value in modifications.items():
            drop_rate = value["drop_rate"]
            flip_rate = value["flip_rate"]
            value1 = value["value"]
            column2 = value["attribute"]
            value2 = value["attribute_value"]
            data = drop_data(data, drop_rate, key, value1, self._label, column2, value2)
            data = flip_data(data, flip_rate, key, value1, self._label, column2, value2)
        return data

    def to_json(self, **json_kw) -> str:
        """
        Returns the dataset as a JSON string.

        * Dataclasses → `asdict`
        * pathlib.Path → str(path)
        * datetime/date/time → ISO‑8601
        * Everything with a `__dict__` → that dict
        * Fallback → `str(obj)`

        Extra **json_kw are forwarded to `json.dumps`
        (e.g. `indent=2`, `sort_keys=True`, …).
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
