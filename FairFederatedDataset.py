# Copyright 2023 Flower Labs GmbH. All Rights Reserved.
#
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
#TODO:
# 1. overall
# -natural output (that can be used for normal FL)
# -modified output
# 2. dataset
# -dataset to be used in the generation of  client - level data, chosen among the ones provided in the ACS dataset
# -name of the dataset(Income, Employment)
# -data sampling strategy(range of datapoints per client(min, max))
# -include or exclude sensitive attributes
# 3.the number of clients that will be involved in the simulation and will need training data,
# -cross - silo(maximum: states)
# -cross - device(split in state, split by attributevalues(column name, percentage))
# 5. the sensitive attributes against which we want to measure the unfairness,
# - choosing(gender, race, mar)
# -binary vs non - binary(merging the smallest groups)
# - distribution of clients unfairneses
# - the unfairness level between different clients and their sensitive attributes.
# 6. output
# - dataset statistics per client(  # datapoints, fairness metrics (gender, race mar), performance metrics, modifications)
# -overall statistics global model FedAVG((  # datapoints, fairness metrics(gender, race mar), performance metrics) before modifications and after
# - global model on pooled dataset
# - the datasets as csv files, local model training as numpy array
#TODO:
# 4. the fairness metric used to evaluate the model unfairness,
# - measure on simple models(logistic regression), raw data
# -unfairness on different attribute values
# -unfairness on different attributes
# -Demographic disparity
# - Equalized odds
# -demographic parity
# - sampling for the fairness?

"""FairFederatedDataset."""

from os import PathLike
from typing import Any, Optional, Union, Literal


import pandas as pd
from datasets import Dataset, DatasetDict
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import Partitioner
from flwr_datasets.preprocessor import Preprocessor, Divider, Merger

from folktables import ACSDataSource, ACSEmployment, ACSIncome

from evaluation import evaluate_fairness


class FairFederatedDataset (FederatedDataset):
    """Representation of a dataset for federated learning/evaluation/analytics.

    Download, partition data among clients (edge devices), or load full dataset.

    Partitions are created per-split-basis using Partitioners from
    `flwr_datasets.partitioner` specified in `partitioners` (see `partitioners`
    parameter for more information).

    Parameters
    ----------
    dataset : str
        The name of the dataset: ACSIncome or ACSEmployment.
    subset : List[str]
        List of the states to be included in the dataset. Default is all states.
    preprocessor : Optional[Union[Preprocessor, Dict[str, Tuple[str, ...]]]]
        `Callable` that transforms `DatasetDict` by resplitting, removing
        features, creating new features, performing any other preprocessing operation,
        or configuration dict for `Merger`. Applied after shuffling. If None,
        no operation is applied.
    partitioners : Dict[str, Union[Partitioner, int]]
    Here we have a dictionary with the splits being the states. then one partioner per state.
        A dictionary mapping the Dataset split (a `str`) to a `Partitioner` or an `int`
        (representing the number of IID partitions that this split should be
        partitioned into, i.e., using the default partitioner
        `IidPartitioner <https://flower.ai/docs/datasets/ref-api/flwr_
        datasets.partitioner.IidPartitioner.html>`_). One or multiple `Partitioner`
        objects can be specified in that manner, but at most, one per split.
    shuffle : bool
        Whether to randomize the order of samples. Applied prior to preprocessing
        operations, speratelly to each of the present splits in the dataset. It uses
        the `seed` argument. Defaults to True.
    seed : Optional[int]
        Seed used for dataset shuffling. It has no effect if `shuffle` is False. The
        seed cannot be set in the later stages. If `None`, then fresh, unpredictable
        entropy will be pulled from the OS. Defaults to 42.
    load_dataset_kwargs : Any
        Additional keyword arguments passed to `datasets.load_dataset` function.
        Currently used paramters used are dataset => path (in load_dataset),
        subset => name (in load_dataset). You can pass e.g., `num_proc=4`,
        `trust_remote_code=True`. Do not pass any parameters that modify the
        return type such as another type than DatasetDict is returned.

    Examples
    --------
    Use MNIST dataset for Federated Learning with 100 clients (edge devices):

    >>> from flwr_datasets import FederatedDataset
    >>>
    >>> fds = FederatedDataset(dataset="mnist", partitioners={"train": 100})
    >>> # Load partition for a client with ID 10.
    >>> partition = fds.load_partition(10)
    >>> # Use test split for centralized evaluation.
    >>> centralized = fds.load_split("test")

    Use CIFAR10 dataset for Federated Laerning with 100 clients:

    >>> from flwr_datasets import FederatedDataset
    >>> from flwr_datasets.partitioner import DirichletPartitioner
    >>>
    >>> partitioner = DirichletPartitioner(num_partitions=10, partition_by="label",
    >>>                                    alpha=0.5, min_partition_size=10)
    >>> fds = FederatedDataset(dataset="cifar10", partitioners={"train": partitioner})
    >>> partition = fds.load_partition(partition_id=0)

    Visualize the partitioned datasets:

    >>> from flwr_datasets.visualization import plot_label_distributions
    >>>
    >>> _ = plot_label_distributions(
    >>>     partitioner=fds.partitioners["train"],
    >>>     label_name="label",
    >>>     legend=True,
    >>> )
    """

    # pylint: disable=too-many-instance-attributes, too-many-arguments
    def __init__(
        self,
        *,
        dataset: str = "ACSIncome",
        subset: Optional[str] = None,
        preprocessor: Optional[Union[Preprocessor, dict[str, tuple[str, ...]]]] = None,
        partitioners: dict[str, Union[Partitioner, int]],
        shuffle: bool = True,
        seed: Optional[int] = 42,
        states: Optional[list[str]] =  None,
        year: Optional[list[str]] =['2018'],
        horizon: Optional[str]= '1-Year',
        binary: Optional[bool]=False,
        fairness_modification: Optional[bool]=False,
        sensitive_attribute: Optional[list[str]]="sex",
        individual_fairness: Literal["attribute", "value","attribute-value"] = "attribute",
        fairness_metric:  Literal["DP", "EO"] = "DP",
        train_test_split: Literal["cross-silo", "cross-device", None]= None,
        perc_train_val_test: Optional[list[float]] = [0.7, 0.15, 0.15],
        path: Optional[PathLike] = None,
        **load_dataset_kwargs: Any,
    ) -> None:
        super().__init__(dataset, subset, preprocessor, partitioners, shuffle, seed, **load_dataset_kwargs)
        self._check_dataset()
        self._initilize_states(states)
        self._year = year
        self._horizon = horizon
        self._binary = binary
        self._fairness_modification = fairness_modification
        self._sensitive_attribute = sensitive_attribute
        self._individual_fairness = individual_fairness
        self._fairness_metric = fairness_metric
        self._train_test_split = train_test_split
        self._perc_train_test_split = perc_train_val_test
        self._path = path


    def save_dataset(self, dataset_path: PathLike) -> None:
        self._dataset.save_to_disk(dataset_dict_path = self._path)

    def evaluate(self, file):
        titles = list(self._dataset.keys())
        fig_dis,axes_dis, df_list_dis, fig, axes, df_list = evaluate_fairness(partitioner_list=self.partitioners,
                                                                              max_num_partitions=None,
                                                                              fairness_metric=self._fairness_metric,
                                                                              fairness=self._individual_fairness,
                                                                              titles=titles,
                                                                              legend=True,
                                                                              class_label=self._label)
        for i in range(len(df_list_dis)):
            df_list_dis[i].to_csv(path_or_buf = f"{self._path}/{titles[i]}_count.csv")
            df_list[i].to_csv(path_or_buf=f"{self._path}/{titles[i]}_{self._fairness_metric}.csv")


    def _split_into_train_val_test(self ):
        divider_dict= {}
        for entry in self._dataset.keys():
            divider_dict[entry] = {f"{entry}_train": self._perc_train_test_split[0], f"{entry}_val": self._perc_train_test_split[1], f"{entry}_test": self._perc_train_test_split[2]}
        divider = Divider(divide_config=divider_dict)
        if self._train_test_split == "cross-silo":
            self._dataset = divider(self._dataset)
        elif self._train_test_split == "cross-device":
            merger_tuple = tuple([f"{entry}_test" for entry in self._dataset.keys()])
            self._dataset = divider(self._dataset)
            merger_dict = {f"{entry}": (f"{entry}", ) for entry in self._dataset.keys()}
            merger_dict["test"] = merger_tuple
            merger = Merger(merge_config=merger_dict)
            self._dataset = merger(self._dataset)

        else:
            raise ValueError("This train-val-test split strategy is not supported.")

    def _prepare_dataset(self) -> None:
        """This is overwritten from FederatedDataset to fit to our Datasets.
        Prepars the dataset (prior to partitioning) by download, shuffle,preprocessing, binary.

        The binary feature is to tag if sensitive attributes should be binarized before
        the dataset is partitioned. We binarize in the following way:

        Run only ONCE when triggered by load_* function. (In future more control whether
        this should happen lazily or not can be added). The operations done here should
        not happen more than once.

        It is controlled by a single flag, `_dataset_prepared` that is set True at the
        end of the function.

        """
        data_source = ACSDataSource(survey_year=self._year, horizon=self._horizon, survey='person')
        self._dataset = DatasetDict()
        for state in self._states:
            acs_data = data_source.get_data(states=[state], download=True)
            if self._dataset_prepared == "ACSEmployment":
                features, label, group = ACSEmployment.df_to_pandas(acs_data)
                self._label = "ESR"
            else:
                features, label, group = ACSIncome.df_to_pandas(acs_data)
                self._label = "PINCP"
            state_data=pd.concat([features, label], axis=1)
            if self._binary:
                # TODO:add implementation here
                pass
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
        if self._fairness_modification:
            self._modify_for_fairness()
        if self._train_test_split is not None:
            self._split_into_train_val_test()
        self.evaluate()
        # TODO: add implementation for throwing out sensitive columns here
        if self._path is not None:
            self.save_dataset(self._path)
        available_splits = list(self._dataset.keys())
        self._event["load_split"] = {split: False for split in available_splits}
        self._dataset_prepared = True

    def _check_dataset(self):
       if self._dataset_name not in ["ACSIncome", "ACSEmployment"]:
            raise ValueError(
                f"This dataset is not compatible. Please choose ACSIncome or ACSEmployment."
            )

    def _initilize_states(self, states):
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

    def _modify_for_fairness(self):
        #TODO
        pass

