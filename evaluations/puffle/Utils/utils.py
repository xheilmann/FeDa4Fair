import json
import os
import random
import shutil
from collections import Counter, OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import flwr as fl
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from DPL.Learning.learning import Learning
from DPL.Regularization.RegularizationLoss import RegularizationLoss
from FederatedDataset.PartitionTypes.iid_partition import IIDPartition
from FederatedDataset.PartitionTypes.non_iid_partition_with_sensitive_feature import (
    NonIIDPartitionWithSensitiveFeature,
)
from FederatedDataset.PartitionTypes.representative import Representative
from FederatedDataset.Utils.utils import PartitionUtils
from flwr.common.typing import Scalar
from opacus import PrivacyEngine
from opacus.grad_sample import GradSampleModule
from opacus.optimizers import DPOptimizer
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VisionDataset
from Utils.model_utils import ModelUtils
from Utils.train_parameters import TrainParameters


class Utils:
    # plot the bar plot of the disparities
    @staticmethod
    def plot_bar_plot(title: str, disparities: list, nodes: list):
        plt.figure(figsize=(20, 8))
        plt.bar(range(len(disparities)), disparities)
        plt.xticks(range(len(nodes)), nodes)
        plt.title(title)
        # add a vertical line on xtick=75
        plt.axvline(x=75, color="r", linestyle="--")
        plt.xticks(rotation=90)
        # font size x axis
        plt.rcParams.update({"font.size": 10})
        plt.savefig(f"./{title}.png")
        plt.tight_layout()

    @staticmethod
    def get_noise(
        mechanism_type: str,
        epsilon: float | None = None,
        sensitivity: float | None = None,
        sigma: float | None = None,
    ):
        rng = np.random.default_rng()
        if mechanism_type == "laplace":
            return rng.laplace(loc=0, scale=sensitivity / epsilon, size=1)
        if mechanism_type == "geometric":
            p = 1 - np.exp(-epsilon / sensitivity)
            return (rng.geometric(p=p, size=1) - rng.geometric(p=p, size=1))[0]
        if mechanism_type == "gaussian":
            return rng.normal(loc=0, scale=sigma, size=1)[0]
        msg = "The mechanism type must be either laplace, geometric or gaussian"
        raise ValueError(msg)

    @staticmethod
    def seed_everything(seed: int):
        torch.manual_seed(seed)
        random.seed(seed)
        # Seed the legacy global generator for libraries that depend on it
        # while also satisfying Ruff NPY002 by creating a generator.
        _ = np.random.default_rng(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True

    @staticmethod
    def setup_wandb(args):
        private = args.epsilon is not None

        if not args.sweep:
            name = "experiment" if not args.run_name else args.run_name
            wandb_run = wandb.init(
                # set the wandb project where this run will be logged
                project=("FL_fairness" if args.project_name is None else args.project_name),
                name=name,
                # track hyperparameters and run metadata
                config={
                    "learning_rate": args.lr,
                    "csv": args.train_csv,
                    "DPL_regularization": args.regularization,
                    "batch_size": args.batch_size,
                    "dataset": args.dataset,
                    "num_rounds": args.num_rounds,
                    "pool_size": args.pool_size,
                    "sampled_clients": args.sampled_clients,
                    "epochs": args.epochs,
                    "private": private,
                    "epsilon": args.epsilon,
                    "gradnorm": args.clipping,
                    "probability_estimation": args.probability_estimation,
                    "perfect_probability_estimation": args.perfect_probability_estimation,
                    "alpha": args.alpha,
                    "percentage_unbalanced_nodes": args.percentage_unbalanced_nodes,
                    "alpha_target_lambda": args.alpha_target_lambda,
                    "target": args.target,
                    "weight_decay_lambda": args.weight_decay_lambda,
                    "regularization_mode": args.regularization_mode,
                    "regularization_lambda": args.regularization_lambda,
                    "momentum": args.momentum,
                    "node_shuffle_seed": args.node_shuffle_seed,
                    "unbalanced_ratio": args.unbalanced_ratio,
                },
            )
        else:
            wandb_run = wandb.init(
                # set the wandb project where this run will be logged
                project=("FL_fairness" if args.project_name is None else args.project_name),
                # track hyperparameters and run metadata
                config={
                    "learning_rate": args.lr,
                    "csv": args.train_csv,
                    "DPL_regularization": args.regularization,
                    "batch_size": args.batch_size,
                    "dataset": args.dataset,
                    "num_rounds": args.num_rounds,
                    "pool_size": args.pool_size,
                    "sampled_clients": args.sampled_clients,
                    "epochs": args.epochs,
                    "private": private,
                    "epsilon": args.epsilon,
                    "gradnorm": args.clipping,
                    "probability_estimation": args.probability_estimation,
                    "perfect_probability_estimation": args.perfect_probability_estimation,
                    "alpha": args.alpha,
                    "percentage_unbalanced_nodes": args.percentage_unbalanced_nodes,
                    "alpha_target_lambda": args.alpha_target_lambda,
                    "target": args.target,
                    "weight_decay_lambda": args.weight_decay_lambda,
                    "regularization_mode": args.regularization_mode,
                    "regularization_lambda": args.regularization_lambda,
                    "momentum": args.momentum,
                    "node_shuffle_seed": args.node_shuffle_seed,
                    "unbalanced_ratio": args.unbalanced_ratio,
                },
            )
        return wandb_run

    @staticmethod
    def get_dataset_statistics(client_dataset, client_disparity, client_metadata):
        sens_features = client_dataset.sensitive_features
        targets = client_dataset.targets
        sens_features_and_targets = list(zip(targets, sens_features, strict=False))
        counter_combination = Counter(sens_features_and_targets)
        counter_sens_features = Counter(sens_features)
        counter_targets = Counter(targets)

        # Return a dictionary with the statistics of the dataset
        return {
            "counter_combination": {str(key): value for key, value in counter_combination.items()},
            "counter_sens_features": {str(key): value for key, value in counter_sens_features.items()},
            "counter_targets": {str(key): value for key, value in counter_targets.items()},
            "client_disparity": client_disparity,
            "unfair_client": client_metadata,
        }

    # DEBUG
    def compute_disparities_debug(self):
        possible_clients = []
        for client in self:
            possible_z = np.array([])
            possible_y = np.array([])
            tmp_y = []
            tmp_z = []
            for sample in client:
                tmp_y.append(sample["y"])
                tmp_z.append(sample["z"])

            unique_z = np.unique(np.array(tmp_z))
            unique_y = np.unique(np.array(tmp_y))
            possible_z = np.unique(np.concatenate((possible_z, unique_z)))
            possible_y = np.unique(np.concatenate((possible_y, unique_y)))
            possible_clients.append((possible_y, possible_z))

        disparities = []
        for node, possible_client in zip(self, possible_clients, strict=False):
            possible_z = possible_client[1]
            possible_y = possible_client[0]
            max_disparity = np.max(
                [
                    RegularizationLoss().compute_violation_with_argmax(
                        predictions_argmax=(
                            np.array([sample["y"] for sample in node])
                            if isinstance(node, list)
                            else np.array(node["y"])
                        ),
                        sensitive_attribute_list=(
                            np.array([sample["z"] for sample in node])
                            if isinstance(node, list)
                            else np.array(node["z"])
                        ),
                        current_target=int(target),
                        current_sensitive_feature=int(sv),
                    )
                    for target in possible_y
                    for sv in possible_z
                ]
            )
            disparities.append(max_disparity)
        print(f"Mean of disparity {np.mean(disparities)} - std {np.std(disparities)}")
        return disparities

    @staticmethod
    def get_dataset_statistics_with_lists(nodes, client_disparity, client_metadata):
        dictionaries = []
        for node, disparity, metadata in zip(nodes, client_disparity, client_metadata, strict=False):
            sens_features = [item.item() for item in node["z"]]
            targets = [item.item() for item in node["y"]]
            sens_features_and_targets = list(zip(targets, sens_features, strict=False))

            counter_combination = Counter(sens_features_and_targets)
            counter_sens_features = Counter(sens_features)
            counter_targets = Counter(targets)

            dictionary = {
                "counter_combination": {str(key): value for key, value in counter_combination.items()},
                "counter_sens_features": {str(key): value for key, value in counter_sens_features.items()},
                "counter_targets": {str(key): value for key, value in counter_targets.items()},
                "client_disparity": disparity,
                "unfair_client": metadata,
            }
            dictionaries.append(dictionary)
        return dictionaries

    @staticmethod
    def get_optimizer(model, train_parameters, lr):
        if train_parameters.optimizer == "adam":
            return torch.optim.Adam(
                model.parameters(),
                lr=lr,
            )
        if train_parameters.optimizer == "sgd":
            return torch.optim.SGD(
                model.parameters(),
                lr=lr,
            )
        if train_parameters.optimizer == "adamW":
            return torch.optim.AdamW(
                model.parameters(),
                lr=lr,
            )
        msg = "Optimizer not recognized"
        raise ValueError(msg)

    @staticmethod
    def rescale_lambda(value, old_min, old_max, new_min, new_max):
        old_range = old_max - old_min
        new_range = new_max - new_min
        return (((value - old_min) * new_range) / old_range) + new_min

    @staticmethod
    def get_params(model: torch.nn.ModuleList) -> list[np.ndarray]:
        """Get model weights as a list of NumPy ndarrays."""
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

    @staticmethod
    def set_params(model: torch.nn.ModuleList, params: list[np.ndarray]):
        """Set model weights from a list of NumPy ndarrays."""
        params_dict = zip(model.state_dict().keys(), params, strict=False)
        state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

    @staticmethod
    def get_transformation(dataset_name: str):
        if dataset_name == "cifar10":
            return transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ],
            )
        if dataset_name == "mnist":
            return transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ],
            )
        if dataset_name == "celeba":
            return transforms.Compose(
                [
                    transforms.Resize((64, 64)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ],
            )
        return None

    @staticmethod
    def get_dataset(path_to_data: Path, cid: str, partition: str, dataset: str):
        # generate path to cid's data
        path_to_data = path_to_data / cid / (partition + ".pt")
        if dataset in {
            "dutch",
            "dutch_prepared",
            "income",
            "income_NO_RACE",
            "income_cross_device",
            "employment",
            "employment_NO_RACE",
            "celeba_prepared",
        }:
            return torch.load(path_to_data)
        return TorchVisionFL(
            path_to_data,
            transform=Utils.get_transformation(dataset),
        )

    @staticmethod
    def get_random_id_splits(total: int, val_ratio: float, shuffle: bool = True):
        """
        Splits a list of length `total` into two following a
        (1-val_ratio):val_ratio partitioning.

        By default the indices are shuffled before creating the split and
        returning.
        """
        indices = list(range(total)) if isinstance(total, int) else total

        split = int(np.floor(val_ratio * len(indices)))
        if shuffle:
            rng = np.random.default_rng()
            rng.shuffle(indices)
        return indices[split:], indices[:split]

    @staticmethod
    def _do_iid_partitioning(idx, pool_size, dataset):
        splitted_indexes = IIDPartition.do_iid_partitioning_with_indexes(
            indexes=idx,
            num_partitions=pool_size,
        )
        return PartitionUtils.create_splitted_dataset_from_tuple(
            splitted_indexes=splitted_indexes,
            dataset=dataset,
        )

    @staticmethod
    def _do_non_iid_partitioning(labels, sensitive_attribute, pool_size, alpha, dataset):
        (
            _,
            _,
            partitions_index_list,
            _,
        ) = NonIIDPartitionWithSensitiveFeature.do_partitioning_with_dataset_list(
            labels=labels,
            sensitive_features=sensitive_attribute,
            num_partitions=pool_size,
            alpha=alpha,
        )
        return PartitionUtils.create_splitted_dataset_from_tuple(
            splitted_indexes=partitions_index_list,
            dataset=dataset,
        )

    @staticmethod
    def _do_representative_partitioning(
        labels,
        sensitive_attribute,
        pool_size,
        group_to_reduce,
        group_to_increment,
        number_of_samples_per_node,
        ratio_unfair_nodes,
        ratio_unfairness,
        one_group_nodes,
        dataset,
    ):
        print("SPLITTING THE DATASET USING the representative partitioning")
        partitions_index_list, _ = Representative.do_partitioning(
            labels=labels,
            sensitive_features=sensitive_attribute,
            num_partitions=pool_size,
            total_num_classes=2,
            group_to_reduce=group_to_reduce,
            group_to_increment=group_to_increment,
            number_of_samples_per_node=number_of_samples_per_node,
            ratio_unfair_nodes=ratio_unfair_nodes,
            ratio_unfairness=ratio_unfairness,
            one_group_nodes=one_group_nodes,
        )
        return PartitionUtils.create_splitted_dataset_from_tuple(
            splitted_indexes=partitions_index_list,
            dataset=dataset,
        )

    @staticmethod
    def perform_fl_partitioning(
        path_to_dataset: str,
        pool_size: int,
        partition_type: str,
        alpha: float = 1,
        train_parameters: TrainParameters = None,
        partition: str = "train",
        group_to_reduce=None,
        group_to_increment=None,
        number_of_samples_per_node=None,
        ratio_unfair_nodes=None,
        ratio_unfairness=None,
        one_group_nodes: bool = False,
        splitted_data_dir: str | None = None,
    ):
        print("Partitioning the dataset")

        images, sensitive_attribute, labels = torch.load(path_to_dataset)
        mapping = {-1: 0, 1: 1, 0: 0}
        if train_parameters.metric in {"disparity", "equalised_odds"}:
            sensitive_attribute = torch.tensor([mapping.get(item, item) for item in sensitive_attribute])
        else:
            sensitive_attribute = torch.tensor(list(sensitive_attribute))

        idx = torch.tensor(list(range(len(images))))
        dataset = [idx, sensitive_attribute, labels]
        print(Counter(labels))

        if partition_type == "iid":
            partitions = Utils._do_iid_partitioning(idx, pool_size, dataset)
        elif partition_type == "non_iid":
            partitions = Utils._do_non_iid_partitioning(labels, sensitive_attribute, pool_size, alpha, dataset)
        elif partition_type == "representative":
            partitions = Utils._do_representative_partitioning(
                labels,
                sensitive_attribute,
                pool_size,
                group_to_reduce,
                group_to_increment,
                number_of_samples_per_node,
                ratio_unfair_nodes,
                ratio_unfairness,
                one_group_nodes,
                dataset,
            )
        else:
            msg = f"Unknown partition type: {partition_type}"
            raise ValueError(msg)

        # now save partitioned dataset to disk
        splits_dir = path_to_dataset.parent / splitted_data_dir
        if splits_dir.exists() and partition == "train":
            shutil.rmtree(splits_dir)
            splits_dir.mkdir(parents=True)

        return Utils._save_partitions(
            splits_dir,
            pool_size,
            partitions,
            images,
            partition,
            possible_y=np.unique(labels),
            possible_z=np.unique(sensitive_attribute),
        )

    @staticmethod
    def _save_partitions(splits_dir, pool_size, partitions, images, partition, possible_y, possible_z):
        nodes = []
        possible_z_all, possible_y_all = possible_z.astype(float), possible_y.astype(float)
        predictions, sensitive_features_all = [], []
        for p in range(pool_size):
            labels, sensitive_features, image_idx = partitions[p][2], partitions[p][1], partitions[p][0]
            imgs = [images[image_id] for image_id in image_idx]
            (splits_dir / str(p)).mkdir(exist_ok=True)
            nodes.append({"y": labels, "z": sensitive_features})
            with (splits_dir / str(p) / ("train.pt" if partition == "train" else "test.pt")).open("wb") as f:
                torch.save([imgs, sensitive_features, labels], f)
            predictions.append([int(item) for item in labels])
            sensitive_features_all.append([int(item) for item in sensitive_features])

        tmp_nodes = [[{"y": int(y), "z": int(z)} for y, z in zip(node["y"], node["z"], strict=False)] for node in nodes]
        disparities = Utils.compute_disparities_debug(tmp_nodes)
        Utils.plot_bar_plot(
            title="Distribution Disparities", disparities=disparities, nodes=[f"{i}" for i in range(len(nodes))]
        )

        possible_y_str = [str(int(item)) for item in possible_y_all.tolist()]
        possible_z_str = [str(int(item)) for item in possible_z_all.tolist()]
        missing_combinations, all_combinations = [], []
        sent_disparity_combinations = [f"1|{sensitive}" for sensitive in possible_z_str]
        for comb in sent_disparity_combinations:
            missing_combinations.append(("0" + comb[1:], comb))
            all_combinations.extend([comb, "0" + comb[1:]])

        json_file = {
            "possible_z": possible_z_str,
            "possible_y": possible_y_str,
            "missing_combinations": missing_combinations,
            "all_combinations": all_combinations,
            "combinations": sent_disparity_combinations,
        }
        with (splits_dir / "metadata.json").open("w") as outfile:
            json.dump(json_file, outfile, indent=4)
        Utils.plot_distributions(
            title="Distribution of the nodes",
            counter_groups=Utils.compute_distribution_debug(predictions, sensitive_features_all),
            all_combinations=all_combinations,
        )
        return splits_dir

    @staticmethod
    def plot_distributions(title: str, counter_groups: list, all_combinations: list):
        plt.figure(figsize=(20, 8))
        previous_sum = []
        for combination in all_combinations:
            counter = [counter[(int(combination[0]), int(combination[-1]))] for counter in counter_groups]
            print(counter)
            if previous_sum:
                plt.bar(range(len(counter)), counter, bottom=previous_sum)
            else:
                plt.bar(range(len(counter)), counter)
                previous_sum = [0 for _ in counter]

            previous_sum = [sum(x) for x in zip(previous_sum, counter, strict=False)]

        plt.xlabel("Client")
        plt.ylabel("Amount of samples")
        plt.title("Samples for each group (target/sensitive Value) per client")
        plt.legend(all_combinations)
        # font size 20
        plt.rcParams.update({"font.size": 20})
        plt.rcParams.update({"font.size": 10})
        plt.savefig(f"./{title}.png")
        plt.tight_layout()

    @staticmethod
    def compute_distribution_debug(predictions, sensitive_features):
        counter_nodes = []
        for prediction, sensitive_feature in zip(predictions, sensitive_features, strict=False):
            counter_node = []
            for pred, sf in zip(prediction, sensitive_feature, strict=False):
                counter_node.append((pred, sf))
            counter_nodes.append(Counter(counter_node))
        return counter_nodes

    @staticmethod
    def prepare_dataset_for_fl(
        dataset,
        base_path: str,
        dataset_name: str,
        partition: str = "train",
    ):
        # fuse all data splits into a single "training.pt"
        data_loc = Path(base_path) / f"{dataset_name}/{dataset_name}-10-batches-py"
        train_path = data_loc / ("training.pt" if partition == "train" else "test.pt")
        print("Generating unified dataset")
        torch.save(
            [
                dataset.samples,
                np.array(dataset.sensitive_attributes),
                np.array(dataset.targets),
            ],
            train_path,
        )

        print("Data Correctly Loaded")

        return train_path

    @staticmethod
    def get_dataloader(
        path_to_data: str,
        cid: str,
        # is_train: bool,
        batch_size: int,
        workers: int,
        dataset: str,
        partition: str = "train",
    ):
        """Generates trainset/valset object and returns appropiate dataloader."""
        partition = "train" if partition == "train" else "test" if partition == "test" else "val"
        dataset = Utils.get_dataset(Path(path_to_data), cid, partition, dataset)

        # we use as number of workers all the cpu cores assigned to this actor
        kwargs = {"num_workers": workers, "pin_memory": True, "drop_last": False}
        return DataLoader(dataset, batch_size=batch_size, **kwargs)

    @staticmethod
    def create_private_model(
        model: torch.nn.Module,
        epsilon: float,
        original_optimizer,
        train_loader,
        epochs: int,
        delta: float,
        max_grad_norm: float,
        noise_multiplier: float = 0,
        accountant=None,
    ) -> tuple[GradSampleModule, DPOptimizer, DataLoader]:
        """
        Create a private model using Opacus.

        Args:
            model (torch.nn.Module): the model to wrap
            epsilon (float): the target epsilon for the privacy budget
            original_optimizer (_type_): the optimizer of the model before
                wrapping it with Privacy Engine
            train_loader (_type_): the train dataloader used to train the model
            epochs (_type_): for how many epochs the model will be trained
            delta (float): the delta for the privacy budget
            max_grad_norm (float): the clipping value for the gradients
            noise_multiplier (float): noise multiplier
            accountant: accountant

        Returns:
            Tuple[GradSampleModule, DPOptimizer, DataLoader]: the wrapped model,
                the wrapped optimizer and the train dataloader

        """
        privacy_engine = PrivacyEngine(accountant="rdp")
        if accountant:
            privacy_engine.accountant = accountant

        # We can wrap the model with Privacy Engine using the
        # method .make_private(). This doesn't require you to
        # specify a epsilon. In this case we need to specify a
        # noise multiplier.
        # make_private_with_epsilon() instead requires you to
        # provide a target epsilon and a target delta. In this
        # case you don't need to specify a noise multiplier.
        if epsilon:
            print(f"Creating private model using epsilon {epsilon}")
            (
                private_model,
                optimizer,
                train_loader,
            ) = privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=original_optimizer,
                data_loader=train_loader,
                epochs=epochs,
                target_epsilon=epsilon,
                target_delta=delta,
                max_grad_norm=max_grad_norm,
            )
        else:
            private_model, optimizer, train_loader = privacy_engine.make_private(
                module=model,
                optimizer=original_optimizer,
                data_loader=train_loader,
                noise_multiplier=noise_multiplier,
                max_grad_norm=max_grad_norm,
            )

        return private_model, optimizer, train_loader, privacy_engine

    @staticmethod
    def get_evaluate_fn(
        test_set,
        dataset_name: str,
        train_parameters: TrainParameters,
        wandb_run: wandb.sdk.wandb_run.Run,
        batch_size: int,
    ) -> Callable[[fl.common.NDArrays], tuple[float, float] | None]:
        """Return an evaluation function for centralized evaluation."""

        def evaluate(
            server_round: int, parameters: fl.common.NDArrays, _config: dict[str, Scalar]
        ) -> tuple[float, float] | None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model = ModelUtils.get_model(dataset_name, device)
            Utils.set_params(model, parameters)
            model.to(device)

            testloader = torch.utils.data.DataLoader(
                test_set,
                batch_size=batch_size,
            )

            (
                test_loss,
                accuracy,
                f1score,
                precision,
                recall,
                max_disparity_test,
            ) = Learning.test(
                model=model,
                test_loader=testloader,
                train_parameters=train_parameters,
                current_epoch=server_round,
            )
            if wandb_run:
                wandb_run.log(
                    {
                        "Centralised Test Loss": test_loss,
                        "Centralised Test Accuracy": accuracy,
                        "Centralised Test F1 Score": f1score,
                        "Centralised Test Precision": precision,
                        "Centralised Test Recall": recall,
                        "Centralised Test Max Disparity": max_disparity_test,
                        "FL Round": server_round,
                    }
                )

            return test_loss, {"Test Accuracy": accuracy}

        return evaluate


class TorchVisionFL(VisionDataset):
    """
    This is just a trimmed down version of torchvision.datasets.MNIST.

    Use this class by either passing a path to a torch file (.pt)
    containing (data, targets) or pass the data, targets directly
    instead.
    """

    def __init__(
        self,
        path_to_data=None,
        data=None,
        targets=None,
        transform: Callable | None = None,
    ) -> None:
        path = path_to_data.parent if path_to_data else None
        self.dataset_path = path.parent.parent.parent if path_to_data else None

        super().__init__(path, transform=transform)
        self.transform = transform

        if path_to_data:
            # load data and targets (path_to_data points to an specific .pt file)
            self.data, self.sensitive_features, self.targets = torch.load(path_to_data)
        else:
            self.data = data
            self.targets = targets

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if isinstance(img, str):
            path = self.dataset_path / "img_align_celeba/" / self.data[index]
            img = Image.open(path).convert(
                "RGB",
            )

        if not isinstance(img, Image.Image):  # if not PIL image
            if not isinstance(img, np.ndarray):  # if torch tensor
                img = img.numpy()

            img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        sensitive_feature = self.sensitive_features[index]

        return img, sensitive_feature, target

    def __len__(self) -> int:
        return len(self.data)
