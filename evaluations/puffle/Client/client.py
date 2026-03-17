import copy
import gc
import json
import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import dill
import flwr as fl
import numpy as np
import ray
import torch
from DPL.Learning.learning import Learning
from DPL.Regularization.RegularizationLoss import RegularizationLoss
from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
from Utils.model_utils import ModelUtils
from Utils.train_parameters import TrainParameters
from Utils.utils import Utils

if TYPE_CHECKING:
    from flwr.common.typing import Scalar


logger = logging.getLogger(__name__)


class FlowerClientDisparity(fl.client.NumPyClient):
    def __init__(
        self,
        cid: str,
        fed_dir_data: str,
        dataset_name: str,
        clipping: float,
        lr: float,
        train_parameters: TrainParameters,
        client_generator,
    ):
        logger.info("Node %s is initializing...", cid)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        self.train_parameters = copy.deepcopy(train_parameters)
        self.cid = cid
        self.fed_dir = Path(fed_dir_data)
        self.properties: dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}
        self.dataset_name = dataset_name
        self.clipping = clipping
        self.lr = lr
        self.client_generator = client_generator
        self.net = ModelUtils.get_model(dataset_name, _device=self.train_parameters.device)
        self.optimizer = self.get_optimizer(model=self.net)

        if self.train_parameters.regularization:
            self.model_regularization = ModelUtils.get_model(
                self.dataset_name,
                _device=self.train_parameters.device,
            )
            self.optimizer_regularization = self.get_optimizer(model=self.model_regularization)

    def _compute_reweighing_weights(self, train_loader):
        reweighing_weights = None
        if self.train_parameters.reweighing:
            counts = {}
            total_samples = 0
            for _, sens_1, sens_2, sens_3, target in train_loader:
                if self.train_parameters.sensitive_attribute == "SEX":
                    sensitive_features = sens_1
                elif self.train_parameters.sensitive_attribute == "MAR":
                    sensitive_features = sens_2
                elif self.train_parameters.sensitive_attribute == "RAC1P":
                    sensitive_features = sens_3
                else:
                    sensitive_features = sens_1

                for z, y in zip(sensitive_features, target, strict=False):
                    z_item = z.item()
                    y_item = y.item()
                    key = (z_item, y_item)
                    counts[key] = counts.get(key, 0) + 1
                    total_samples += 1

            prob_z = {}
            prob_y = {}
            unique_z = {k[0] for k in counts}
            unique_y = {k[1] for k in counts}

            for z in unique_z:
                n_z = sum(counts.get((z, y), 0) for y in unique_y)
                prob_z[z] = n_z / total_samples

            for y in unique_y:
                n_y = sum(counts.get((z, y), 0) for z in unique_z)
                prob_y[y] = n_y / total_samples

            reweighing_weights = {}
            for k, count in counts.items():
                z, y = k
                prob_zy = count / total_samples
                if prob_zy > 0:
                    reweighing_weights[k] = (prob_z[z] * prob_y[y]) / prob_zy
                else:
                    reweighing_weights[k] = 1.0
        return reweighing_weights

    def _setup_privacy_and_noise(self, train_loader):
        loaded_privacy_engine = None
        loaded_privacy_engine_regularization = None
        first_round = False

        if (self.fed_dir / f"privacy_engine_{self.cid}.pkl").exists():
            with (self.fed_dir / f"privacy_engine_{self.cid}.pkl").open("rb") as file:
                loaded_privacy_engine = dill.load(file)  # noqa: S301

            if (self.fed_dir / f"privacy_engine_regularization_{self.cid}.pkl").exists():
                with (self.fed_dir / f"privacy_engine_regularization_{self.cid}.pkl").open("rb") as file:
                    loaded_privacy_engine_regularization = dill.load(file)  # noqa: S301
        else:
            if self.train_parameters.regularization_mode == "tunable":
                self.train_parameters.regularization_lambda = 0
            first_round = True

        if self.train_parameters.epsilon is None:
            self.noise_multiplier = 0
            self.original_epsilon = None
        elif (self.fed_dir / f"noise_level_{self.cid}.pkl").exists():
            with (self.fed_dir / f"noise_level_{self.cid}.pkl").open("rb") as file:
                self.noise_multiplier = dill.load(file)  # noqa: S301
                self.original_epsilon = self.train_parameters.epsilon
                self.train_parameters.epsilon = None
        else:
            noise = self.get_noise(dataset=train_loader)
            with (self.fed_dir / f"noise_level_{self.cid}.pkl").open("wb") as file:
                dill.dump(noise, file)
            self.noise_multiplier = noise
            self.original_epsilon = self.train_parameters.epsilon
            self.train_parameters.epsilon = None
        return loaded_privacy_engine, loaded_privacy_engine_regularization, first_round

    def get_optimizer(self, model):
        if self.train_parameters.optimizer == "adam":
            return torch.optim.Adam(model.parameters(), lr=self.lr)
        if self.train_parameters.optimizer == "sgd":
            return torch.optim.SGD(model.parameters(), lr=self.lr)
        if self.train_parameters.optimizer == "adamW":
            return torch.optim.AdamW(model.parameters(), lr=self.lr)
        msg = "Optimizer not recognized"
        raise ValueError(msg)

    def get_parameters(self, config):
        return Utils.get_params(self.net)

    def fit(self, parameters, config, average_probabilities=None):
        is_tunable = self.train_parameters.regularization_mode == "tunable"
        current_fl_round = config["server_round"]
        random_generator = np.random.default_rng(seed=[int(self.client_generator.random() * 2**32), current_fl_round])
        seed = int(random_generator.random() * 2**32)
        Utils.seed_everything(seed)

        if (self.fed_dir / "avg_proba.pkl").exists():
            with (self.fed_dir / "avg_proba.pkl").open("rb") as file:
                average_probabilities = dill.load(file)  # noqa: S301

        Utils.set_params(self.net, parameters)

        with (self.fed_dir / "counter_sampling.pkl").open("rb") as f:
            counter_sampling = dill.load(f)  # noqa: S301
            self.sampling_frequency = counter_sampling[str(self.cid)]

        num_workers = int(ray.get_runtime_context().get_assigned_resources()["CPU"])
        train_loader = Utils.get_dataloader(
            self.fed_dir,
            self.cid,
            batch_size=config["batch_size"],
            workers=num_workers,
            dataset=self.dataset_name,
            partition="train",
        )

        print(f"Client {self.cid} has {len(train_loader.dataset)} samples")
        reweighing_weights = self._compute_reweighing_weights(train_loader)
        self.delta = (1 / len(train_loader.dataset)) / 3

        sigma_update_lambda = self._get_sigma_update_lambda(train_loader)
        loaded_pe, loaded_pe_reg, first_round = self._setup_privacy_and_noise(train_loader)

        (private_net, private_optimizer, train_loader, privacy_engine) = Utils.create_private_model(
            model=self.net,
            epsilon=self.train_parameters.epsilon,
            original_optimizer=self.optimizer,
            train_loader=train_loader,
            epochs=self.train_parameters.epochs,
            delta=self.delta,
            max_grad_norm=self.clipping,
            noise_multiplier=self.noise_multiplier,
            accountant=loaded_pe,
        )
        private_net.to(self.train_parameters.device)

        max_disp_before = RegularizationLoss().violation_with_dataset(
            model=private_net,
            dataset=train_loader,
            device=self.train_parameters.device,
            average_probabilities=average_probabilities,
        )

        if not first_round and self.train_parameters.target and is_tunable:
            self.train_parameters.regularization_lambda = self.compute_starting_lambda_with_disparity(max_disp_before)

        private_model_reg, private_opt_reg, pe_reg = self._setup_regularization_model(train_loader, loaded_pe_reg)

        gc.collect()
        all_metrics, all_losses, history_lambda = self._run_training_epochs(
            private_net,
            private_model_reg,
            private_optimizer,
            private_opt_reg,
            train_loader,
            current_fl_round,
            average_probabilities,
            sigma_update_lambda,
            reweighing_weights,
            max_disp_before,
        )

        Utils.set_params(self.net, Utils.get_params(private_net))
        self._save_privacy_state(privacy_engine, pe_reg)

        final_epsilon = self._compute_final_epsilon()
        probabilities, counters, counters_no_noise = self._compute_counters(private_net, train_loader)

        del private_net
        if private_model_reg:
            del private_model_reg
        gc.collect()

        return (
            Utils.get_params(self.net),
            len(train_loader.dataset),
            {
                "train_losses": all_losses,
                "train_loss": all_metrics[-1]["Train Loss"],
                "train_loss_with_regularization": all_metrics[-1]["Train Loss + Regularizaion"],
                "train_accuracy": all_metrics[-1]["Train Accuracy"],
                "epsilon": final_epsilon,
                "delta": self.delta,
                "probabilities": probabilities,
                "cid": self.cid,
                "targets": all_metrics[-1].get("targets", []),
                "sensitive_attributes": all_metrics[-1].get("sensitive_attributes", []),
                "Disparity Train": all_metrics[-1]["Max Unfairness Train"],
                "Lambda": self.train_parameters.regularization_lambda,
                "counters": counters,
                "counters_no_noise": counters_no_noise,
                "Max Disparity Train Before Local Epoch": max_disp_before,
                "history_lambda": history_lambda,
            },
        )

    def _get_sigma_update_lambda(self, train_loader):
        if self.train_parameters.epsilon_lambda is None:
            return None
        sampling_ratio = 1 / len(train_loader)
        iterations = self.sampling_frequency * self.train_parameters.epochs * len(train_loader)
        return get_noise_multiplier(
            target_epsilon=self.train_parameters.epsilon_lambda,
            target_delta=self.delta,
            sample_rate=sampling_ratio,
            steps=iterations,
            accountant="rdp",
        )

    def _setup_regularization_model(self, train_loader, loaded_pe_reg):
        if not self.train_parameters.regularization:
            return None, None, None
        (pm_reg, po_reg, _, pe_reg) = Utils.create_private_model(
            model=self.model_regularization,
            epsilon=self.train_parameters.epsilon,
            original_optimizer=self.optimizer_regularization,
            train_loader=train_loader,
            epochs=self.train_parameters.epochs,
            delta=self.delta,
            max_grad_norm=self.clipping,
            noise_multiplier=self.noise_multiplier,
            accountant=loaded_pe_reg,
        )
        pm_reg.to(self.train_parameters.device)
        return pm_reg, po_reg, pe_reg

    def _run_training_epochs(self, net, pm_reg, opt, po_reg, loader, fl_round, avg_prob, sigma, weights, max_disp):
        all_metrics, all_losses, history_lambda = [], [], []
        for epoch in range(self.train_parameters.epochs):
            metrics = Learning.train_private_model(
                train_parameters=self.train_parameters,
                model=net,
                model_regularization=pm_reg,
                optimizer=opt,
                optimizer_regularization=po_reg,
                train_loader=loader,
                _test_loader=None,
                average_probabilities=avg_prob,
                current_epoch=epoch,
                current_fl_round=fl_round,
                node_id=self.cid,
                sigma_update_lambda=sigma,
                epoch=epoch,
                reweighing_weights=weights,
            )
            metrics["Max Disparity Train Before Local Epoch"] = max_disp
            history_lambda.extend(metrics["history_lambda"])
            all_metrics.append(metrics)
            all_losses.append(metrics["Train Loss"])
        return all_metrics, all_losses, history_lambda

    def _save_privacy_state(self, pe, pe_reg):
        with (self.fed_dir / f"privacy_engine_{self.cid}.pkl").open("wb") as f:
            dill.dump(pe.accountant, f)
        if self.train_parameters.regularization:
            with (self.fed_dir / f"privacy_engine_regularization_{self.cid}.pkl").open("wb") as f:
                dill.dump(pe_reg.accountant, f)
            with (self.fed_dir / f"regularization_lambda_{self.cid}.pkl").open("wb") as f:
                dill.dump(self.train_parameters.regularization_lambda, f)

    def _compute_final_epsilon(self):
        if not self.original_epsilon:
            return float("inf")
        return (
            self.original_epsilon
            + (self.train_parameters.epsilon_lambda or 0)
            + (self.train_parameters.epsilon_statistics or 0)
        )

    def _compute_counters(self, net, loader):
        (preds, s_attrs, _, _, targets, s_features, _, _, _) = Learning.test_prediction(
            model=net,
            test_loader=loader,
            train_parameters=self.train_parameters,
        )
        probs, counters = RegularizationLoss.compute_probabilities(
            predictions=preds,
            sensitive_attribute_list=s_attrs,
            device=self.train_parameters.device,
            possible_sensitive_attributes=s_features,
            _possible_targets=targets,
        )
        counters_no_noise = copy.deepcopy(counters)
        if self.train_parameters.epsilon_statistics is not None:
            self._add_noise_to_counters(counters)
        return probs, counters, counters_no_noise

    def _add_noise_to_counters(self, counters):
        if not (self.fed_dir / "metadata.json").exists():
            return
        with (self.fed_dir / "metadata.json").open() as f:
            meta = json.load(f)
        combs = meta["combinations"]
        sigma = get_noise_multiplier(
            target_epsilon=self.train_parameters.epsilon_statistics,
            target_delta=self.delta,
            sample_rate=1,
            steps=self.sampling_frequency * len(combs),
            accountant="rdp",
        )
        for key in counters:
            if key in combs:
                counters[key] += Utils.get_noise(mechanism_type="gaussian", sigma=sigma)

    def evaluate(self, parameters, config):
        if (self.fed_dir / "avg_proba.pkl").exists():
            with (self.fed_dir / "avg_proba.pkl").open("rb") as file:
                avg_prob = dill.load(file)  # noqa: S301
        else:
            avg_prob = None
        Utils.set_params(self.net, parameters)
        num_workers = int(ray.get_runtime_context().get_assigned_resources()["CPU"])
        partition = (
            "test"
            if self.train_parameters.cross_silo and not self.train_parameters.sweep
            else ("val" if self.train_parameters.sweep else "train")
        )
        dataset = Utils.get_dataloader(
            self.fed_dir,
            self.cid,
            batch_size=self.train_parameters.batch_size,
            workers=num_workers,
            dataset=self.dataset_name,
            partition=partition,
        )
        self.net.to(self.train_parameters.device)

        res = Learning.test(self.net, dataset, self.train_parameters, avg_prob)
        res2 = Learning.test_2(self.net, dataset, self.train_parameters, avg_prob)
        res3 = Learning.test_3(self.net, dataset, self.train_parameters, avg_prob)

        (preds, s1, s2, s3, targets, f1, f2, f3, _) = Learning.test_prediction(self.net, dataset, self.train_parameters)
        p1, c1 = RegularizationLoss.compute_probabilities(preds, s1, self.train_parameters.device, f1, targets)
        p2, c2 = RegularizationLoss.compute_probabilities(preds, s2, self.train_parameters.device, f2, targets)
        p3, c3 = RegularizationLoss.compute_probabilities(preds, s3, self.train_parameters.device, f3, targets)

        self.net.to("cpu")
        gc.collect()

        metrics = self._get_metrics_dict(res, res2, res3, p1, p2, p3, c1, c2, c3)
        return float(res[0]), len(dataset.dataset), metrics

    def _get_metrics_dict(self, r1, r2, r3, p1, p2, p3, c1, c2, c3):
        is_sex = self.train_parameters.sensitive_attribute == "SEX"
        if self.train_parameters.sweep:
            return {
                "validation_accuracy": float(r1[1]),
                "max_disparity_validation": float(r1[5]) if is_sex else float(r2[5]),
                "max_disparity_validation_second": float(r2[5]) if is_sex else float(r1[5]),
                "max_disparity_validation_third": float(r3[5]) if is_sex else float(r1[5]),
                "validation_loss": r1[0],
                "probabilities": p1,
                "second_probabilities": p2,
                "cid": self.cid,
                "counters": c1 if is_sex else c2,
                "second_counters": c2 if is_sex else c1,
                "third_counters": c3,
                "f1_score": r1[2],
                "max_group_validation": r1[9],
                "max_group_validation_second": r2[7],
                "max_group_validation_third": r3[9],
            }
        return {
            "test_accuracy": float(r1[1]),
            "max_disparity_test": float(r1[5]),
            "max_disparity_test_second": float(r2[5]),
            "max_disparity_test_third": float(r3[5]),
            "test_loss": r1[0],
            "probabilities": p1,
            "second_probabilities": p2,
            "third_probabilities": p3,
            "cid": self.cid,
            "counters": c1,
            "second_counters": c2,
            "third_counters": c3,
            "f1_score": r1[2],
            "max_group_test": r1[9],
            "max_group_test_second": r2[7],
            "max_group_test_third": r3[9],
        }

    def compute_starting_lambda_with_avg(self):
        loaded_clients_list = []
        if (self.fed_dir / "clients_last_round.pkl").exists():
            with (self.fed_dir / "clients_last_round.pkl").open("rb") as file:
                loaded_clients_list = dill.load(file)  # noqa: S301
        lambda_list = []
        if loaded_clients_list:
            for client_cid in loaded_clients_list:
                if (self.fed_dir / f"regularization_lambda_{client_cid}.pkl").exists():
                    with (self.fed_dir / f"regularization_lambda_{client_cid}.pkl").open("rb") as file:
                        loaded_lambda = dill.load(file)  # noqa: S301
                        lambda_list.append(loaded_lambda)
        return np.mean(lambda_list) if lambda_list else 0

    def compute_starting_lambda_with_disparity(self, disparity_training: float):
        """
        This function computes the starting Lambda based on
        the disparity of the training dataset and the target disparity.
        Given a certain target disparity and the actual disparity of the training
        dataset, what we do is to compute the difference between the two values.
        If the difference is positive, it means that we want to use a Lambda = 0.
        If the difference is negative then we can use the difference as a Lambda but
        instead of using it directly we have to rescale it in the range [0, 1].

        We need to rescale it because when we compute the difference we can have a negative
        value that is in the range [0, 1 - target_disparity]. However, we want to use the
        dellta in the range [0, 1] so we have to rescale it. To rescale it we have
        a function and we have to use as old min value 0 and as old max value 1 - target_disparity.

        Even if it seems that this is a small detail, it is important to rescale the lambda
        in the correct way because starting with a wrong lambda can lead to a wrong
        regularization and so to a wrong model. Even if the lambda is updated during the
        training, the starting value is important.
        """
        delta = self.train_parameters.target - disparity_training
        if delta > 0:
            return 0
        return Utils.rescale_lambda(
            value=abs(delta),
            old_min=0,
            old_max=1 - self.train_parameters.target,
            new_min=0,
            new_max=1,
        )

    def get_noise(self, dataset, target_epsilon=None):
        model_noise = ModelUtils.get_model(self.dataset_name, _device=self.train_parameters.device)
        privacy_engine = PrivacyEngine(accountant="rdp")
        optimizer_noise = Utils.get_optimizer(model_noise, self.train_parameters, self.lr)
        (
            _,
            private_optimizer,
            _,
        ) = privacy_engine.make_private_with_epsilon(
            module=model_noise,
            optimizer=optimizer_noise,
            data_loader=dataset,
            epochs=self.sampling_frequency * self.train_parameters.epochs,
            target_epsilon=self.train_parameters.epsilon if target_epsilon is None else target_epsilon,
            target_delta=self.delta,
            max_grad_norm=self.clipping,
        )

        return private_optimizer.noise_multiplier
