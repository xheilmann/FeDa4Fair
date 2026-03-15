"""
This file implements the Learning class that is used to train the model
If you want to train the model using the unfairness mitigation through
Regularization and Differential Privacy, you need to use this class for the training
(or implement something similar to this one)
"""

import numpy as np
import torch
from opacus.utils.batch_memory_manager import BatchMemoryManager
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn

from ..DPLUtils.regularization_config import RegularizationConfig
from ..DPLUtils.utils import Utils
from ..Regularization.RegularizationLoss import RegularizationLoss


def exp_lr_scheduler(initial_alpha, current_fl_round, decay_rate=0.001):
    """
    Decay learning rate by a factor of decay_rate every epoch.

    Args:
        initial_alpha (float): initial learning rate
        current_fl_round (int): the current fl round in which the client was selected
        decay_rate (float, optional): decay rate. Defaults to 0.001.

    """
    return initial_alpha * decay_rate ** (current_fl_round + 1)


class Learning:
    @staticmethod
    def train_private_model(
        train_parameters: RegularizationConfig,
        model: torch.nn.Module,
        model_regularization: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        optimizer_regularization: torch.optim.Optimizer,
        train_loader: torch.utils.data.DataLoader,
        _test_loader: torch.utils.data.DataLoader,
        average_probabilities: dict | None = None,
        current_epoch: int | None = None,
        current_fl_round: int | None = None,
        _max_num_epochs: int | None = None,
        node_id: int = 0,
        wandb_run=None,
        sigma_update_lambda: float | None = None,
        epoch: int = 0,
        reweighing_weights: dict | None = None,
    ) -> dict:
        """
        This function is used to train the private model.

        Args:
            train_parameters (RegularizationConfig): the configuration of all the possible
                settings to train the model
            model (torch.nn.Module): the model to train
            model_regularization (torch.nn.Module): the model used to compute
                the regularization term.
            optimizer (torch.optim.Optimizer): the optimizer used to train the
                model
            optimizer_regularization (torch.optim.Optimizer): the optimizer used
                to train the model_regularization
            train_loader (torch.utils.data.DataLoader): the training set
            _test_loader (torch.utils.data.DataLoader): the test set (unused)
            average_probabilities (dict): the average probabilities computed by the server
            current_epoch (int): the current epoch
            current_fl_round (int): the current fl round in which the client was selected
            _max_num_epochs (int): the maximum number of epochs to train the model (unused)
            node_id (int): id of the node. this is used only in FL
            wandb_run (wandb.Run): the wandb run used to log the metrics
            sigma_update_lambda (float): the sigma parameter used to update the lambda
            epoch (int): the current epoch
            reweighing_weights (dict): weights for reweighing the loss. Defaults to None

        """
        if train_parameters.metric != "disparity":
            msg = "The metric is not supported"
            raise ValueError(msg)
        criterion_reg = RegularizationLoss()
        losses, losses_with_reg, total_correct, total, velocity = [], [], 0, 0, 0
        history_lambda = []

        model.train()
        if model_regularization:
            model_regularization.train()

        with BatchMemoryManager(
            data_loader=train_loader,
            max_physical_batch_size=512,
            optimizer=optimizer,
        ) as memory_safe_loader:
            for batch_number, (batch_data, sens_1, sens_2, _, batch_target) in enumerate(memory_safe_loader, 0):
                sens_feature = Learning._get_sensitive_feature(train_parameters, sens_1, sens_2)
                if model_regularization is not None:
                    Utils.sync_models(model_regularization, model)

                optimizer.zero_grad()
                target, data = batch_target.long().to(train_parameters.device), batch_data.to(train_parameters.device)
                sens_feature = sens_feature.to(train_parameters.device)

                reg_results = Learning._handle_regularization(
                    train_parameters,
                    model_regularization,
                    data,
                    target,
                    sens_feature,
                    criterion_reg,
                    average_probabilities,
                    epoch,
                    batch_number,
                    node_id,
                    wandb_run,
                )
                reg_term, fairness_violation = reg_results["regularization_term"], reg_results["fairness_violation"]

                outputs = model(data)
                history_lambda.append(train_parameters.regularization_lambda)
                classic_loss = Learning._compute_classic_loss(
                    outputs, target, sens_feature, train_parameters, reweighing_weights
                )
                loss = Learning._handle_backward_and_gradients(
                    train_parameters, reg_term, classic_loss, model, model_regularization, optimizer
                )

                losses.append(loss.item())
                if reg_term is not None:
                    losses_with_reg.append((reg_term + loss).item())
                if optimizer_regularization:
                    optimizer_regularization.zero_grad()

                total_correct += (outputs.argmax(dim=1) == target).float().sum()
                total += target.size(0)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if (
                    train_parameters.regularization_mode == "tunable"
                    and fairness_violation is not None
                    and train_parameters.target
                ):
                    velocity = Learning._update_lambda_tunable(
                        train_parameters,
                        model,
                        data,
                        target,
                        sens_feature,
                        criterion_reg,
                        average_probabilities,
                        epoch,
                        batch_number,
                        sigma_update_lambda,
                        velocity,
                    )

        train_parameters.alpha = Learning._update_alpha(train_parameters, current_fl_round)
        max_unfairness_train = criterion_reg.violation_with_dataset(
            model=model,
            dataset=train_loader,
            device=train_parameters.device,
            average_probabilities=average_probabilities,
        )
        return {
            "epoch": current_epoch,
            "Train Loss": np.mean(losses),
            "Train Loss + Regularizaion": np.mean(losses_with_reg),
            "Train Accuracy": total_correct / total,
            "Max Unfairness Train": max_unfairness_train,
            "history_lambda": history_lambda,
        }

    @staticmethod
    def _get_sensitive_feature(train_parameters, sens_1, sens_2):
        if train_parameters.sensitive_attribute == "SEX":
            return sens_1
        if train_parameters.sensitive_attribute == "MAR":
            return sens_2
        return None

    @staticmethod
    def _handle_regularization(
        train_parameters,
        model_regularization,
        data,
        target,
        sensitive_feature,
        criterion_regularization,
        average_probabilities,
        epoch,
        batch_number,
        node_id,
        wandb_run,
    ):
        regularization_term = None
        fairness_violation = None
        if train_parameters.regularization and model_regularization:
            output_regularization = model_regularization(data)
            fairness_violation = Learning.compute_regularization_term(
                _data=None,  # data is unused
                targets=target,
                sensitive_feature=sensitive_feature,
                train_parameters=train_parameters,
                criterion_regularization=criterion_regularization,
                outputs=output_regularization,
                average_probabilities=average_probabilities,
                batch=(epoch + 1) * batch_number,
            )

            if fairness_violation is not None and fairness_violation > 0:
                regularization_term = train_parameters.regularization_lambda * fairness_violation
                try:
                    regularization_term.backward()
                except RuntimeError:
                    print(
                        f"EXCEPTION while computing the backward pass: Node id {node_id} "
                        f"Outputs: {len(data)} target: {len(target)} "
                        f"sensitive_attribute_list: {len(sensitive_feature)} "
                        f"FAIRNESS VIOLATION: {fairness_violation} reg term: {regularization_term} "
                        f"current sens features: {[item.item() for item in sensitive_feature]}"
                    )

        if wandb_run:
            val = (
                regularization_term.item()
                if isinstance(regularization_term, torch.Tensor)
                else (regularization_term or 0)
            )
            wandb_run.log(
                {
                    "batch": (epoch + 1) * batch_number,
                    "Unfairness metric Batch": val,
                }
            )

        return {"regularization_term": regularization_term, "fairness_violation": fairness_violation}

    @staticmethod
    def _compute_classic_loss(outputs, target, sensitive_feature, train_parameters, reweighing_weights):
        if reweighing_weights is not None:
            per_sample_loss = nn.CrossEntropyLoss(reduction="none")(outputs, target)
            batch_weights = torch.tensor(
                [
                    reweighing_weights.get((z.item(), y.item()), 1.0)
                    for z, y in zip(sensitive_feature, target, strict=False)
                ]
            ).to(train_parameters.device)
            return (per_sample_loss * batch_weights).mean()
        return nn.CrossEntropyLoss()(outputs, target)

    @staticmethod
    def _handle_backward_and_gradients(
        train_parameters, regularization_term, classic_loss, model, model_regularization, optimizer
    ):
        loss = (
            (1 - train_parameters.regularization_lambda) * classic_loss
            if regularization_term is not None
            else classic_loss
        )
        loss.backward()
        if regularization_term and train_parameters.regularization_lambda > 0:
            for p1, p2 in zip(model.parameters(), model_regularization.parameters(), strict=False):
                if p1.grad_sample is not None and p2.grad_sample is not None:
                    p1.grad_sample += p2.grad_sample
        optimizer.step()
        optimizer.zero_grad()
        return loss

    @staticmethod
    def _update_lambda_tunable(
        train_parameters,
        model,
        data,
        target,
        sensitive_feature,
        criterion_regularization,
        average_probabilities,
        epoch,
        batch_number,
        sigma_update_lambda,
        velocity,
    ):
        model.eval()
        with torch.no_grad():
            output_reg = model(data)
            fairness_violation = Learning.compute_regularization_term(
                _data=None,
                targets=target,
                sensitive_feature=sensitive_feature,
                train_parameters=train_parameters,
                criterion_regularization=criterion_regularization,
                outputs=output_reg,
                average_probabilities=average_probabilities,
                batch=(epoch + 1) * batch_number,
            )
        model.train()
        noise = Utils.get_noise(mechanism_type="gaussian", sigma=sigma_update_lambda) if sigma_update_lambda else 0
        delta = train_parameters.target - (fairness_violation.item() + noise)
        velocity = train_parameters.momentum * velocity + delta
        train_parameters.regularization_lambda = np.clip(
            train_parameters.regularization_lambda - train_parameters.alpha * velocity, 0, 1
        )
        return velocity

    @staticmethod
    def _update_alpha(train_parameters, current_fl_round):
        if train_parameters.weight_decay_alpha:
            return exp_lr_scheduler(
                initial_alpha=train_parameters.alpha,
                current_fl_round=current_fl_round,
                decay_rate=train_parameters.weight_decay_alpha,
            )
        return train_parameters.alpha

    @staticmethod
    def test(
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        train_parameters: RegularizationConfig,
        average_probabilities=None,
    ) -> tuple:
        """Test the model."""
        return Learning._test_base(
            model,
            test_loader,
            train_parameters,
            average_probabilities,
            sensitive_attr_index=1,
            violation_method="violation_with_dataset",
        )

    @staticmethod
    def test_2(
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        train_parameters: RegularizationConfig,
        average_probabilities=None,
    ) -> tuple:
        """Test the model using second sensitive attribute."""
        res = Learning._test_base(
            model,
            test_loader,
            train_parameters,
            average_probabilities,
            sensitive_attr_index=2,
            violation_method="violation_with_dataset_2",
        )
        return (res[0], res[1], res[2], res[3], res[4], res[5], res[6], res[9])

    @staticmethod
    def test_3(
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        train_parameters: RegularizationConfig,
        average_probabilities=None,
    ) -> tuple:
        """Test the model using third sensitive attribute."""
        return Learning._test_base(
            model,
            test_loader,
            train_parameters,
            average_probabilities,
            sensitive_attr_index=3,
            violation_method="violation_with_dataset_3",
        )

    @staticmethod
    def _test_base(
        model, test_loader, train_parameters, average_probabilities, sensitive_attr_index, violation_method
    ) -> tuple:
        model.eval()
        criterion = nn.CrossEntropyLoss()
        correct, total = 0, 0
        y_pred, y_true, losses, colors = [], [], [], []
        with torch.no_grad():
            for batch in test_loader:
                batch_data, batch_target = (
                    batch[0].to(train_parameters.device),
                    batch[4].long().to(train_parameters.device),
                )
                color = batch[sensitive_attr_index]
                output = model(batch_data)
                total += batch_target.size(0)
                losses.append(criterion(output, batch_target).item())
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(batch_target.view_as(pred)).sum().item()
                y_pred.extend(pred)
                y_true.extend(batch_target)
                colors.extend([item.item() for item in color])
        unfairness_test, max_group = 0, None
        if train_parameters.metric == "disparity":
            method = getattr(RegularizationLoss(), violation_method)
            unfairness_test, max_group = method(
                model=model,
                dataset=test_loader,
                device=train_parameters.device,
                average_probabilities=average_probabilities,
                return_group=True,
            )
        y_true_list, y_pred_list = [item.item() for item in y_true], [item.item() for item in y_pred]
        return (
            np.mean(losses),
            correct / total,
            f1_score(y_true_list, y_pred_list, average="macro"),
            precision_score(y_true_list, y_pred_list, average="macro"),
            recall_score(y_true_list, y_pred_list, average="macro"),
            unfairness_test,
            y_true_list,
            y_pred_list,
            colors,
            max_group,
        )

    @staticmethod
    def compute_regularization_term(
        _data: torch.utils.data.DataLoader | None,
        targets: torch.Tensor,
        sensitive_feature: torch.Tensor,
        criterion_regularization: RegularizationLoss,
        train_parameters: RegularizationConfig,
        outputs: torch.Tensor,
        average_probabilities: dict | None,
        wandb_run=None,
        batch: int = 0,
    ) -> torch.Tensor:
        """Compute the regularization term."""
        if train_parameters.metric != "disparity":
            msg = "The metric is not supported"
            raise ValueError(msg)
        return criterion_regularization(
            sensitive_attribute_list=sensitive_feature,
            device=train_parameters.device,
            predictions=outputs,
            possible_sensitive_attributes=list({item.item() for item in sensitive_feature}),
            possible_targets=list({item.item() for item in targets}),
            average_probabilities=average_probabilities,
            wandb_run=wandb_run,
            batch=batch,
        )

    @staticmethod
    def test_prediction(
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        train_parameters: RegularizationConfig,
    ) -> tuple:
        """Test the model returning raw predictions."""
        model.eval()
        y_true, predictions, sensitive_attributes, second_sensitive_attributes, third_sensitive_attributes = (
            [],
            [],
            [],
            [],
            [],
        )
        with torch.no_grad():
            for batch_data, s1, s2, s3, batch_target in test_loader:
                target, data = batch_target.long().to(train_parameters.device), batch_data.to(train_parameters.device)
                predictions.append(model(data))
                y_true.extend(target)
                sensitive_attributes.extend([item.item() for item in s1])
                second_sensitive_attributes.extend([item.item() for item in s2])
                third_sensitive_attributes.extend([item.item() for item in s3])
        y_true_items = [item.item() for item in y_true]
        return (
            torch.cat(predictions, dim=0),
            torch.tensor(sensitive_attributes),
            torch.tensor(second_sensitive_attributes),
            torch.tensor(third_sensitive_attributes),
            set(y_true_items),
            set(sensitive_attributes),
            set(second_sensitive_attributes),
            set(third_sensitive_attributes),
            y_true_items,
        )
