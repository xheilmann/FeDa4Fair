import numpy as np
import torch
from torch import nn
from torch.nn import functional


class RegularizationLoss(nn.Module):
    """
    This class defines the regularization loss as proposed in
    https://arxiv.org/abs/2302.09183.
    It uses the definition of demographic parity to compute the
    fairness violation term for each batch and then it uses this
    violation term as a regularization term to add to the loss.
    """

    def __init__(self, weight=None, size_average=True, estimation=0.5) -> None:
        """Initialization of the regularization loss."""
        super().__init__()
        self.estimation = estimation

    def forward(
        self,
        sensitive_attribute_list: torch.tensor,
        device: torch.device,
        predictions: torch.tensor,
        possible_sensitive_attributes: list,
        possible_targets: list,
        average_probabilities: dict | None = None,
        wandb_run=None,
        batch=None,
        return_group: bool = False,
    ) -> torch.tensor:
        """
        This function computes the regularization term.

        Args:
            sensitive_attribute_list (np.array): a list with the value of
                the sensitive attribute for each sample in the batch
            device (str): the device we're using to train the model (gpu or cpu)
            predictions (np.array): the predictions of the model for the batch of data
            possible_sensitive_attributes (list): the possible values of the sensitive
                attribute
            possible_targets (list): the possible target values we have in this
                dataset
            average_probabilities (dict): in case of Federated learning, if a client
                has only a subset of the possible sensitive attributes, we can use the
                average probabilities of the other clients to estimate the probabilities
                of the missing sensitive attributes. This is None in centralised learning
            wandb_run (wandb.Run): the wandb run we're using to log the metrics
            batch (int): the number of the batch we're considering
            return_group (bool): If True, returns the group (target, sensitive_attribute)
                that caused the maximum violation. Defaults to False.

        Returns:
            float: the disparity metric computed on the data passed as parameter

        """
        fairness_violations = []
        softmax_ = functional.softmax(predictions, dim=1)

        sens_attr_tensor = torch.tensor([int(item) for item in sensitive_attribute_list]).to(device)
        predictions_argmax = torch.argmax(torch.tensor(predictions), dim=1).to(device)

        targets = [int(item) for item in possible_targets]
        sensitive_attrs = [int(item) for item in possible_sensitive_attributes]

        for target in targets:
            for z in sensitive_attrs:
                Z_eq_z = len(sens_attr_tensor[sens_attr_tensor == z])
                Z_not_eq_z = len(sens_attr_tensor[sens_attr_tensor != z])

                Y_eq_k_and_Z_eq_z = torch.sum(
                    softmax_[(predictions_argmax == target) & (sens_attr_tensor == z)][:, target]
                )
                Y_eq_k_and_Z_not_eq_z = torch.sum(
                    softmax_[(predictions_argmax == target) & (sens_attr_tensor != z)][:, target]
                )

                if (Y_eq_k_and_Z_eq_z == 0 and Y_eq_k_and_Z_not_eq_z != 0) or (Z_eq_z == 0 and Z_not_eq_z != 0):
                    denominator = 1 if z == 1 else 0
                    if average_probabilities and average_probabilities.get(f"{target}|{denominator}", None):
                        violation_term = torch.abs(
                            average_probabilities[f"{target}|{denominator}"] - Y_eq_k_and_Z_not_eq_z / Z_not_eq_z
                        )
                    else:
                        violation_term = torch.tensor(0.0).to(device)
                elif (Y_eq_k_and_Z_eq_z != 0 and Y_eq_k_and_Z_not_eq_z == 0) or (Z_not_eq_z == 0 and Z_eq_z != 0):
                    denominator = 1 if z == 0 else 0
                    if average_probabilities and average_probabilities.get(f"{target}|{denominator}", None):
                        violation_term = torch.abs(
                            (Y_eq_k_and_Z_eq_z / Z_eq_z) - average_probabilities[f"{target}|{denominator}"]
                        )
                    else:
                        violation_term = torch.tensor(0.0).to(device)
                else:
                    violation_term = torch.abs((Y_eq_k_and_Z_eq_z / Z_eq_z) - (Y_eq_k_and_Z_not_eq_z / Z_not_eq_z))

                fairness_violations.append(violation_term)

        fairness_violations_val = [item.item() if isinstance(item, torch.Tensor) else item for item in fairness_violations]
        max_idx = fairness_violations_val.index(max(fairness_violations_val))

        fairness_violations_tensor = torch.stack(fairness_violations)
        mask = torch.zeros(fairness_violations_tensor.shape[0]).to(device)
        mask[max_idx] = 1
        res = torch.sum(mask * fairness_violations_tensor)

        if return_group:
            target_idx = max_idx // len(sensitive_attrs)
            z_idx = max_idx % len(sensitive_attrs)
            return res, (targets[target_idx], sensitive_attrs[z_idx])

        return res

    def violation_with_dataset(
        self,
        model: torch.nn.Module,
        dataset: torch.utils.data.DataLoader,
        average_probabilities: dict,
        device: torch.device,
        return_group: bool = False,
    ) -> torch.tensor:
        """Compute disparity metric on the entire dataset."""
        predictions = torch.tensor([]).to(device)
        sensitive_attribute_list = torch.tensor([]).to(device)
        targets = []
        model.eval()
        with torch.no_grad():
            for batch_images, batch_sensitive_attributes, _, _, batch_target in dataset:
                output = model(batch_images.to(device))
                predictions = torch.cat((predictions, output), 0)
                sensitive_attribute_list = torch.cat((sensitive_attribute_list, batch_sensitive_attributes.to(device)), 0)
                targets += batch_target.tolist()

        return self.forward(
            sensitive_attribute_list, device, predictions,
            list({item.item() for item in sensitive_attribute_list}),
            list(set(targets)), average_probabilities=average_probabilities,
            return_group=return_group,
        )

    def violation_with_dataset_2(
        self,
        model: torch.nn.Module,
        dataset: torch.utils.data.DataLoader,
        average_probabilities: dict,
        device: torch.device,
        return_group: bool = False,
    ) -> torch.tensor:
        """Compute disparity metric on the entire dataset using second sensitive attribute."""
        predictions = torch.tensor([]).to(device)
        sensitive_attribute_list = torch.tensor([]).to(device)
        targets = []
        model.eval()
        with torch.no_grad():
            for batch_images, _, batch_sensitive_attributes, _, batch_target in dataset:
                output = model(batch_images.to(device))
                predictions = torch.cat((predictions, output), 0)
                sensitive_attribute_list = torch.cat((sensitive_attribute_list, batch_sensitive_attributes.to(device)), 0)
                targets += batch_target.tolist()

        return self.forward(
            sensitive_attribute_list, device, predictions,
            list({item.item() for item in sensitive_attribute_list}),
            list(set(targets)), average_probabilities=average_probabilities,
            return_group=return_group,
        )

    def violation_with_dataset_3(
        self,
        model: torch.nn.Module,
        dataset: torch.utils.data.DataLoader,
        average_probabilities: dict,
        device: torch.device,
        return_group: bool = False,
    ) -> torch.tensor:
        """Compute disparity metric on the entire dataset using third sensitive attribute."""
        predictions = torch.tensor([]).to(device)
        sensitive_attribute_list = torch.tensor([]).to(device)
        targets = []
        model.eval()
        with torch.no_grad():
            for batch_images, _, _, batch_sensitive_attributes, batch_target in dataset:
                output = model(batch_images.to(device))
                predictions = torch.cat((predictions, output), 0)
                sensitive_attribute_list = torch.cat((sensitive_attribute_list, batch_sensitive_attributes.to(device)), 0)
                targets += batch_target.tolist()

        return self.forward(
            sensitive_attribute_list, device, predictions,
            list({item.item() for item in sensitive_attribute_list}),
            list(set(targets)), average_probabilities=average_probabilities,
            return_group=return_group,
        )

    def compute_violation_with_argmax(
        self,
        predictions_argmax: torch.tensor,
        sensitive_attribute_list: torch.tensor,
        current_target: int,
        current_sensitive_feature: int,
        weights: dict | None = None,
    ):
        """Debug function used to compute the DPL using the argmax function."""
        opp_sens_feature = 0 if current_sensitive_feature == 1 else 1
        Z_eq_z_argmax, Z_not_eq_z_argmax = 0, 0
        Y_eq_k_and_Z_eq_z_argmax, Y_eq_k_and_Z_not_eq_z_argmax = 0, 0

        for pred, sf in zip(predictions_argmax, sensitive_attribute_list, strict=False):
            w = weights.get(f"(Y={int(pred)}, Z={int(sf)})", 1) if weights else 1
            if sf == current_sensitive_feature:
                Z_eq_z_argmax += w
                if pred == current_target:
                    Y_eq_k_and_Z_eq_z_argmax += w
            else:
                Z_not_eq_z_argmax += w
                if pred == current_target and sf == opp_sens_feature:
                    Y_eq_k_and_Z_not_eq_z_argmax += w

        if Z_eq_z_argmax == 0 and Z_not_eq_z_argmax != 0:
            return np.abs(Y_eq_k_and_Z_not_eq_z_argmax / Z_not_eq_z_argmax).item()
        if Z_eq_z_argmax != 0 and Z_not_eq_z_argmax == 0:
            return np.abs(Y_eq_k_and_Z_eq_z_argmax / Z_eq_z_argmax).item()
        if Z_eq_z_argmax == 0 and Z_not_eq_z_argmax == 0:
            return 0
        return np.abs(
            Y_eq_k_and_Z_eq_z_argmax / Z_eq_z_argmax - Y_eq_k_and_Z_not_eq_z_argmax / Z_not_eq_z_argmax
        ).item()

    @staticmethod
    def compute_probabilities(
        predictions,
        sensitive_attribute_list,
        device: torch.device,
        possible_sensitive_attributes: list,
        _possible_targets: list,
    ) -> tuple[dict, dict]:
        """Compute the probabilities and the counters of each possible combination."""
        softmax_ = functional.softmax(predictions, dim=1)
        predictions_argmax = torch.argmax(torch.tensor(predictions), dim=1).to(device)
        sens_attr_tensor = torch.tensor([int(item) for item in sensitive_attribute_list]).to(device)

        probabilities, counters = {}, {}
        sensitive_attrs = [int(item) for item in possible_sensitive_attributes]

        for z_val in sensitive_attrs:
            target_val = 1
            Z_eq_z = len(sens_attr_tensor[sens_attr_tensor == z_val])
            Y_eq_k_and_Z_eq_z = torch.sum(softmax_[(predictions_argmax == target_val) & (sens_attr_tensor == z_val)][:, target_val])
            Y_eq_k_and_Z_eq_z_argmax = len(predictions_argmax[(predictions_argmax == target_val) & (sens_attr_tensor == z_val)])

            probabilities[f"{target_val}|{z_val}"] = Y_eq_k_and_Z_eq_z
            probabilities[f"{z_val}"] = Z_eq_z
            counters[f"{target_val}|{z_val}"] = Y_eq_k_and_Z_eq_z_argmax
            counters[f"{z_val}"] = Z_eq_z

        return probabilities, counters
