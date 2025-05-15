import copy
import json
import os
from itertools import product

import dill
import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame, false_positive_rate, selection_rate, true_positive_rate
from sklearn.metrics import confusion_matrix

# def demographic_disparity_by_group(y_pred, sensitive_features):
#     """
#     Calculates demographic disparity (statistical parity difference) across sensitive groups,
#     and identifies the pair of groups with the maximum disparity.

#     Parameters:
#     y_pred (np.array): Predicted labels (binary: 0 or 1).
#     sensitive_features (np.array or pd.Series): Sensitive feature(s) defining the groups.

#     Returns:
#     tuple:
#         float: The maximum demographic disparity.
#         tuple: The pair of sensitive groups (group_a, group_b) with the maximum disparity.
#     """
#     if not isinstance(sensitive_features, pd.Series):
#         sensitive_features = pd.Series(sensitive_features)

#     unique_groups = sensitive_features.unique()
#     print(unique_groups)
#     positive_rates = {}

#     # Calculate positive prediction rate for each group
#     for group in unique_groups:
#         group_indices = sensitive_features[sensitive_features == group].index
#         y_pred_group = y_pred[group_indices]
#         positive_rate = np.mean(y_pred_group)  # Pr(Ŷ=1 | A=group)
#         positive_rates[group] = positive_rate

#     # Find the maximum absolute difference and the group pair responsible
#     max_disparity = 0.0
#     responsible_pair = (None, None)

#     for i, group_a in enumerate(unique_groups):
#         for group_b in unique_groups[i + 1 :]:
#             disparity = abs(positive_rates[group_a] - positive_rates[group_b])
#             if disparity > max_disparity:
#                 max_disparity = disparity
#                 responsible_pair = (group_a, group_b)

#     return max_disparity, responsible_pair


# def equalized_odds_difference_by_outcome(y_true, y_pred, sensitive_features):
#     """
#     Calculates the equalized odds difference, considering outcomes separately for each group.

#     Parameters:
#     y_true (np.array): True labels.
#     y_pred (np.array): Predicted labels.
#     sensitive_features (np.array): Sensitive feature(s) that define the groups.

#     Returns:
#     float: The equalized odds difference.
#     """
#     if not isinstance(sensitive_features, pd.Series):
#         sensitive_features = pd.Series(sensitive_features)

#     unique_groups = sensitive_features.unique()
#     possible_outcomes = np.unique(y_true)

#     tpr_values = {group: {} for group in unique_groups}
#     fpr_values = {group: {} for group in unique_groups}

#     for group in unique_groups:
#         group_indices = sensitive_features[sensitive_features == group].index
#         y_true_group = y_true[group_indices]
#         y_pred_group = y_pred[group_indices]

#         tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group).ravel()
#         tpr_values[group][1] = tp / (tp + fn) if (tp + fn) > 0 else 0  # TPR for outcome 1
#         fpr_values[group][1] = fp / (fp + tn) if (fp + tn) > 0 else 0  # FPR for outcome 1
#         # For binary classification where outcomes are 0 and 1, we can also consider metrics for the negative outcome (0)
#         if len(possible_outcomes) == 2:
#             tn_r, fn_r, fp_r, tp_r = confusion_matrix(1 - y_true_group, 1 - y_pred_group).ravel()
#             tpr_values[group][0] = tp_r / (tp_r + fp_r) if (tp_r + fp_r) > 0 else 0  # TNR (TPR for outcome 0)
#             fpr_values[group][0] = fn_r / (fn_r + tn_r) if (fn_r + tn_r) > 0 else 0  # FNR (FPR for outcome 0)

#     tpr_diffs = []
#     fpr_diffs = []

#     for outcome in possible_outcomes:
#         tprs_outcome = [tpr_values[group].get(outcome, 0) for group in unique_groups]
#         fprs_outcome = [fpr_values[group].get(outcome, 0) for group in unique_groups]

#         tpr_diffs.append(max(tprs_outcome) - min(tprs_outcome))
#         fpr_diffs.append(max(fprs_outcome) - min(fprs_outcome))

#     return max(abs(max(tpr_diffs)), abs(max(fpr_diffs)))


class AggregationFunctions:
    def agg_metrics_test(
        metrics: list,
        server_round: int,
        train_parameters,
        wandb_run,
        args,
        fed_dir: str,
    ) -> dict:
        total_examples = sum([n_examples for n_examples, _ in metrics])

        for _, metric in metrics:
            if "y_true" in metric:
                node_name = metric["cid"]
                y_true = np.array([int(item) for item in metric["y_true"]])
                y_pred = np.array(metric["y_pred"])
                sensitive_attribute_1 = np.array(list(metric["sensitive_attributes_1"]))
                sensitive_attribute_2 = np.array(list(metric["sensitive_attributes_2"]))
                sensitive_attribute_3 = np.array(list(metric["sensitive_attributes_3"]))

                # # Calculate the equalized odds difference
                # equalized_odds_diff = equalized_odds_difference_by_outcome(
                #     y_true=y_true,
                #     y_pred=y_pred,
                #     sensitive_features=sensitive_attribute_1,
                # )
                # # print(f" ------> Test Node {node_name}, Equalized Odds Difference: {equalized_odds_diff}")
                # # Calculate the equalized odds difference
                # equalized_odds_diff_second = equalized_odds_difference_by_outcome(
                #     y_true=y_true,
                #     y_pred=y_pred,
                #     sensitive_features=sensitive_attribute_2,
                # )
                # # print(f" ------> Test Node {node_name}, Equalized Odds Difference Second: {equalized_odds_diff_second}")
                # # Calculate the equalized odds difference
                # # equalized_odds_diff_third = equalized_odds_difference_by_outcome(
                # #     y_true=y_true,
                # #     y_pred=y_pred,
                # #     sensitive_features=sensitive_attribute_3,
                # # )
                # # print(f" ------> Test Node {node_name}, Equalized Odds Difference Third: {equalized_odds_diff_third}")
                # agg_metrics = {
                #     f"Test Node {node_name} - EO.": equalized_odds_diff,
                #     f"Test Node {node_name} - Second EO.": equalized_odds_diff_second,
                #     # f"Test Node {node_name} - Third EO.": equalized_odds_diff_third,
                #     "FL Round": server_round,
                # }
                # if wandb_run:
                #     wandb_run.log(agg_metrics)

                def dem_parity_by_group(sens_attribute_list: list, y_true: list, y_pred):
                    sf_data = pd.DataFrame({"DP_RACE": sens_attribute_list})

                    sel_rate = MetricFrame(
                        metrics={"sel": selection_rate},
                        y_true=y_true,
                        y_pred=y_pred,
                        sensitive_features=sf_data,
                    )
                    df = sel_rate.by_group
                    diff_matrix = df["sel"].values[:, None] - df["sel"].values[None, :]
                    index = df.index.values
                    column_names = [f"{index[i]}_{index[j]}" for i, j in product(range(len(df)), repeat=2)]

                    diff_df = pd.Series(diff_matrix.flatten(), index=column_names)
                    diff_df = pd.Series([diff_df.max(), diff_df.idxmax()], index=[f"DP_SEX_DP", f"DP_SEX_val"])
                    dp_third = diff_df[0]
                    group_3 = diff_df[1].split("_")[1]
                    return dp_third, group_3

                dp_diff, group_1 = dem_parity_by_group(
                    y_true=y_true,
                    y_pred=y_pred,
                    sens_attribute_list=sensitive_attribute_1,
                )
                dp_second, group_2 = dem_parity_by_group(
                    y_pred=y_pred,
                    y_true=y_true,
                    sens_attribute_list=sensitive_attribute_2,
                )

                dp_third, group_3 = dem_parity_by_group(
                    y_true=y_true,
                    y_pred=y_pred,
                    sens_attribute_list=sensitive_attribute_3,
                )

                agg_metrics = {
                    f"Test Node {node_name} - First DP NEW.": dp_diff,
                    f"Test Node {node_name} - Second DP NEW.": dp_second,
                    f"Test Node {node_name} - Third DP NEW.": dp_third,
                    f"Test Node {node_name} - Group 1": group_1,
                    f"Test Node {node_name} - Group 2": group_2,
                    f"Test Node {node_name} - Group 3": float(int(group_3)),
                    "FL Round": server_round,
                }
                if wandb_run:
                    wandb_run.log(agg_metrics)

        loss_test = (
            sum(
                [
                    n_examples * metric["test_loss" if not train_parameters.sweep else "validation_loss"]
                    for n_examples, metric in metrics
                ]
            )
            / total_examples
        )
        accuracy_test = (
            sum(
                [
                    n_examples * metric[("test_accuracy" if not train_parameters.sweep else "validation_accuracy")]
                    for n_examples, metric in metrics
                ]
            )
            / total_examples
        )
        f1_test = sum([n_examples * metric["f1_score"] for n_examples, metric in metrics]) / total_examples

        if args.metric == "disparity":
            # Log data from the different test clients:
            for _, metric in metrics:
                node_name = metric["cid"]
                disparity = metric[("max_disparity_test" if not train_parameters.sweep else "max_disparity_validation")]
                disparity_second = metric[
                    ("max_disparity_test_second" if not train_parameters.sweep else "max_disparity_validation_second")
                ]
                disparity_third = metric[
                    ("max_disparity_test_third" if not train_parameters.sweep else "max_disparity_validation_third")
                ]
                accuracy = metric[("test_accuracy" if not train_parameters.sweep else "validation_accuracy")]
                disparity_dataset = metric.get("max_disparity_dataset", 0)
                agg_metrics = {
                    f"Test Node {node_name} - Acc.": accuracy,
                    f"Test Node {node_name} - Disp.": disparity,
                    f"Test Node {node_name} - Second Disp.": disparity_second,
                    f"Test Node {node_name} - Third Disp.": disparity_third,
                    f"Test Node {node_name} - Disp. Dataset": disparity_dataset,
                    "FL Round": server_round,
                }
                if wandb_run:
                    wandb_run.log(agg_metrics)
            (
                sum_counters,
                sum_targets,
                average_probabilities,
                max_disparity_statistics,
                disparity_combinations,
            ) = AggregationFunctions.handle_counters(metrics, "counters", fed_dir)

            (
                _,
                _,
                _,
                max_disparity_statistics_second_value,
                _,
            ) = AggregationFunctions.handle_counters(metrics, "second_counters", fed_dir)

            (
                _,
                _,
                _,
                max_disparity_statistics_third_value,
                _,
            ) = AggregationFunctions.handle_counters(metrics, "third_counters", fed_dir)

            # sex_unfair_income = [15, 17, 9, 5, 13, 7, 18, 8, 19, 6]
            # mar_unfair_income = [14, 3, 0, 4, 11, 16, 12, 2, 10, 1]

            # sex_unfair_employment = [14, 3, 0, 4, 11, 16, 12, 2, 10, 1]
            # mar_unfair_employment = [15, 17, 9, 5, 13, 7, 18, 8, 19, 6]

            # (
            #     _,
            #     _,
            #     _,
            #     max_disparity_statistics_SEX_unfair,
            #     _,
            # ) = AggregationFunctions.handle_counters(
            #     metrics,
            #     "counters",
            #     fed_dir,
            #     unfair_list=sex_unfair_income if train_parameters.dataset_name == "income" else sex_unfair_employment,
            # )

            # (
            #     _,
            #     _,
            #     _,
            #     max_disparity_statistics_MAR_unfair,
            #     _,
            # ) = AggregationFunctions.handle_counters(
            #     metrics,
            #     "second_counters",
            #     fed_dir,
            #     unfair_list=mar_unfair_income if train_parameters.dataset_name == "income" else mar_unfair_employment,
            # )

            if wandb_run:
                for combination in disparity_combinations:
                    target, sensitive_value, disparity = combination
                    wandb_run.log(
                        {
                            "FL Round": server_round,
                            f"Test Disparity P({target}, {sensitive_value}) - P({target}, NOT {sensitive_value})": abs(
                                disparity
                            ),
                        }
                    )

        if args.metric == "disparity":
            agg_metrics = {
                "Test Loss": loss_test,
                "Test Accuracy": accuracy_test,
                "Test Disparity with statistics": max_disparity_statistics,
                "Test Disparity with statistics Second value": max_disparity_statistics_second_value,
                "Test Disparity with statistics Third value": max_disparity_statistics_third_value,
                # "Test Disparity with statistics SEX": max_disparity_statistics_SEX_unfair,
                # "Test Disparity with statistics MAR": max_disparity_statistics_MAR_unfair,
                "FL Round": server_round,
                "Test F1": f1_test,
            }

        if wandb_run:
            wandb_run.log(agg_metrics)
        return agg_metrics

    def agg_metrics_evaluation(
        metrics: list,
        server_round: int,
        train_parameters,
        wandb_run,
        args,
        fed_dir: str,
    ) -> dict:
        total_examples = sum([n_examples for n_examples, _ in metrics])
        loss_evaluation = (
            sum(
                [
                    n_examples * metric["test_loss" if not train_parameters.sweep else "validation_loss"]
                    for n_examples, metric in metrics
                ]
            )
            / total_examples
        )
        accuracy_evaluation = (
            sum(
                [
                    n_examples * metric[("test_accuracy" if not train_parameters.sweep else "validation_accuracy")]
                    for n_examples, metric in metrics
                ]
            )
            / total_examples
        )
        f1_validation = sum([n_examples * metric["f1_score"] for n_examples, metric in metrics]) / total_examples

        if args.metric == "disparity":
            (
                sum_counters,
                sum_targets,
                average_probabilities,
                max_disparity_statistics,
                disparity_combinations,
            ) = AggregationFunctions.handle_counters(metrics, "counters", fed_dir)
            if wandb_run:
                for combination in disparity_combinations:
                    target, sensitive_value, disparity = combination
                    wandb_run.log(
                        {
                            "FL Round": server_round,
                            f"Validation Disparity P({target}, {sensitive_value}) - P({target}, NOT {sensitive_value})": abs(
                                disparity
                            ),
                        }
                    )

        custom_metric = accuracy_evaluation
        if args.target:
            if args.metric == "disparity":
                distance = args.target - max_disparity_statistics

            if distance > 0:
                penalty = 0
            else:
                penalty = -float("inf")

            custom_metric = accuracy_evaluation + penalty

        if args.metric == "disparity":
            agg_metrics = {
                "Validation Loss": loss_evaluation,
                "Validation_Accuracy": accuracy_evaluation,
                "Validation Disparity with statistics": max_disparity_statistics,
                "Custom_metric": custom_metric,
                "FL Round": server_round,
                "Validation F1": f1_validation,
            }

        if wandb_run:
            wandb_run.log(agg_metrics)
        return agg_metrics

    def agg_metrics_train(
        metrics: list,
        server_round: int,
        current_max_epsilon: float,
        fed_dir,
        wandb_run=None,
        args=None,
    ) -> dict:
        losses = []
        losses_with_regularization = []
        epsilon_list = []
        accuracies = []
        lambda_list = []
        max_disparity_train = []

        total_examples = sum([n_examples for n_examples, _ in metrics])
        agg_metrics = {
            "FL Round": server_round,
        }

        # Generic statistics that are logged for each round
        # and that are not dependent on the metric we are using
        for n_examples, node_metrics in metrics:
            losses.append(n_examples * node_metrics["train_loss"])

            losses_with_regularization.append(n_examples * node_metrics["train_loss_with_regularization"])
            epsilon_list.append(node_metrics["epsilon"])
            accuracies.append(n_examples * node_metrics["train_accuracy"])
            lambda_list.append(node_metrics["Lambda"])
            client_id = node_metrics["cid"]
            DPL_lambda = node_metrics["Lambda"]

            if DPL_lambda:
                agg_metrics[f"Lambda Client {client_id}"] = DPL_lambda

            if args.metric == "disparity":
                disparity_client_after_local_epoch = node_metrics["Disparity Train"]
                agg_metrics = {
                    f"Disparity Client {client_id} After Local train": disparity_client_after_local_epoch,
                }

        current_max_epsilon = max(current_max_epsilon, *epsilon_list)
        agg_metrics["Train Loss"] = sum(losses) / total_examples
        agg_metrics["Train Accuracy"] = sum(accuracies) / total_examples
        agg_metrics["Train Loss with Regularization"] = sum(losses_with_regularization) / total_examples
        agg_metrics["Aggregated Lambda"] = (
            sum(lambda_list) / len(lambda_list) if args.regularization_mode == "tunable" else args.regularization_lambda
        )
        agg_metrics["Train Epsilon"] = current_max_epsilon

        if wandb_run:
            wandb_run.log(
                agg_metrics,
            )

        # now we compute some other aggregated metrics on the entire
        # metrics list returned by the clients
        if args.metric == "disparity":
            (
                sum_counters,
                sum_targets,
                average_probabilities,
                max_disparity_statistics,
                _,
            ) = AggregationFunctions.handle_counters(metrics, "counters", fed_dir)
            with open(f"{fed_dir}/avg_proba.pkl", "wb") as file:
                dill.dump(average_probabilities, file)
            (
                sum_counters_no_noise,
                sum_targets_no_noise,
                _,
                max_disparity_statistics_no_noise,
                disparity_combinations_no_noise,
            ) = AggregationFunctions.handle_counters(metrics, "counters_no_noise", fed_dir)
            if wandb_run:
                for combination in disparity_combinations_no_noise:
                    target, sensitive_value, disparity = combination
                    wandb_run.log(
                        {
                            "FL Round": server_round,
                            f"Train Disparity P({target}, {sensitive_value}) - P({target}, NOT {sensitive_value})": abs(
                                disparity
                            ),
                        }
                    )

            wandb_run.log(
                {
                    "Training Disparity with statistics": max_disparity_statistics,
                    "Training Disparity with statistics no noise": max_disparity_statistics_no_noise,
                    "FL Round": server_round,
                    "Average Probabilities": average_probabilities,
                }
            )

        return agg_metrics

    def handle_counters(metrics, key, fed_dir, unfair_list=None):
        # open the metadata file and update the counters
        with open(f"{fed_dir}/metadata.json", "r") as infile:
            json_file = json.load(infile)

        combinations = json_file["combinations"]  # ["1|0", "1|1"]
        all_combinations = json_file["all_combinations"]  # ["0|0", "0|1", "1|0", "1|1"]
        missing_combinations = json_file["missing_combinations"]  # [("0|0", "1|0"), ("0|1", "1|1")]
        # sum_counters = {"0|0": 0, "0|1": 0, "1|0": 0, "1|1": 0}
        sum_counters = {key: 0 for key in all_combinations}
        possible_sensitive_attributes = json_file["possible_z"]
        possible_targets = json_file["possible_y"]

        sum_possible_sensitive_attributes = {key: 0 for key in possible_sensitive_attributes}  # {"0": 0, "1": 0}

        for _, metric in metrics:
            metric_copy = copy.deepcopy(metric)
            metric = metric[key]
            if unfair_list:
                cid = metric_copy["cid"]
                if int(cid) not in unfair_list:
                    continue

            for combination in combinations:
                try:
                    sum_counters[combination] += metric[combination]
                except:
                    continue

            for sensitive_attribute in possible_sensitive_attributes:
                try:
                    sum_possible_sensitive_attributes[sensitive_attribute] += metric[sensitive_attribute]
                except:
                    continue

        for non_existing, existing in missing_combinations:
            sum_counters[non_existing] = (
                sum_possible_sensitive_attributes[existing[-1]] - sum_counters[existing]
                if sum_possible_sensitive_attributes[existing[-1]] - sum_counters[existing] > 0
                else 0
            )
        average_probabilities = {}
        for combination in all_combinations:
            try:
                proba = sum_counters[combination] / sum_possible_sensitive_attributes[combination[2]]
                if proba > 1:
                    proba = 1
                if proba < 0:
                    proba = 0
                average_probabilities[combination] = proba
            except:
                continue

        max_disparity_statistics = []
        combinations_disparity = []
        for target in possible_targets:
            for sensitive_value in possible_sensitive_attributes:
                Y_target_Z_sensitive_value = sum_counters[f"{target}|{sensitive_value}"]
                Z_sensitive_value = sum_possible_sensitive_attributes[sensitive_value]
                Z_not_sensitive_value = 0
                Y_target_Z_not_sensitive_value = 0
                for not_sensitive_value in possible_sensitive_attributes:
                    if not_sensitive_value != sensitive_value:
                        Y_target_Z_not_sensitive_value += sum_counters[f"{target}|{not_sensitive_value}"]
                        Z_not_sensitive_value += sum_possible_sensitive_attributes[not_sensitive_value]

                if Z_sensitive_value == 0 and Z_not_sensitive_value == 0:
                    continue

                if Z_sensitive_value == 0:
                    disparity = abs(Y_target_Z_not_sensitive_value / Z_not_sensitive_value)
                elif Z_not_sensitive_value == 0:
                    disparity = abs(Y_target_Z_sensitive_value / Z_sensitive_value)
                else:
                    disparity = abs(
                        Y_target_Z_sensitive_value / Z_sensitive_value
                        - Y_target_Z_not_sensitive_value / Z_not_sensitive_value
                    )

                max_disparity_statistics.append(disparity)
                combinations_disparity.append((target, sensitive_value))

        max_disparity_with_statistics = max(max_disparity_statistics)
        if max_disparity_with_statistics < 0:
            max_disparity_with_statistics = 0
        if max_disparity_with_statistics > 1:
            max_disparity_with_statistics = 1

        combinations = [
            (target, sv, disparity) for (target, sv), disparity in zip(combinations_disparity, max_disparity_statistics)
        ]

        return (
            sum_counters,
            sum_possible_sensitive_attributes,
            average_probabilities,
            max_disparity_with_statistics,  # max_disparity_statistics,
            combinations,
        )
