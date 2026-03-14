import json
from itertools import product
from pathlib import Path

import dill
import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame, selection_rate


class AggregationFunctions:
    @staticmethod
    def _compute_dem_parity_by_group(sens_attribute_list, y_true, y_pred):
        sf_data = pd.DataFrame({"DP_RACE": sens_attribute_list})
        sel_rate = MetricFrame(
            metrics={"sel": selection_rate},
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sf_data,
        )
        df = sel_rate.by_group
        sel_values = df["sel"].to_numpy()
        diff_matrix = sel_values[:, None] - sel_values[None, :]
        index = df.index.to_numpy()
        column_names = [f"{index[i]}_{index[j]}" for i, j in product(range(len(df)), repeat=2)]

        diff_series = pd.Series(diff_matrix.flatten(), index=column_names)
        max_diff = diff_series.max()
        max_group_pair = diff_series.idxmax()
        group_3 = max_group_pair.split("_")[1]
        return max_diff, group_3

    def agg_metrics_test(
        self: list,
        server_round: int,
        train_parameters,
        wandb_run,
        args,
        fed_dir: str,
    ) -> dict:
        total_examples = sum([n_examples for n_examples, _ in self])

        for _, metric in self:
            if "y_true" in metric:
                node_name = metric["cid"]
                y_true = np.array([int(item) for item in metric["y_true"]])
                y_pred = np.array(metric["y_pred"])
                s1 = np.array(list(metric["sensitive_attributes_1"]))
                s2 = np.array(list(metric["sensitive_attributes_2"]))
                s3 = np.array(list(metric["sensitive_attributes_3"]))

                dp_diff, group_1 = AggregationFunctions._compute_dem_parity_by_group(s1, y_true, y_pred)
                dp_second, group_2 = AggregationFunctions._compute_dem_parity_by_group(s2, y_true, y_pred)
                dp_third, group_3 = AggregationFunctions._compute_dem_parity_by_group(s3, y_true, y_pred)

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

        loss_key = "test_loss" if not train_parameters.sweep else "validation_loss"
        acc_key = "test_accuracy" if not train_parameters.sweep else "validation_accuracy"
        disp_key = "max_disparity_test" if not train_parameters.sweep else "max_disparity_validation"
        disp_key_2 = "max_disparity_test_second" if not train_parameters.sweep else "max_disparity_validation_second"
        disp_key_3 = "max_disparity_test_third" if not train_parameters.sweep else "max_disparity_validation_third"

        loss_test = sum([n * m[loss_key] for n, m in self]) / total_examples
        accuracy_test = sum([n * m[acc_key] for n, m in self]) / total_examples
        f1_test = sum([n * m["f1_score"] for n, m in self]) / total_examples

        if args.metric == "disparity":
            # Log data from the different test clients:
            for _, metric in self:
                node_name = metric["cid"]
                agg_metrics = {
                    f"Test Node {node_name} - Acc.": metric[acc_key],
                    f"Test Node {node_name} - Disp.": metric[disp_key],
                    f"Test Node {node_name} - Second Disp.": metric[disp_key_2],
                    f"Test Node {node_name} - Third Disp.": metric[disp_key_3],
                    f"Test Node {node_name} - Disp. Dataset": metric.get("max_disparity_dataset", 0),
                    "FL Round": server_round,
                }

                max_group = metric.get("max_group_test" if not train_parameters.sweep else "max_group_validation")
                if max_group:
                    agg_metrics[f"Max Group Client {node_name}"] = {
                        "client_id": node_name,
                        "max_y": max_group[0],
                        "max_z": max_group[1],
                    }

                if wandb_run:
                    wandb_run.log(agg_metrics)

            (_, _, _, max_disparity_statistics, combinations) = AggregationFunctions.handle_counters(self, "counters", fed_dir)
            (_, _, _, max_disparity_statistics_second_value, _) = AggregationFunctions.handle_counters(self, "second_counters", fed_dir)
            (_, _, _, max_disparity_statistics_third_value, _) = AggregationFunctions.handle_counters(self, "third_counters", fed_dir)

            if wandb_run:
                for target, sv, disparity in combinations:
                    wandb_run.log({
                        "FL Round": server_round,
                        f"Test Disparity P({target}, {sv}) - P({target}, NOT {sv})": abs(disparity),
                    })

        if args.metric == "disparity":
            agg_metrics = {
                "Test Loss": loss_test,
                "Test Accuracy": accuracy_test,
                "Test Disparity with statistics": max_disparity_statistics,
                "Test Disparity with statistics Second value": max_disparity_statistics_second_value,
                "Test Disparity with statistics Third value": max_disparity_statistics_third_value,
                "FL Round": server_round,
                "Test F1": f1_test,
            }

        if wandb_run:
            wandb_run.log(agg_metrics)
        return agg_metrics

    def agg_metrics_evaluation(
        self: list,
        server_round: int,
        train_parameters,
        wandb_run,
        args,
        fed_dir: str,
    ) -> dict:
        total_examples = sum([n_examples for n_examples, _ in self])
        loss_key = "test_loss" if not train_parameters.sweep else "validation_loss"
        acc_key = "test_accuracy" if not train_parameters.sweep else "validation_accuracy"

        loss_evaluation = sum([n * m[loss_key] for n, m in self]) / total_examples
        accuracy_evaluation = sum([n * m[acc_key] for n, m in self]) / total_examples
        f1_validation = sum([n * m["f1_score"] for n, m in self]) / total_examples

        if args.metric == "disparity":
            (_, _, _, max_disparity_statistics, combinations) = AggregationFunctions.handle_counters(self, "counters", fed_dir)
            if wandb_run:
                for target, sv, disparity in combinations:
                    wandb_run.log({
                        "FL Round": server_round,
                        f"Validation Disparity P({target}, {sv}) - P({target}, NOT {sv})": abs(disparity),
                    })

        custom_metric = accuracy_evaluation
        if args.target and args.metric == "disparity":
            distance = args.target - max_disparity_statistics
            penalty = 0 if distance > 0 else -float("inf")
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
        self: list,
        server_round: int,
        current_max_epsilon: float,
        fed_dir,
        wandb_run=None,
        args=None,
    ) -> dict:
        losses, losses_with_reg, epsilons, accuracies, lambdas = [], [], [], [], []
        total_examples = sum([n_examples for n_examples, _ in self])
        agg_metrics = {"FL Round": server_round}

        for n_examples, node_metrics in self:
            losses.append(n_examples * node_metrics["train_loss"])
            losses_with_reg.append(n_examples * node_metrics["train_loss_with_regularization"])
            epsilons.append(node_metrics["epsilon"])
            accuracies.append(n_examples * node_metrics["train_accuracy"])
            lambdas.append(node_metrics["Lambda"])
            cid, dpl_lambda = node_metrics["cid"], node_metrics["Lambda"]

            if dpl_lambda:
                agg_metrics[f"Lambda Client {cid}"] = dpl_lambda

            if args.metric == "disparity":
                agg_metrics[f"Disparity Client {cid} After Local train"] = node_metrics["Disparity Train"]

        current_max_epsilon = max(current_max_epsilon, *epsilons)
        agg_metrics.update({
            "Train Loss": sum(losses) / total_examples,
            "Train Accuracy": sum(accuracies) / total_examples,
            "Train Loss with Regularization": sum(losses_with_reg) / total_examples,
            "Aggregated Lambda": sum(lambdas) / len(lambdas) if args.regularization_mode == "tunable" else args.regularization_lambda,
            "Train Epsilon": current_max_epsilon,
        })

        if wandb_run:
            wandb_run.log(agg_metrics)

        if args.metric == "disparity":
            (_, _, avg_proba, max_disp, _) = AggregationFunctions.handle_counters(self, "counters", fed_dir)
            with Path(f"{fed_dir}/avg_proba.pkl").open("wb") as file:
                dill.dump(avg_proba, file)
            (_, _, _, max_disp_no_noise, combinations_no_noise) = AggregationFunctions.handle_counters(self, "counters_no_noise", fed_dir)

            if wandb_run:
                for target, sv, disparity in combinations_no_noise:
                    wandb_run.log({
                        "FL Round": server_round,
                        f"Train Disparity P({target}, {sv}) - P({target}, NOT {sv})": abs(disparity),
                    })
                wandb_run.log({
                    "Training Disparity with statistics": max_disp,
                    "Training Disparity with statistics no noise": max_disp_no_noise,
                    "FL Round": server_round,
                    "Average Probabilities": avg_proba,
                })
        return agg_metrics

    @staticmethod
    def handle_counters(metrics_list, key, fed_dir, unfair_list=None):
        with Path(f"{fed_dir}/metadata.json").open() as infile:
            meta = json.load(infile)

        all_combs, combinations, missing_combs = meta["all_combinations"], meta["combinations"], meta["missing_combinations"]
        possible_z, possible_y = meta["possible_z"], meta["possible_y"]

        sum_counters = dict.fromkeys(all_combs, 0)
        sum_sensitive = dict.fromkeys(possible_z, 0)

        for _, metric_entry in metrics_list:
            m_data = metric_entry[key]
            if unfair_list and int(metric_entry["cid"]) not in unfair_list:
                continue

            for comb in combinations:
                sum_counters[comb] += m_data.get(comb, 0)
            for sz in possible_z:
                sum_sensitive[sz] += m_data.get(sz, 0)

        for non_existing, existing in missing_combs:
            sum_counters[non_existing] = max(0, sum_sensitive[existing[-1]] - sum_counters[existing])

        avg_proba = {}
        for comb in all_combs:
            if sum_sensitive[comb[2]] > 0:
                avg_proba[comb] = clamp(sum_counters[comb] / sum_sensitive[comb[2]], 0, 1)

        max_disp_stats, disp_combs = [], []
        for target in possible_y:
            for sz in possible_z:
                sz_total = sum_sensitive[sz]
                y_sz = sum_counters[f"{target}|{sz}"]
                y_not_sz, not_sz_total = 0, 0
                for not_sz in possible_z:
                    if not_sz != sz:
                        y_not_sz += sum_counters[f"{target}|{not_sz}"]
                        not_sz_total += sum_sensitive[not_sz]

                if sz_total == 0 and not_sz_total == 0:
                    continue
                d = abs((y_sz / sz_total if sz_total > 0 else 0) - (y_not_sz / not_sz_total if not_sz_total > 0 else 0))
                max_disp_stats.append(d)
                disp_combs.append((target, sz))

        max_disp_final = clamp(max(max_disp_stats), 0, 1) if max_disp_stats else 0
        final_combs = [(t, s, d) for (t, s), d in zip(disp_combs, max_disp_stats, strict=False)]
        return sum_counters, sum_sensitive, avg_proba, max_disp_final, final_combs


def clamp(v, low, high):
    return max(low, min(v, high))
