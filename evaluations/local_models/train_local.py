import argparse
import json
import os
import sys
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from pathlib import Path

# Add puffle to python path to import TabularDataset
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../puffle")))
try:
    from Utils.dutch import TabularDataset
    from Utils.tabular_data_loader import prepare_tabular_data
except ImportError:
    # Fallback if the path structure is different, try adding ../../puffle
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../puffle")))
    from Utils.dutch import TabularDataset
    from Utils.tabular_data_loader import prepare_tabular_data


def load_data(file_path):
    dataset = torch.load(file_path)
    # TabularDataset stores data as lists or tensors
    # We need to stack them if they are tensors or convert lists to numpy

    X = dataset.samples
    if isinstance(X, list):
        if isinstance(X[0], torch.Tensor):
            X = torch.stack(X).numpy()
        else:
            X = np.array(X)
    elif isinstance(X, torch.Tensor):
        X = X.numpy()

    y = dataset.targets
    if isinstance(y, list):
        y = np.array(y)
    elif isinstance(y, torch.Tensor):
        y = y.numpy()

    z = dataset.sensitive_features
    if isinstance(z, list):
        z = np.array(z)
    elif isinstance(z, torch.Tensor):
        z = z.numpy()

    w = getattr(dataset, "sensitive_features_2", None)
    if w is not None:
        if isinstance(w, list):
            w = np.array(w)
        elif isinstance(w, torch.Tensor):
            w = w.numpy()

    return X, y, z, w


def calculate_fairness_metrics(y_true, y_pred, z):
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    z = np.array(z)

    unique_z = np.unique(z)

    # Demographic Disparity (Max absolute difference in positive prediction rates)
    # P(Y_pred=1 | Z=z)
    positive_rates = {}
    for g in unique_z:
        mask = z == g
        if np.sum(mask) > 0:
            positive_rates[g] = np.mean(y_pred[mask])
        else:
            positive_rates[g] = 0.0  # Or nan

    rates = list(positive_rates.values())
    if len(rates) > 1:
        demographic_disparity = max(rates) - min(rates)
    else:
        demographic_disparity = 0.0

    # Equalized Odds (Max absolute difference in TPR and FPR between groups)
    # TPR: P(Y_pred=1 | Y_true=1, Z=z)
    # FPR: P(Y_pred=1 | Y_true=0, Z=z)
    tpr_by_group = {}
    fpr_by_group = {}

    for g in unique_z:
        mask = z == g

        # TPR
        mask_pos = mask & (y_true == 1)
        if np.sum(mask_pos) > 0:
            tpr_by_group[g] = np.mean(y_pred[mask_pos])
        else:
            tpr_by_group[g] = 0.0

        # FPR
        mask_neg = mask & (y_true == 0)
        if np.sum(mask_neg) > 0:
            fpr_by_group[g] = np.mean(y_pred[mask_neg])
        else:
            fpr_by_group[g] = 0.0

    tprs = list(tpr_by_group.values())
    fprs = list(fpr_by_group.values())

    if len(tprs) > 1:
        tpr_diff = max(tprs) - min(tprs)
        fpr_diff = max(fprs) - min(fprs)
        equalized_odds = max(tpr_diff, fpr_diff)
    else:
        equalized_odds = 0.0

    return {
        "demographic_disparity": float(demographic_disparity),
        "equalized_odds": float(equalized_odds),
        "positive_rates": {str(k): float(v) for k, v in positive_rates.items()},
        "tpr_by_group": {str(k): float(v) for k, v in tpr_by_group.items()},
        "fpr_by_group": {str(k): float(v) for k, v in fpr_by_group.items()},
    }


def train_and_eval(
    X_train, y_train, X_test, y_test, z_test, w_test, model_type, sensitive_feature, second_sensitive_feature
):
    if model_type == "lr":
        model = LogisticRegression(max_iter=1000)
    elif model_type == "xgb":
        model = XGBClassifier(eval_metric="logloss", use_label_encoder=False)
    else:
        raise ValueError("Unknown model type")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")

    fairness_z = calculate_fairness_metrics(y_test, y_pred, z_test)

    metrics = {
        "accuracy": float(acc),
        "f1_score": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        f"{sensitive_feature}_fairness": fairness_z,
    }

    if w_test is not None and second_sensitive_feature:
        fairness_w = calculate_fairness_metrics(y_test, y_pred, w_test)
        metrics[f"{second_sensitive_feature}_fairness"] = fairness_w

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train local models")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the directory containing data")
    parser.add_argument(
        "--dataset_type",
        type=str,
        required=True,
        help="Identifier for the dataset type (e.g. dutch_cross_silo_attribute)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Scenario name (e.g. medium, mild, strong). If provided, results are nested under this key.",
    )
    parser.add_argument(
        "--dataset_name", type=str, default="dutch_prepared", help="Dataset name for prepare_tabular_data"
    )
    parser.add_argument("--num_nodes", type=int, default=50, help="Number of nodes/clients")
    parser.add_argument(
        "--cross_silo", type=str, default="True", help="Cross silo flag"
    )  # Changed type to str to handle shell input easier
    parser.add_argument("--splitted_data_dir", type=str, default="federated", help="Directory for splitted data")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument(
        "--sensitive_feature", type=str, default="sex_binary", help="Name of the first sensitive feature"
    )
    parser.add_argument(
        "--second_sensitive_feature", type=str, default="Marital_status", help="Name of the second sensitive feature"
    )
    parser.add_argument("--target", type=str, default="occupation_binary", help="Name of the target variable")

    args = parser.parse_args()

    # Handle boolean string
    cross_silo = args.cross_silo.lower() == "true"

    print(f"Preparing data for {args.dataset_type} {f'({args.scenario})' if args.scenario else ''}...")
    try:
        # We call prepare_tabular_data to ensure .pt files exist
        fed_dir, _ = prepare_tabular_data(
            dataset_path=args.dataset_path,
            dataset_name=args.dataset_name,
            approach="egalitarian",  # dummy
            num_nodes=args.num_nodes,
            ratio_unfair_nodes=0,  # dummy
            opposite_direction=False,  # dummy
            ratio_unfairness=(0, 0),  # dummy
            do_iid_split=False,
            splitted_data_dir=args.splitted_data_dir,
            cross_silo=cross_silo,
            sweep=False,
            seed=42,
            validation_seed=42,
        )
        print(f"Data prepared in {fed_dir}")
    except Exception as e:
        print(f"Error preparing data: {e}")
        fed_dir = os.path.join(args.dataset_path, args.splitted_data_dir)
        print(f"Attempting to continue with expected path {fed_dir}")

    dataset_path = Path(fed_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = Path(args.output_dir) / f"{args.dataset_type}.json"

    # Load existing results if they exist
    full_results = {}
    if output_file.exists():
        try:
            with open(output_file, "r") as f:
                full_results = json.load(f)
        except json.JSONDecodeError:
            print("Warning: Could not decode existing JSON, starting fresh.")
            full_results = {}

    current_results = {}

    if not dataset_path.exists():
        print(f"Error: Dataset path {dataset_path} does not exist.")
        return

    # Iterate over client directories (0, 1, 2, ...)
    client_dirs = [d for d in dataset_path.iterdir() if d.is_dir() and d.name.isdigit()]
    client_dirs.sort(key=lambda x: int(x.name))

    if not client_dirs:
        print(f"No client directories found in {dataset_path}")
        # Don't return yet, maybe we want to save empty? But better to skip.
        return

    for client_dir in client_dirs:
        client_id = client_dir.name
        # print(f"Processing client {client_id}...")

        train_path = client_dir / "train.pt"
        test_path = client_dir / "test.pt"

        if not train_path.exists():
            print(f"  Missing train.pt for client {client_id}, skipping.")
            continue

        try:
            if test_path.exists():
                X_train, y_train, z_train, w_train = load_data(train_path)
                X_test, y_test, z_test, w_test = load_data(test_path)
            else:
                # Split train.pt
                X_all, y_all, z_all, w_all = load_data(train_path)
                if len(y_all) < 5:
                    print(f"  Client {client_id} has too few samples ({len(y_all)}), skipping.")
                    continue

                if w_all is not None:
                    X_train, X_test, y_train, y_test, z_train, z_test, w_train, w_test = train_test_split(
                        X_all, y_all, z_all, w_all, test_size=0.2, random_state=42
                    )
                else:
                    X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(
                        X_all, y_all, z_all, test_size=0.2, random_state=42
                    )
                    w_train, w_test = None, None

            if len(np.unique(y_train)) < 2:
                print(f"  Client {client_id} has only one class in training data, skipping.")
                continue

            client_results = {}

            # Train Logistic Regression
            # print(f"  Training LR...")
            lr_metrics = train_and_eval(
                X_train,
                y_train,
                X_test,
                y_test,
                z_test,
                w_test,
                "lr",
                args.sensitive_feature,
                args.second_sensitive_feature,
            )
            client_results["lr"] = lr_metrics

            # Train XGBoost
            # print(f"  Training XGB...")
            xgb_metrics = train_and_eval(
                X_train,
                y_train,
                X_test,
                y_test,
                z_test,
                w_test,
                "xgb",
                args.sensitive_feature,
                args.second_sensitive_feature,
            )
            client_results["xgb"] = xgb_metrics

            current_results[client_id] = client_results

        except Exception as e:
            print(f"  Error processing client {client_id}: {e}")
            # import traceback
            # traceback.print_exc()

    # Update full results
    if args.scenario:
        full_results[args.scenario] = current_results
    else:
        full_results.update(current_results)  # If no scenario, assuming flat structure or overwriting

    # Save results
    with open(output_file, "w") as f:
        json.dump(full_results, f, indent=4)

    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
