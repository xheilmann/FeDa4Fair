import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Add puffle to python path to import TabularDataset
sys.path.append(str(Path(__file__).resolve().parent / "../puffle"))
try:
    from Utils.tabular_data_loader import prepare_tabular_data
except ImportError:
    # Fallback if the path structure is different, try adding ../../puffle
    sys.path.append(str(Path(__file__).resolve().parent / "../../puffle"))
    from Utils.tabular_data_loader import prepare_tabular_data

RANDOM_SEED = 42
TEST_SIZE = 0.2
MIN_SAMPLES = 5
MAX_ITER = 1000


def load_data(file_path):
    dataset = torch.load(file_path)
    # TabularDataset stores data as lists or tensors
    # We need to stack them if they are tensors or convert lists to numpy

    x = dataset.samples
    if isinstance(x, list):
        x = torch.stack(x).numpy() if isinstance(x[0], torch.Tensor) else np.array(x)
    elif isinstance(x, torch.Tensor):
        x = x.numpy()

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

    return x, y, z, w


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
    demographic_disparity = max(rates) - min(rates) if len(rates) > 1 else 0.0

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
    x_train, y_train, x_test, y_test, z_test, w_test, model_type, sensitive_feature, second_sensitive_feature
):
    if model_type == "lr":
        model = LogisticRegression(max_iter=MAX_ITER)
    elif model_type == "xgb":
        model = XGBClassifier(eval_metric="logloss", use_label_encoder=False)
    else:
        msg = "Unknown model type"
        raise ValueError(msg)

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

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
        "--dataset_type", type=str, required=True, help="Type of dataset (e.g., dutch_cross_device_value)"
    )
    parser.add_argument("--splitted_data_dir", type=str, default="federated", help="Directory for splitted data")
    parser.add_argument("--num_nodes", type=int, default=150, help="Number of clients")
    parser.add_argument("--cross_silo", type=str, default="False", help="Whether to use cross-silo")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--dataset_name", type=str, default="", help="Name of the dataset (e.g., dutch)")
    parser.add_argument("--scenario", type=str, default="", help="Scenario name")
    parser.add_argument("--sensitive_attribute", type=str, default="", help="Sensitive attribute")
    parser.add_argument("--second_sensitive_attribute", type=str, default="", help="Second sensitive attribute")

    args = parser.parse_args()

    # Handle boolean string
    cross_silo = args.cross_silo.lower() == "true"

    print(f"Preparing data for {args.dataset_type} {f'({args.scenario})' if args.scenario else ''}...")
    try:
        # We call prepare_tabular_data to ensure .pt files exist
        fed_dir = prepare_tabular_data(
            dataset_path=args.dataset_path,
            num_nodes=args.num_nodes,
            ratio_unfair_nodes=0,  # dummy
            opposite_direction=False,  # dummy
            ratio_unfairness=(0, 0),  # dummy
            do_iid_split=False,
            splitted_data_dir=args.splitted_data_dir,
            cross_silo=cross_silo,
            sweep=False,
            seed=RANDOM_SEED,
            validation_seed=RANDOM_SEED,
        )
        print(f"Data prepared in {fed_dir}")
    except (ValueError, OSError, RuntimeError) as e:
        print(f"Error preparing data: {e}")
        fed_dir = Path(args.dataset_path) / args.splitted_data_dir
        print(f"Attempting to continue with expected path {fed_dir}")

    dataset_path = Path(fed_dir)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    output_file = Path(args.output_dir) / f"{args.dataset_type}.json"

    # Load existing results if they exist
    full_results = {}
    if output_file.exists():
        try:
            with output_file.open() as f:
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

    for client_dir in client_dirs:
        client_id = client_dir.name

        train_path = client_dir / "train.pt"
        test_path = client_dir / "test.pt"

        if not train_path.exists():
            continue

        try:
            if test_path.exists():
                x_train, y_train, z_train, w_train = load_data(train_path)
                x_test, y_test, z_test, w_test = load_data(test_path)
            else:
                # Split train.pt
                x_all, y_all, z_all, w_all = load_data(train_path)
                if len(y_all) < MIN_SAMPLES:
                    print(f"  Client {client_id} has too few samples ({len(y_all)}), skipping.")
                    continue

                if w_all is not None:
                    x_train, x_test, y_train, y_test, z_train, z_test, w_train, w_test = train_test_split(
                        x_all, y_all, z_all, w_all, test_size=TEST_SIZE, random_state=RANDOM_SEED
                    )
                else:
                    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(
                        x_all, y_all, z_all, test_size=TEST_SIZE, random_state=RANDOM_SEED
                    )
                    w_train, w_test = None, None

            # Train Logistic Regression
            lr_metrics = train_and_eval(
                x_train,
                y_train,
                x_test,
                y_test,
                z_test,
                w_test,
                "lr",
                args.sensitive_attribute,
                args.second_sensitive_attribute,
            )

            # Train XGBoost
            xgb_metrics = train_and_eval(
                x_train,
                y_train,
                x_test,
                y_test,
                z_test,
                w_test,
                "xgb",
                args.sensitive_attribute,
                args.second_sensitive_attribute,
            )

            current_results[client_id] = {"lr": lr_metrics, "xgb": xgb_metrics, "num_samples": len(y_train) + len(y_test)}

        except (ValueError, OSError, RuntimeError) as e:
            print(f"  Error processing client {client_id}: {e}")
            continue

    # Update full results
    full_results[args.scenario if args.scenario else "default"] = current_results

    # Save results
    with output_file.open("w") as f:
        json.dump(full_results, f, indent=4)

    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
