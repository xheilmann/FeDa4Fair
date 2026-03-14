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


def load_data(file_path):
    dataset = torch.load(file_path)
    # TabularDataset stores data as lists or tensors
    # We need to stack them if they are tensors or convert lists to numpy

    X = dataset.samples
    if isinstance(X, list):
        X = torch.stack(X).numpy() if isinstance(X[0], torch.Tensor) else np.array(X)
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
        model = LogisticRegression(max_iter=1000)
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


def prepare_data_step(args, cross_silo):
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
            splitted_data_dir=args.splitted_data_dir,
            cross_silo=cross_silo,
            sweep=False,
            seed=42,
            validation_seed=42,
        )
        print(f"Data prepared in {fed_dir}")
        return Path(fed_dir)
    except (RuntimeError, ValueError, ImportError) as e:
        print(f"Error preparing data: {e}")
        fed_dir = Path(args.dataset_path) / args.splitted_data_dir
        print(f"Attempting to continue with expected path {fed_dir}")
        return fed_dir


def load_existing_results(output_file):
    if output_file.exists():
        try:
            with output_file.open() as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            print("Warning: Could not decode existing JSON, starting fresh.")
    return {}


def process_client(client_dir, args):
    client_id = client_dir.name
    train_path = client_dir / "train.pt"
    test_path = client_dir / "test.pt"

    if not train_path.exists():
        print(f"  Missing train.pt for client {client_id}, skipping.")
        return None

    try:
        if test_path.exists():
            X_train, y_train, _, _ = load_data(train_path)
            X_test, y_test, z_test, w_test = load_data(test_path)
        else:
            # Split train.pt
            X_all, y_all, z_all, w_all = load_data(train_path)
            min_samples = 5
            if len(y_all) < min_samples:
                print(f"  Client {client_id} has too few samples ({len(y_all)}), skipping.")
                return None

            if w_all is not None:
                X_train, X_test, y_train, y_test, _, z_test, _, w_test = train_test_split(
                    X_all, y_all, z_all, w_all, test_size=0.2, random_state=42
                )
            else:
                X_train, X_test, y_train, y_test, _, z_test = train_test_split(
                    X_all, y_all, z_all, test_size=0.2, random_state=42
                )
                w_test = None

        MIN_CLASSES = 2
        if len(np.unique(y_train)) < MIN_CLASSES:
            print(f"  Client {client_id} has only one class in training data, skipping.")
            return None

        client_results = {}
        # Train Logistic Regression
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
    except (RuntimeError, ValueError, TypeError) as e:
        print(f"  Error processing client {client_id}: {e}")
        return None
    else:
        return client_results


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

    dataset_path = prepare_data_step(args, cross_silo)

    output_dir_path = Path(args.output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    output_file = output_dir_path / f"{args.dataset_type}.json"

    full_results = load_existing_results(output_file)
    current_results = {}

    if not dataset_path.exists():
        print(f"Error: Dataset path {dataset_path} does not exist.")
        return

    # Iterate over client directories (0, 1, 2, ...)
    client_dirs = [d for d in dataset_path.iterdir() if d.is_dir() and d.name.isdigit()]
    client_dirs.sort(key=lambda x: int(x.name))

    if not client_dirs:
        print(f"No client directories found in {dataset_path}")
        return

    for client_dir in client_dirs:
        res = process_client(client_dir, args)
        if res:
            current_results[client_dir.name] = res

    # Update full results
    if args.scenario:
        full_results[args.scenario] = current_results
    else:
        full_results.update(current_results)

    # Save results
    with output_file.open("w") as f:
        json.dump(full_results, f, indent=4)

    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
