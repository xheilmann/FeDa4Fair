from sklearn.linear_model import LogisticRegression

from FeDa4Fair.dataset import FairFederatedDataset
from FeDa4Fair.metrics.fairness import compute_multi_fairness
from FeDa4Fair.utils.data_utils import generate_multiobjective_bias

CROSS_SILO_CLIENT_COUNT = 50


def test_bias_level(drop_mean, flip_mean, num_clients, scenario_type):
    # In Dutch Census:
    # sex_binary=0 (Male) mean label ~0.67
    # sex_binary=1 (Female) mean label ~0.37
    # Baseline DP ~ 0.30

    # To REDUCE DP (Mild 0.15), we should drop samples from the PRIVILEGED group (0)
    # or mitigate.
    # To INCREASE DP (Strong 0.35), we should drop samples from the UNPRIVILEGED group (1).

    print(f"Testing {scenario_type}: drop={drop_mean}, flip={flip_mean}")

    # If we want 0.15 (Mild), we need to MITIGATE first then maybe add some bias back?
    # Or just use low drop rates on the unprivileged group if the baseline is lower.
    # Wait, the baseline DP is already 0.30.
    # So "Mild" (0.15) actually means REDUCING the natural bias.
    # "Strong" (0.35) means INCREASING the natural bias.

    # For simplicity, let's assume the user wants average unfairness of 0.15, 0.25, 0.35.
    # If the natural unfairness is 0.30, we need mitigation for 0.15 and 0.25.

    group_configs = [
        {
            "group_id": "G",
            "num_clients": num_clients,
            "configs": [
                {
                    "attribute": "sex_binary",
                    "value": 1,  # Targeting females
                    "drop_mean": drop_mean,
                    "drop_std": 0.05,
                    "flip_mean": flip_mean,
                    "flip_std": 0.02,
                    "mitigate": drop_mean < 0,  # Dummy way to trigger mitigation in this test script
                }
            ],
        }
    ]

    modifications = generate_multiobjective_bias(num_clients, group_configs)

    fds = FairFederatedDataset(
        dataset="lucacorbucci/Dutch_census_binary_marital_status",
        split="all",
        partitioners={"train": num_clients},
        label_name="occupation_binary",
        sensitive_attributes=["sex_binary"],
        modification_dict=modifications,
        fl_setting="cross-silo" if num_clients == CROSS_SILO_CLIENT_COUNT else "cross-device",
        perc_train_val_test=[0.8, 0.2],
    )

    fds.prepare()

    results = compute_multi_fairness(
        partitioner=fds.partitioners["train_train"],
        partitioner_test=fds.partitioners["train_test"],
        model=LogisticRegression(max_iter=1000, solver="liblinear"),
        sens_atts=["sex_binary"],
        fairness_metric="DP",
        label_name="occupation_binary",
        fds=fds,
        split="train_train",
        test_split="train_test",
    )

    avg_dp = results["sex_binary_DP"].mean()
    print(f"Average DP: {avg_dp:.4f}\n")
    return avg_dp


if __name__ == "__main__":
    # Baseline DP is ~0.30

    print("--- Increasing Bias (Target > 0.30) ---")
    test_bias_level(0.3, 0.1, CROSS_SILO_CLIENT_COUNT, "attribute")  # Target 0.35?
    test_bias_level(0.6, 0.2, CROSS_SILO_CLIENT_COUNT, "attribute")

    print("--- Reducing Bias (Target < 0.30) ---")
    # To reduce bias, we use 'mitigate': True in the create scripts.
    # But let's see what happens if we drop from the privileged group (0)
    # test_bias_level with value=0
