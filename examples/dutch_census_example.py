import os
import sys

# Add src to path to import FeDa4Fair
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/FeDa4Fair")))

from FairFederatedDataset import FairFederatedDataset
from fairness_computation import compute_fairness


def main():
    print("Loading Dutch Census dataset...")

    # Path to store the dataset partitions
    output_path = os.path.join(os.path.dirname(__file__), "dutch_data")

    # Initialize FairFederatedDataset with the generic HF dataset
    # "lucacorbucci/Dutch_Census"
    fds = FairFederatedDataset(
        dataset="lucacorbucci/Dutch_Census",
        partitioners={"train": 10},
        label_name="occupation_binary",
        sensitive_attributes=["sex_binary"],
        fairness_metric="DP",
        path=output_path,
    )

    print(f"Dataset initialized. Preparing and saving to {output_path}...")
    fds.prepare()
    print("Dataset prepared and saved.")

    # Access a partition
    client_0 = fds.load_partition(0, split="train")
    print(f"Client 0 (train) size: {len(client_0)}")

    # Evaluate and PRINT fairness metrics (Demographic Parity)
    print("\nComputing Fairness Metrics on Partitions (Data Bias):")

    # We use compute_fairness directly to get the results as a DataFrame for printing
    partitioner = fds.partitioners["train"]

    metrics_df = compute_fairness(
        partitioner=partitioner,
        partitioner_test=partitioner,  # Using same for data bias check
        model=None,
        sens_att="sex_binary",
        fairness_metric="DP",
        label_name=fds.label_column,
        size_unit="attribute",
    )

    print("\nFairness Results (Demographic Parity Difference):")
    print(metrics_df)

    print("\nExample finished successfully.")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
