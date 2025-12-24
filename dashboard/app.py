import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner

from FeDa4Fair.dataset.fair_dataset import FairFederatedDataset
from FeDa4Fair.metrics.fairness import compute_fairness
from FeDa4Fair.utils.data_utils import generate_bias_by_groups

st.set_page_config(page_title="FeDa4Fair Dashboard", layout="wide")

st.title("FeDa4Fair: Fairness in Federated Learning Dashboard")

US_STATES = [
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
    "PR",
]

st.sidebar.header("Dataset Configuration")

dataset_name = st.sidebar.selectbox("Select Dataset", ["ACSIncome", "ACSEmployment", "lucacorbucci/Dutch_Census", "Other (Hugging Face)"])

selected_states = None
if dataset_name in ["ACSIncome", "ACSEmployment"]:
    label_name = None  # Inferred
    sensitive_attributes = None  # Default
    year = st.sidebar.selectbox("Year", ["2014", "2015", "2016", "2017", "2018"], index=4)
    horizon = st.sidebar.selectbox("Horizon", ["1-Year", "5-Year"], index=0)

    select_all = st.sidebar.checkbox("Select All States")
    default_states = US_STATES if select_all else ["CA"]
    selected_states = st.sidebar.multiselect("Select States to Load", US_STATES, default=default_states)
elif dataset_name == "Other (Hugging Face)":
    dataset_name = st.sidebar.text_input("HF Dataset Name", "adult")
    subset = st.sidebar.text_input("Subset (Optional)", None)
    split = st.sidebar.text_input("Split", "train")
    label_name = st.sidebar.text_input("Label Column", "income")
    sens_attr = st.sidebar.text_input("Sensitive Attribute", "sex")
    sensitive_attributes = [sens_attr]
    year = horizon = None
    selected_states = None
else:
    label_name = "occupation_binary"
    sensitive_attributes = ["sex_binary"]
    year = "2018"  # Dummy
    horizon = "1-Year"  # Dummy
    selected_states = None

seed = st.sidebar.number_input("Random Seed", value=42)
shuffle = st.sidebar.checkbox("Shuffle Data?", value=True)
sample_cap = st.sidebar.number_input("Sample Cap per Client (Optional)", min_value=0, value=0, help="0 means no cap. Caps total samples per client maintaining distribution.")

st.sidebar.header("Partitioning")
num_partitions = st.sidebar.slider("Number of Clients (per State/Split)", min_value=1, max_value=50, value=5)
partition_strategy = st.sidebar.selectbox("Partition Strategy", ["IID", "Dirichlet (Non-IID)"])

alpha = 1.0
if partition_strategy == "Dirichlet (Non-IID)":
    alpha = st.sidebar.slider("Alpha (Concentration)", 0.1, 10.0, 1.0)


def create_partitioner():
    if partition_strategy == "Dirichlet (Non-IID)":
        return DirichletPartitioner(
            num_partitions=num_partitions, partition_by=label_name if label_name else "PINCP", alpha=alpha, seed=seed
        )
    return IidPartitioner(num_partitions=num_partitions)


st.sidebar.header("Data Modification (Bias Injection)")
inject_bias = st.sidebar.checkbox("Inject Bias?")
modification_dict = None

if inject_bias:
    st.sidebar.subheader("Group-Based Bias Injection")
    
    mitigation_threshold = st.sidebar.slider("Mitigation Threshold", 0.0, 0.2, 0.08, 0.01, help="Target unfairness threshold for mitigation.")

    if "bias_groups" not in st.session_state:
        st.session_state.bias_groups = [
            {
                "group_id": "Group A",
                "num_clients": num_partitions,
                "sensitive_attr": "SEX" if "ACS" in dataset_name else "sex_binary",
                "sensitive_value": 1,
                "drop_mean": 0.2,
                "drop_std": 0.05,
                "flip_mean": 0.1,
                "flip_std": 0.02,
            }
        ]

    # UI to Add/Remove Groups
    col_add, col_rem = st.sidebar.columns(2)
    if col_add.button("➕ Add Group"):
        st.session_state.bias_groups.append(
            {
                "group_id": f"Group {len(st.session_state.bias_groups) + 1}",
                "num_clients": 0,
                "sensitive_attr": "SEX" if "ACS" in dataset_name else "sex_binary",
                "sensitive_value": 1,
                "drop_mean": 0.0,
                "drop_std": 0.0,
                "flip_mean": 0.0,
                "flip_std": 0.0,
            }
        )

    if col_rem.button("➖ Remove Group") and len(st.session_state.bias_groups) > 1:
        st.session_state.bias_groups.pop()

    # Render Group Forms
    group_configs = []
    for i, group in enumerate(st.session_state.bias_groups):
        with st.sidebar.expander(f"⚙️ {group['group_id']}", expanded=(i == 0)):
            g_id = st.text_input("Group Name", group["group_id"], key=f"id_{i}")
            n_c = st.number_input("Clients in Group", 0, 1000, group["num_clients"], key=f"nc_{i}")

            s_attr = st.text_input("Sensitive Attr", group["sensitive_attr"], key=f"sa_{i}")
            s_val = st.number_input("Underrepresented group", value=group["sensitive_value"], key=f"sv_{i}")

            i_attr = st.text_input("Intersectional Attr (Optional)", key=f"ia_{i}")
            i_val = None
            if i_attr and i_attr.strip() != "" and i_attr != "None":
                i_val = st.number_input("Intersectional Value", value=0, key=f"iv_{i}")

            mitigate = st.checkbox("Mitigate Existing Bias (Balance Groups)?", value=False, key=f"mit_{i}")

            if not mitigate:
                st.markdown("**Sampling Distribution (Truncated Normal)**")
                c1, c2 = st.columns(2)
                d_m = c1.number_input("Drop Mean", 0.0, 1.0, group["drop_mean"], key=f"dm_{i}")
                d_s = c2.number_input("Drop Std", 0.0, 1.0, group["drop_std"], key=f"ds_{i}")

                f_m = c1.number_input("Flip Mean", 0.0, 1.0, group["flip_mean"], key=f"fm_{i}")
                f_s = c2.number_input("Flip Std", 0.0, 1.0, group["flip_std"], key=f"fs_{i}")
            else:
                d_m, d_s, f_m, f_s = 0.0, 0.0, 0.0, 0.0

            group_configs.append(
                {
                    "group_id": g_id,
                    "num_clients": n_c,
                    "sensitive_attr": s_attr,
                    "sensitive_value": s_val,
                    "intersectional_attr": i_attr if i_attr and i_attr.strip() != "" and i_attr != "None" else None,
                    "intersectional_value": i_val,
                    "drop_mean": d_m,
                    "drop_std": d_s,
                    "flip_mean": f_m,
                    "flip_std": f_s,
                    "mitigate": mitigate,
                }
            )

    # Validate Sum
    # Correct calculation: Total clients = States * Partitions_per_state (if ACS) or just partitions
    expected_total = len(selected_states) * num_partitions if selected_states else num_partitions
    total_assigned = sum(g["num_clients"] for g in group_configs)

    if total_assigned != expected_total:
        st.sidebar.error(f"Validation Failed: {total_assigned}/{expected_total} clients assigned.")
    else:
        st.sidebar.success("✅ Client allocation valid.")
        # Generate the modification_dict
        modification_dict = generate_bias_by_groups(expected_total, group_configs)

st.sidebar.header("Evaluation Settings")
fairness_metric = st.sidebar.selectbox("Fairness Metric", ["DP", "EO"])
size_unit = st.sidebar.selectbox("Fairness Level (Size Unit)", ["attribute", "value", "attribute-value"])
max_parts_eval = st.sidebar.number_input("Max Partitions to Evaluate", min_value=1, value=num_partitions)

train_model_opt = st.sidebar.checkbox("Train Model for Fairness?")
model_choice = None
if train_model_opt:
    model_choice = st.sidebar.selectbox("Model Type", ["LogisticRegression", "DecisionTree"])

st.sidebar.markdown("---")
st.sidebar.header("Save Dataset")
save_path = st.sidebar.text_input("Save Path", "data/saved_dataset")
if st.sidebar.button("Save Dataset to Disk"):
    if "fds" in st.session_state:
        try:
            st.session_state["fds"].save_dataset(save_path)
            st.sidebar.success(f"Dataset saved to {save_path}")
        except Exception as e:  # noqa: BLE001
            st.error(f"Error saving dataset: {e}")
    else:
        st.sidebar.warning("Please load the dataset first.")


@st.cache_data
def get_raw_acs_data(dataset_name, states, year, horizon):
    """
    Load raw ACS data (before modification) and cache it.
    """
    return FairFederatedDataset.load_acs_raw_data(dataset_name, states, year, horizon)


if st.button("Load and Evaluate"):
    with st.spinner("Loading dataset..."):
        try:
            partitioners_config = {}
            preloaded_data = None

            if dataset_name in ["ACSIncome", "ACSEmployment"]:
                # "The ACS Income end Employment datasets should only be divided into different parts...
                # In particular, each state can be divided into N parts"
                if not selected_states:
                    st.error("Please select at least one state.")
                    st.stop()

                # Apply the partitioner to each selected state
                states_to_process = selected_states if selected_states is not None else []
                partitioners_config = {state: create_partitioner() for state in states_to_process}
                states_to_load = selected_states

                # Use cached data loading
                preloaded_data = get_raw_acs_data(dataset_name, selected_states, year, horizon)

            else:
                partitioners_config = {"train": create_partitioner()}
                states_to_load = None

            fds = FairFederatedDataset(
                dataset=dataset_name,
                subset=subset if "subset" in locals() else None,
                split=split if "split" in locals() else None,
                year=year,
                horizon=horizon,
                states=states_to_load,
                partitioners=partitioners_config,
                label_name=label_name,
                sensitive_attributes=sensitive_attributes,
                modification_dict=modification_dict,
                fairness_metric=fairness_metric,
                fairness_level=size_unit,
                seed=seed,
                shuffle=shuffle,
                preloaded_data=preloaded_data,
                client_names=client_names if "client_names" in locals() else None,
                sample_cap=sample_cap if sample_cap > 0 else None,
            )
            fds.prepare()

            # Iterative Mitigation Loop
            if inject_bias and any(g.get("mitigate", False) for g in group_configs):
                st.subheader("Mitigation Progress")
                max_iterations = 3
                for iteration in range(max_iterations):
                    st.write(f"Iteration {iteration + 1}: Checking clients...")

                    # 1. Compute current fairness
                    splits_to_check = selected_states if selected_states else ["train"]
                    all_met_threshold = True
                    failing_clients = []

                    for s in splits_to_check:
                        dataframe = compute_fairness(
                            partitioner=fds.partitioners[s],
                            partitioner_test=fds.partitioners[s],
                            model=None,
                            sens_att=sensitive_attributes[0] if sensitive_attributes else "SEX",
                            fairness_metric=fairness_metric,
                            label_name=fds.label_column,
                            size_unit=size_unit,
                            fds=fds,
                            split=s
                        )

                        # Check threshold
                        metric_col = dataframe.columns[0]
                        for idx, val in dataframe[metric_col].items():
                            if not (0 <= val <= mitigation_threshold):
                                all_met_threshold = False
                                failing_clients.append((s, idx))
                    
                    if all_met_threshold:
                        st.success(f"✅ All mitigated clients are within unfairness threshold (0 - {mitigation_threshold}).")
                        break
                    
                    if iteration < max_iterations - 1:
                        st.warning(f"⚠️ {len(failing_clients)} client(s) still above threshold. Re-balancing...")
                        # Re-run prepare (which re-applies balance_data) or manually call it?
                        # Since fds.prepare() uses balance_data which is randomized undersampling,
                        # calling it again on the same fds object isn't easy without resetting.
                        # For simplicity in this dashboard demo, we inform that perfect balance is hard with small data.
                        # In a real scenario, we'd iteratively prune.
                        fds.prepare() # Re-prepare might help due to randomness
                    else:
                        st.error("❌ Could not reach threshold after maximum iterations. Dataset might be too small or skewed.")

                st.write(f"📊 Total samples removed during mitigation: **{fds._total_removed_samples}**")

            # Save fds to session state for persistence
            st.session_state["fds"] = fds

            st.success("Dataset Loaded!")

            # Evaluate Fairness

            # Determine which splits to evaluate
            splits_to_eval = selected_states if dataset_name in ["ACSIncome", "ACSEmployment"] else ["train"]

            sens_att_to_use = sensitive_attributes[0] if sensitive_attributes else "SEX"

            # Helper to run compute_fairness over multiple splits
            def compute_all_fairness(splits, model_class=None):
                total_steps = 0
                for split in splits:
                    part_obj = fds.partitioners[split]
                    num_parts = min(max_parts_eval, part_obj.num_partitions)
                    total_steps += num_parts

                progress_bar = st.progress(0, text="Loading partitions...")
                steps_done = 0

                def progress_callback(_pid):
                    nonlocal steps_done
                    steps_done += 1
                    p = min(steps_done / total_steps, 1.0)
                    progress_bar.progress(p, text=f"Loading partitions... ({steps_done}/{total_steps})")

                all_results = []
                for split in splits:
                    part_obj = fds.partitioners[split]

                    model_instance = None
                    if model_class:
                        model_instance = model_class()  # Fresh instance per split (and re-fit per partition)

                    dataframe = compute_fairness(
                        partitioner=part_obj,
                        partitioner_test=part_obj,
                        model=model_instance,
                        sens_att=sens_att_to_use,
                        fairness_metric=fairness_metric,
                        label_name=fds.label_column,
                        size_unit=size_unit,
                        max_num_partitions=max_parts_eval,
                        progress_callback=progress_callback,
                        fds=fds,
                        split=split,
                    )
                    # Rename index to include split name
                    dataframe.index = [f"{split}_{i}" for i in dataframe.index]
                    all_results.append(dataframe)

                progress_bar.empty()
                return pd.concat(all_results)

            def get_display_df(df, unit):
                cols_to_keep = []
                if unit == "attribute":
                    cols_to_keep = [df.columns[0]]
                elif unit == "value":
                    cols_to_keep = [df.columns[0], df.columns[1]]
                else:
                    # For attribute-value, keep all fairness columns (those before Accuracy/Sample Count)
                    cols_to_keep = [c for c in df.columns if c not in ["Accuracy", "Sample Count"]]

                if "Accuracy" in df.columns:
                    cols_to_keep.append("Accuracy")
                if "Sample Count" in df.columns:
                    cols_to_keep.append("Sample Count")
                
                return df[cols_to_keep]

            # 1. Dataset Fairness (DP only)
            if fairness_metric == "DP":
                st.subheader("Dataset Fairness (Bias)")
                with st.spinner("Computing Dataset Bias..."):
                    df_data_fairness = compute_all_fairness(splits_to_eval, model_class=None)

                disp_data_df = get_display_df(df_data_fairness, size_unit)
                st.dataframe(disp_data_df)

                fig_d, ax_d = plt.subplots(figsize=(10, 5))
                metric_col = disp_data_df.columns[0]
                if metric_col in disp_data_df.columns:
                    disp_data_df[metric_col].plot(kind="bar", ax=ax_d)
                    ax_d.set_ylabel("Demographic Parity Difference")
                    ax_d.set_xlabel("Partition ID (State_ID)")
                    plt.xticks(rotation=45, ha="right")
                    st.pyplot(fig_d)
            elif not train_model_opt:
                st.warning(f"Metric '{fairness_metric}' requires a model. Please select 'Train Model for Fairness?'.")

            # 2. Model Fairness
            if train_model_opt:
                st.markdown("---")
                st.subheader(f"Model Fairness ({fairness_metric})")

                # Choose class
                m_class = None
                if model_choice == "LogisticRegression":
                    m_class = LogisticRegression
                elif model_choice == "DecisionTree":
                    m_class = DecisionTreeClassifier

                with st.spinner("Training Model & Computing Fairness..."):
                    df_model_fairness = compute_all_fairness(splits_to_eval, model_class=m_class)

                disp_model_df = get_display_df(df_model_fairness, size_unit)
                st.dataframe(disp_model_df)

                fig_m, ax_m = plt.subplots(figsize=(10, 5))
                metric_col_m = disp_model_df.columns[0]
                if metric_col_m in disp_model_df.columns:
                    disp_model_df[metric_col_m].plot(kind="bar", ax=ax_m, color="green")
                    ax_m.set_ylabel(f"{fairness_metric} Difference")
                    ax_m.set_xlabel("Partition ID (State_ID)")
                    plt.xticks(rotation=45, ha="right")
                    st.pyplot(fig_m)

                if "Accuracy" in disp_model_df.columns:
                    st.subheader("Model Accuracy")
                    fig_a, ax_a = plt.subplots(figsize=(10, 5))
                    disp_model_df["Accuracy"].plot(kind="bar", ax=ax_a, color="orange")
                    ax_a.set_ylabel("Accuracy")
                    ax_a.set_xlabel("Partition ID (State_ID)")
                    plt.xticks(rotation=45, ha="right")
                    st.pyplot(fig_a)

        except Exception as e:  # noqa: BLE001
            st.error(f"An error occurred: {e}")
            st.exception(e)

st.markdown("---")
if not st.session_state.get("fds"):
    st.info("Run this dashboard using: `uv run streamlit run dashboard/app.py`")
