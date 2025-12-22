import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/FeDa4Fair")))

from FairFederatedDataset import FairFederatedDataset
from fairness_computation import compute_fairness
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner

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

dataset_name = st.sidebar.selectbox("Select Dataset", ["ACSIncome", "ACSEmployment", "lucacorbucci/Dutch_Census"])

selected_states = None
if dataset_name in ["ACSIncome", "ACSEmployment"]:
    label_name = None  # Inferred
    sensitive_attributes = None  # Default
    year = st.sidebar.selectbox("Year", ["2014", "2015", "2016", "2017", "2018"], index=4)
    horizon = st.sidebar.selectbox("Horizon", ["1-Year", "5-Year"], index=0)

    select_all = st.sidebar.checkbox("Select All States")
    default_states = US_STATES if select_all else ["CA"]
    selected_states = st.sidebar.multiselect("Select States to Load", US_STATES, default=default_states)
else:
    label_name = "occupation_binary"
    sensitive_attributes = ["sex_binary"]
    year = "2018"  # Dummy
    horizon = "1-Year"  # Dummy
    selected_states = None

seed = st.sidebar.number_input("Random Seed", value=42)
shuffle = st.sidebar.checkbox("Shuffle Data?", value=True)

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
    st.sidebar.subheader("Bias Settings")

    # Determine options for Target Split/State
    if dataset_name in ["ACSIncome", "ACSEmployment"]:
        target_options = selected_states if selected_states else US_STATES
        target_state = st.sidebar.selectbox("Target Split/State", target_options)
    else:
        target_state = st.sidebar.text_input("Target Split/State (e.g., train)", "train")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        drop_rate = st.sidebar.slider("Drop Rate", 0.0, 1.0, 0.0)
    with col2:
        flip_rate = st.sidebar.slider("Flip Rate", 0.0, 1.0, 0.0)

    # Simplified modification for demo
    sens_attr_name = st.sidebar.text_input("Sensitive Attribute Name", "SEX" if "ACS" in dataset_name else "sex_binary")
    sens_attr_val = st.sidebar.number_input("Sensitive Attribute Value", value=1)

    target_label_val = st.sidebar.number_input("Target Label Value (to drop/flip)", value=1)

    if drop_rate > 0 or flip_rate > 0:
        modification_dict = {
            target_state: {
                sens_attr_name: {
                    "drop_rate": drop_rate,
                    "flip_rate": flip_rate,
                    "value": target_label_val,
                    "attribute": sens_attr_name,
                    "attribute_value": sens_attr_val,
                }
            }
        }

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
                partitioners_config = {state: create_partitioner() for state in selected_states}
                states_to_load = selected_states

                # Use cached data loading
                preloaded_data = get_raw_acs_data(dataset_name, selected_states, year, horizon)

            else:
                partitioners_config = {"train": create_partitioner()}
                states_to_load = None

            fds = FairFederatedDataset(
                dataset=dataset_name,
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
            )
            fds.prepare()

            # Save fds to session state for persistence
            st.session_state["fds"] = fds

            st.success("Dataset Loaded!")

            # Evaluate Fairness

            # Determine which splits to evaluate
            splits_to_eval = selected_states if dataset_name in ["ACSIncome", "ACSEmployment"] else ["train"]

            sens_att_to_use = sensitive_attributes[0] if sensitive_attributes else "SEX"

            # Helper to run compute_fairness over multiple splits
            def compute_all_fairness(splits, model_class=None):
                all_results = []
                for split in splits:
                     part_obj = fds.partitioners[split]

                     model_instance = None
                     if model_class:
                         model_instance = model_class() # Fresh instance per split (and re-fit per partition)

                     dataframe = compute_fairness(
                        partitioner=part_obj,
                        partitioner_test=part_obj,
                        model=model_instance,
                        sens_att=sens_att_to_use,
                        fairness_metric=fairness_metric,
                        label_name=fds.label_column,
                        size_unit=size_unit,
                        max_num_partitions=max_parts_eval,
                     )
                     # Rename index to include split name
                     dataframe.index = [f"{split}_{i}" for i in dataframe.index]
                     all_results.append(dataframe)
                return pd.concat(all_results)

            # Helper for display logic
            def get_display_df(df, unit):
                acc_col = None
                if "Accuracy" in df.columns:
                    acc_col = df[["Accuracy"]]
                    df_vals = df.drop(columns=["Accuracy"])
                else:
                    df_vals = df

                if unit == "attribute":
                    d_df = df_vals.iloc[:, :1]
                elif unit == "value":
                    d_df = df_vals.iloc[:, :2]
                else:
                    d_df = df_vals

                if acc_col is not None:
                    d_df = pd.concat([d_df, acc_col], axis=1)
                return d_df

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
st.info("Run this dashboard using: `uv run streamlit run dashboard/app.py`")
