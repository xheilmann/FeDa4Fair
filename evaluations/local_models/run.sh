#!/bin/bash

# Ensure we are in the correct directory (evaluations)
# Or assume run from local_models.
# Let's verify location.
# If this script is in local_models, we can cd ..
# But let's assume the user runs it as "./local_models/run.sh" from evaluations
# OR "cd local_models && ./run.sh".
# I'll make it robust.

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# SCRIPT_DIR is .../evaluations/local_models
EVALUATIONS_DIR="$SCRIPT_DIR/.."
DATASETS_DIR="$EVALUATIONS_DIR/../datasets"

echo "Running local model training..."

# Function to run training
run_training() {
    DATASET_PATH=$1
    DATASET_TYPE=$2
    DATASET_NAME=$3
    NUM_NODES=$4
    CROSS_SILO=$5
    
    SCENARIO=$6
    
    echo "------------------------------------------------"
    echo "Dataset: $DATASET_TYPE ($SCENARIO)"
    echo "Path: $DATASET_PATH"
    
    CMD="uv run python \"$SCRIPT_DIR/train_local.py\" \
        --dataset_path \"$DATASET_PATH\" \
        --dataset_type \"$DATASET_TYPE\" \
        --dataset_name \"$DATASET_NAME\" \
        --num_nodes \"$NUM_NODES\" \
        --cross_silo \"$CROSS_SILO\" \
        --output_dir \"$SCRIPT_DIR/results\""

    if [ -n "$SCENARIO" ]; then
        CMD="$CMD --scenario \"$SCENARIO\""
    fi
    
    eval $CMD
}

# Dutch Datasets
# Attribute
run_training "$DATASETS_DIR/dutch/cross_silo_attribute/medium" "dutch_cross_silo_attribute" "dutch_prepared" 50 "True" "medium"
run_training "$DATASETS_DIR/dutch/cross_silo_attribute/mild" "dutch_cross_silo_attribute" "dutch_prepared" 50 "True" "mild"
run_training "$DATASETS_DIR/dutch/cross_silo_attribute/strong" "dutch_cross_silo_attribute" "dutch_prepared" 50 "True" "strong"

# Value
run_training "$DATASETS_DIR/dutch/cross_silo_value/medium" "dutch_cross_silo_value" "dutch_prepared" 50 "True" "medium"
run_training "$DATASETS_DIR/dutch/cross_silo_value/mild" "dutch_cross_silo_value" "dutch_prepared" 50 "True" "mild"
run_training "$DATASETS_DIR/dutch/cross_silo_value/strong" "dutch_cross_silo_value" "dutch_prepared" 50 "True" "strong"

# ACS Income (Cross Device)
# Attribute
run_training "$DATASETS_DIR/acs_income/cross_device_attribute_final/FL_data" "acs_income_cross_device_attribute" "income_cross_device" 111 "False" ""
run_training "$DATASETS_DIR/acs_income/cross_silo_attribute_final/FL_data" "acs_income_cross_silo_attribute" "income_cross_silo" 111 "False" ""

# Value
run_training "$DATASETS_DIR/acs_income/cross_device_value_final/FL_data" "acs_income_cross_device_value" "income_cross_device" 51 "False" ""
run_training "$DATASETS_DIR/acs_income/cross_silo_value_final/FL_data" "acs_income_cross_silo_value" "income_cross_silo" 51 "False" ""

echo "Done."
