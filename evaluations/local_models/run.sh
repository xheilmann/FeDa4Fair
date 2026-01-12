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
    SENSITIVE_FEATURE=$7
    SECOND_SENSITIVE_FEATURE=$8
    TARGET=$9
    
    echo "------------------------------------------------"
    echo "Dataset: $DATASET_TYPE ($SCENARIO)"
    echo "Path: $DATASET_PATH"
    
    CMD="uv run python \"$SCRIPT_DIR/train_local.py\" \
        --dataset_path \"$DATASET_PATH\" \
        --dataset_type \"$DATASET_TYPE\" \
        --dataset_name \"$DATASET_NAME\" \
        --num_nodes \"$NUM_NODES\" \
        --cross_silo \"$CROSS_SILO\" \
        --sensitive_feature \"$SENSITIVE_FEATURE\" \
        --second_sensitive_feature \"$SECOND_SENSITIVE_FEATURE\" \
        --target \"$TARGET\" \
        --output_dir \"$SCRIPT_DIR/results\""

    if [ -n "$SCENARIO" ]; then
        CMD="$CMD --scenario \"$SCENARIO\""
    fi
    
    eval $CMD
}

# Dutch Datasets
# Attribute Cross-Silo
run_training "$DATASETS_DIR/dutch/cross_silo_attribute/medium" "dutch_cross_silo_attribute" "dutch_prepared" 50 "True" "medium" "sex_binary" "Marital_status" "occupation_binary"

# Value Cross-Silo
run_training "$DATASETS_DIR/dutch/cross_silo_value/medium" "dutch_cross_silo_value" "dutch_prepared" 50 "True" "medium" "sex_binary" "" "occupation_binary"

# Attribute Cross-Device
run_training "$DATASETS_DIR/dutch/cross_device_attribute/medium" "dutch_cross_device_attribute" "dutch_prepared" 150 "False" "medium" "sex_binary" "Marital_status" "occupation_binary"

# Value Cross-Device
run_training "$DATASETS_DIR/dutch/cross_device_value/medium" "dutch_cross_device_value" "dutch_prepared" 150 "False" "medium" "sex_binary" "" "occupation_binary"


echo "Done."
