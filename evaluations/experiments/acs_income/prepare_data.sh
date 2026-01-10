#!/bin/bash

echo "Preparing ACS Income datasets..."

# Prepare cross-device attribute data
echo "Processing cross_device_attribute_final..."
uv run python pre_processing.py --folder_name ../../../datasets/acs_income/cross_device_attribute_final/

# Prepare cross-silo attribute data
echo "Processing cross_silo_attribute_final..."
uv run python pre_processing.py --folder_name ../../../datasets/acs_income/cross_silo_attribute_final/ --cross_silo True

# Prepare cross-device value data
echo "Processing cross_device_value_final..."
uv run python pre_processing.py --folder_name ../../../datasets/acs_income/cross_device_value_final/

# Prepare cross-silo value data
echo "Processing cross_silo_value_final..."
uv run python pre_processing.py --folder_name ../../../datasets/acs_income/cross_silo_value_final/ --cross_silo True

echo "Preparation complete."
