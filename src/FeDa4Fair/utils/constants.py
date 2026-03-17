# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Constants used across the FeDa4Fair project."""

# Dataset Names
ACS_INCOME = "ACSIncome"
ACS_EMPLOYMENT = "ACSEmployment"

# Sensitive Attributes
SEX = "SEX"
MARITAL_STATUS = "MAR"
RACE = "RACE"
ACS_RACE = "RAC1P"
DEFAULT_SENSITIVE_ATTRIBUTES = [SEX, MARITAL_STATUS, RACE]

# Labels
INCOME_LABEL = "PINCP"
EMPLOYMENT_LABEL = "ESR"

# Fairness Metrics
DEMOGRAPHIC_PARITY = "DP"
EQUALIZED_ODDS = "EO"

# Fairness Levels
LEVEL_ATTRIBUTE = "attribute"
LEVEL_VALUE = "value"
LEVEL_ATTRIBUTE_VALUE = "attribute-value"

# FL Settings
CROSS_SILO = "cross-silo"
CROSS_DEVICE = "cross-device"

# Default Values
DEFAULT_SEED = 42
DEFAULT_TEST_SIZE = 0.2
DEFAULT_TRAIN_VAL_TEST_SPLIT = [0.7, 0.15, 0.15]

# Generation Thresholds
MIN_BIAS_THRESHOLD = 0.09
RACE_BIAS_LOWER_BOUND = 0.12
RACE_BIAS_UPPER_BOUND = 0.175
