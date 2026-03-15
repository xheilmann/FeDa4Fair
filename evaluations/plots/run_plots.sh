# # DUTCH 

# # Comparison between XGBoost vs Baseline model - Cross silo - attribute
# uv run python plots.py --dataset_name dutch --experiment_name cross_silo_attribute --wandb_url lucacorbucci/Feda4Fair_results_facct_dutch/runs/135sv59a --local_results_path ../local_models/results/dutch/dutch_cross_silo_attribute.json --experiment_type baseline --num_clients 50

# # Comparison between XGBoost vs PUFFLE model - Cross silo - attribute
# uv run python plots.py --dataset_name dutch --experiment_name cross_silo_attribute --wandb_url lucacorbucci/Feda4Fair_results_facct_dutch/runs/pb46ehjo --local_results_path ../local_models/results/dutch/dutch_cross_silo_attribute.json --experiment_type puffle --num_clients 50

# # Comparison between XGBoost vs Reweighing model - Cross silo - attribute
# uv run python plots.py --dataset_name dutch --experiment_name cross_silo_attribute --wandb_url lucacorbucci/Feda4Fair_results_facct_dutch/runs/bxwefpcd --local_results_path ../local_models/results/dutch/dutch_cross_silo_attribute.json --experiment_type reweighing --num_clients 50


# # Comparison between XGBoost vs Baseline model - Cross device - attribute
# uv run python plots.py --dataset_name dutch --experiment_name cross_device_attribute --wandb_url lucacorbucci/Feda4Fair_results_facct_dutch/runs/08ytmic4 --local_results_path ../local_models/results/dutch/dutch_cross_device_attribute.json --experiment_type baseline --num_clients 150

# # Comparison between XGBoost vs PUFFLE model - Cross device - attribute
# uv run python plots.py --dataset_name dutch --experiment_name cross_device_attribute --wandb_url lucacorbucci/Feda4Fair_results_facct_dutch/runs/5pmdt20z --local_results_path ../local_models/results/dutch/dutch_cross_device_attribute.json --experiment_type puffle --num_clients 150

# # Comparison between XGBoost vs Reweighing model - Cross device - attribute
# uv run python plots.py --dataset_name dutch --experiment_name cross_device_attribute --wandb_url lucacorbucci/Feda4Fair_results_facct_dutch/runs/0qsnk05s --local_results_path ../local_models/results/dutch/dutch_cross_device_attribute.json --experiment_type reweighing --num_clients 150


# # Comparison between XGBoost vs Baseline model - Cross silo - value
# uv run python plots.py --dataset_name dutch --experiment_name cross_silo_value --wandb_url lucacorbucci/Feda4Fair_results_facct_dutch/runs/135sv59a --local_results_path ../local_models/results/dutch/dutch_cross_silo_value.json --experiment_type baseline --num_clients 50

# # Comparison between XGBoost vs PUFFLE model - Cross silo - value
# uv run python plots.py --dataset_name dutch --experiment_name cross_silo_value --wandb_url lucacorbucci/Feda4Fair_results_facct_dutch/runs/3y649jvs --local_results_path ../local_models/results/dutch/dutch_cross_silo_value.json --experiment_type puffle --num_clients 50

# # Comparison between XGBoost vs Reweighing model - Cross silo - value
# uv run python plots.py --dataset_name dutch --experiment_name cross_silo_value --wandb_url lucacorbucci/Feda4Fair_results_facct_dutch/runs/9anx9rur --local_results_path ../local_models/results/dutch/dutch_cross_silo_value.json --experiment_type reweighing --num_clients 50


# # Comparison between XGBoost vs Baseline model - Cross device - value
# uv run python plots.py --dataset_name dutch --experiment_name cross_device_value --wandb_url lucacorbucci/Feda4Fair_results_facct_dutch/runs/0pqrfcen --local_results_path ../local_models/results/dutch/dutch_cross_device_value.json --experiment_type baseline --num_clients 150

# # Comparison between XGBoost vs PUFFLE model - Cross device - value
# uv run python plots.py --dataset_name dutch --experiment_name cross_device_value --wandb_url lucacorbucci/Feda4Fair_results_facct_dutch/runs/tz4ng9ay --local_results_path ../local_models/results/dutch/dutch_cross_device_value.json --experiment_type puffle --num_clients 150

# # Comparison between XGBoost vs Reweighing model - Cross device - value
# uv run python plots.py --dataset_name dutch --experiment_name cross_device_value --wandb_url lucacorbucci/Feda4Fair_results_facct_dutch/runs/zd17ugjy --local_results_path ../local_models/results/dutch/dutch_cross_device_value.json --experiment_type reweighing --num_clients 150


# # # ACS INCOME

# Comparison between XGBoost vs Reweighing model - Cross device - attribute - BARPLOT OK
#uv run python plots.py --dataset_name acs_income --experiment_name reweighing_device_attribute_income --wandb_url lucacorbucci/Feda4Fair_results_facct/runs/cff2cpkf --local_results_path ../local_models/results/acs_income/baseline_attribute_cross_device/model_perf_DP.csv --experiment_type reweighing --partition_names_path ../../datasets/acs_income/cross_device_attribute_final/FL_data/federated/partitions_names.json

# Comparison between XGBoost vs Reweighing model - Cross silo - attribute
#uv run python plots.py --num_clients 51 --dataset_name acs_income --experiment_name reweighing_silo_attribute_income --wandb_url lucacorbucci/Feda4Fair_results_facct/runs/i39i4j66 --local_results_path ../local_models/results/acs_income/baseline_attribute_cross_silo/model_perf_DP.csv --experiment_type reweighing --partition_names_path ../../datasets/acs_income/cross_silo_attribute_final/FL_data/federated/partitions_names.json --strip_dataset_suffix

# Comparison between XGBoost vs Reweighing model - Cross device - value - BARPLOT OK
uv run python plots.py --dataset_name acs_income --experiment_name reweighing_device_value_income --wandb_url lucacorbucci/Feda4Fair_results_facct/runs/na1alzqf --local_results_path ../local_models/results/acs_income/baseline_value_cross_device/model_perf_DP.csv --experiment_type reweighing --partition_names_path ../../datasets/acs_income/cross_device_value_final/FL_data/federated/partitions_names.json
# 
# Comparison between XGBoost vs Reweighing model - Cross silo - value
uv run python plots.py --num_clients 51 --dataset_name acs_income --experiment_name reweighing_silo_value_income --wandb_url lucacorbucci/Feda4Fair_results_facct/runs/cm7blbsw --local_results_path ../local_models/results/acs_income/baseline_value_cross_silo/model_perf_DP.csv --experiment_type reweighing --partition_names_path ../../datasets/acs_income/cross_silo_value_final/FL_data/federated/partitions_names.json --strip_dataset_suffix

# # CELEBA
# uv run python plots_celeba.py
