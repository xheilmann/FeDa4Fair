# # Comparison between XGBoost vs Baseline model - Cross silo - attribute
# uv run python plots.py --dataset_name dutch --experiment_name cross_silo_attribute --wandb_url lucacorbucci/Feda4Fair_results_facct/runs/ubrioo7l --local_results_path ../local_models/results/dutch/dutch_cross_silo_attribute.json --experiment_type baseline --num_clients 50

# # Comparison between XGBoost vs PUFFLE model - Cross silo - attribute
# uv run python plots.py --dataset_name dutch --experiment_name cross_silo_attribute --wandb_url lucacorbucci/Feda4Fair_results_facct/runs/tg2n63au --local_results_path ../local_models/results/dutch/dutch_cross_silo_attribute.json --experiment_type puffle --num_clients 50

# # Comparison between XGBoost vs Reweighing model - Cross silo - attribute
# uv run python plots.py --dataset_name dutch --experiment_name cross_silo_attribute --wandb_url lucacorbucci/Feda4Fair_results_facct/runs/4cjrrx7n --local_results_path ../local_models/results/dutch/dutch_cross_silo_attribute.json --experiment_type reweighing --num_clients 50



# Comparison between XGBoost vs Baseline model - Cross device - attribute
uv run python plots.py --dataset_name dutch --experiment_name cross_device_attribute --wandb_url lucacorbucci/Feda4Fair_results_facct/runs/ddpdesxb --local_results_path ../local_models/results/dutch/dutch_cross_device_attribute.json --experiment_type baseline --num_clients 150

# Comparison between XGBoost vs PUFFLE model - Cross device - attribute
uv run python plots.py --dataset_name dutch --experiment_name cross_device_attribute --wandb_url lucacorbucci/Feda4Fair_results_facct/runs/sz4f5po3 --local_results_path ../local_models/results/dutch/dutch_cross_device_attribute.json --experiment_type puffle --num_clients 150

# Comparison between XGBoost vs Reweighing model - Cross device - attribute
uv run python plots.py --dataset_name dutch --experiment_name cross_device_attribute --wandb_url lucacorbucci/Feda4Fair_results_facct/runs/s5aa6cof --local_results_path ../local_models/results/dutch/dutch_cross_device_attribute.json --experiment_type reweighing --num_clients 150




# Comparison between XGBoost vs Baseline model - Cross silo - value
uv run python plots.py --dataset_name dutch --experiment_name cross_silo_value --wandb_url lucacorbucci/Feda4Fair_results_facct/runs/aj2p865a --local_results_path ../local_models/results/dutch/dutch_cross_silo_value.json --experiment_type baseline --num_clients 50

# Comparison between XGBoost vs PUFFLE model - Cross silo - value
uv run python plots.py --dataset_name dutch --experiment_name cross_silo_value --wandb_url lucacorbucci/Feda4Fair_results_facct/runs/mvbx1aaa --local_results_path ../local_models/results/dutch/dutch_cross_silo_value.json --experiment_type puffle --num_clients 50

# Comparison between XGBoost vs Reweighing model - Cross silo - value
uv run python plots.py --dataset_name dutch --experiment_name cross_silo_value --wandb_url lucacorbucci/Feda4Fair_results_facct/runs/eefwjyt2 --local_results_path ../local_models/results/dutch/dutch_cross_silo_value.json --experiment_type reweighing --num_clients 50



# Comparison between XGBoost vs Baseline model - Cross device - value
uv run python plots.py --dataset_name dutch --experiment_name cross_device_value --wandb_url lucacorbucci/Feda4Fair_results_facct/runs/7vda4r2y --local_results_path ../local_models/results/dutch/dutch_cross_device_value.json --experiment_type baseline --num_clients 150

# Comparison between XGBoost vs PUFFLE model - Cross device - value
uv run python plots.py --dataset_name dutch --experiment_name cross_device_value --wandb_url lucacorbucci/Feda4Fair_results_facct/runs/sh142gas --local_results_path ../local_models/results/dutch/dutch_cross_device_value.json --experiment_type puffle --num_clients 150

# Comparison between XGBoost vs Reweighing model - Cross device - value
uv run python plots.py --dataset_name dutch --experiment_name cross_device_value --wandb_url lucacorbucci/Feda4Fair_results_facct/runs/b4g5ihov --local_results_path ../local_models/results/dutch/dutch_cross_device_value.json --experiment_type reweighing --num_clients 150