# Comparison between XGBoost vs Baseline model - Cross silo - attribute
uv run python plots.py --dataset_name dutch --experiment_name cross_silo_attribute --wandb_url lucacorbucci/Feda4Fair_results_facct/runs/ubrioo7l --local_results_path ../local_models/results/dutch/dutch_cross_silo_attribute.json --experiment_type baseline --num_clients 50

# Comparison between XGBoost vs PUFFLE model - Cross silo - attribute
uv run python plots.py --dataset_name dutch --experiment_name cross_silo_attribute --wandb_url lucacorbucci/Feda4Fair_results_facct/runs/tg2n63au --local_results_path ../local_models/results/dutch/dutch_cross_silo_attribute.json --experiment_type puffle --num_clients 50

# Comparison between XGBoost vs Reweighing model - Cross silo - attribute
uv run python plots.py --dataset_name dutch --experiment_name cross_silo_attribute --wandb_url lucacorbucci/Feda4Fair_results_facct/runs/4cjrrx7n --local_results_path ../local_models/results/dutch/dutch_cross_silo_attribute.json --experiment_type reweighing --num_clients 50