from dataimport import load_data, choose_data
from evaluation import evaluate


def make_FL_dataset(
    dataset_name,
    binary,
    individual_fairness,
    client_rage,
    num_clients,
    fairness_metric,
    unfairness_level,
    unfairness_distribution,
    output_strategy,
):
    k = load_data()
    print("k")
    evaluate()
    choose_data()
    evaluate()
    # save_data()


load_data()
