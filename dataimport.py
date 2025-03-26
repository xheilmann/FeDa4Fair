#TODO:
# 1. overall
# -natural output (that can be used for normal FL)
# -modified output
# 2. dataset
# -dataset to be used in the generation of  client - level data, chosen among the ones provided in the ACS dataset
# -name of the dataset(Income, Employment)
# -data sampling strategy(range of datapoints per client(min, max))
# -include or exclude sensitive attributes
# 3.the number of clients that will be involved in the simulation and will need training data,
# -cross - silo(maximum: states)
# -cross - device(split in state, split by attributevalues(column name, percentage))
# 5. the sensitive attributes against which we want to measure the unfairness,
# - choosing(gender, race, mar)
# -binary vs non - binary(merging the smallest groups)
# - distribution of clients unfairneses
# - the unfairness level between different clients and their sensitive attributes.
# 6. output
# - dataset statistics per client(  # datapoints, fairness metrics (gender, race mar), performance metrics, modifications)
# -overall statistics global model FedAVG((  # datapoints, fairness metrics(gender, race mar), performance metrics) before modifications and after
# - global model on pooled dataset
# - the datasets as csv files, local model training as numpy array


def load_data (dataset_name, binary, individual_fairness, client_rage, num_clients, fairness_metric):
    #load data
    #split data if more than 50 clients
    #first evaluation gives back a fairness_dict of clients which are unfair towards the targets choosen
    return dataset

def choose_data (dataset, fairness_dict, unfairness_level, unfairness_distribution, output_strategy):
    if output_strategy == "natural":
        #if clients less than 50 choose the ones that fir most naturally to the given fairness requirements
        #ow save the current clients
        pass
    else:
        #fairness_modifications
        pass
