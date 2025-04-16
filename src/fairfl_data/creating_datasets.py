from statistics import LinearRegression

import pandas as pd
from sklearn.linear_model import LogisticRegression

from FairFederatedDataset import FairFederatedDataset
from evaluation import evaluate_models_on_datasets, local_client_fairness_plot
from sklearn.model_selection import train_test_split

#example mapping parameter:
#mapping: {"MAR": {2:1, 3:1, 4:1, 5:1}, "RAC1P": {8:6, 7:6, 9:6}}

#example for modification dict (this is always after the mapping!
modification_dict = {
"CT": {
"MAR":
{
"drop_rate": 0.2,
"flip_rate": 0.1,
"value": 2,
"attribute": "SEX",
"attribute_value": 2,
},
"SEX":
{
"drop_rate": 0.3,
"flip_rate": 0.2,
"value": 2,
"attribute": None,
"attribute_value": None,
},
}}

#df = pd.read_csv("/home/heilmann/Dokumente/fairFL-data/data_stats/RAC1P_DP_df.csv")
#fig = local_client_fairness_plot(df.iloc[:5],df.iloc[5:])
#fig.show()



ffds = FairFederatedDataset(dataset="ACSIncome",  fl_setting=None,
                            fairness_metric="DP", fairness_level="attribute",
                            model=LogisticRegression(max_iter=1000))


