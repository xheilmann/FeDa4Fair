# Datasheet for dataset FeDa4Fair-cross-device-value-bias

Questions from the [Datasheets for Datasets](https://arxiv.org/abs/1803.09010) paper, v7.

Jump to section:

- [Motivation](#motivation)
- [Composition](#composition)
- [Collection process](#collection-process)
- [Preprocessing/cleaning/labeling](#preprocessingcleaninglabeling)
- [Uses](#uses)
- [Distribution](#distribution)
- [Maintenance](#maintenance)

## Motivation

_The questions in this section are primarily intended to encourage dataset creators
to clearly articulate their reasons for creating the dataset and to promote transparency
about funding interests._

### For what purpose was the dataset created? 

_Was there a specific task in mind? Was there a specific gap that needed to be filled?
Please provide a description._

This dataset was created for the purpose of testing bias mitigation techniques in federated learning via the [FeDa4Fair](https://github.com/xheilmann/FeDa4Fair) library.
This dataset was created so that it represents **value bias** in a federated learning context, by which we mean that clients have data biased toward different values of the same sensitive attribute.
FeDa4Fair is built on top of the [ACS PUMS](https://www.census.gov/programs-surveys/acs/microdata.html) subset of the US census data 
obtained via the [folktables](https://github.com/social-foundations/folktables) APIs.
To clarify the relationship between the present dataset and the U.S. census data:

* The American Community Survey (ACS) is a yearly initiative undertaken by the U.S. Census Bureau with the objective to represent demographics and social status in the U.S.

* The ACS Public Use Microdata Sample (ACS PUMS) is a subset of the ACS (about 1 percent) which is released to the general public.

* Folktables is a library that uses API endpoints offered by the Census Bureau to download PUMS data and test it in the space of algorithmic fairness.

* FeDa4Fair is a library that employs Folktables to offer data, divided at the U.S. state level and further, to employ it in the space of federated learning 
and fairness. FeDa4Fair offers some flexibility in the sense that researchers might employ different partitioning of the data beyond the state-level, for instance
to test client-level federated learning techniques.

* The present dataset was obtained with FeDa4Fair in April 2025.

### Who created the dataset (e.g., which team, research group) and on behalf of which entity (e.g., company, institution, organization)?

FeDa4Fair is created and maintained by Xenia Heilmann (JGU Mainz), Luca Corbucci (University of Pisa), Anna Monreale (University of Pisa) and Mattia Cerrato (JGU Mainz).
The same authors generated this dataset.

### Who funded the creation of the dataset? 

For FeDa4Fair: XH and MC were funded by the “TOPML: Trading Off Non-Functional Properties of Machine Learning” project funded by the
Carl-Zeiss-Stiftung in the Förderprogramm “Durchbrüche”, identifying code P2021-02-014.
LC was funded by The European Union Horizon 2020 program under grant agreement No. 101120763 (TANGO).
AM was funded by the National Recovery and Resilience Plan (PNRR), under agreements: PNRR - M4C2 - Investimento 1.3, Partenariato Esteso PE00000013 - "FAIR - Future Artificial Intelligence Research" - Spoke 1 "Human-centered AI"

### Any other comments?

Not at this time.

## Composition

_Most of these questions are intended to provide dataset consumers with the
information they need to make informed decisions about using the dataset for
specific tasks. The answers to some of these questions reveal information
about compliance with the EU’s General Data Protection Regulation (GDPR) or
comparable regulations in other jurisdictions._

### What do the instances that comprise the dataset represent (e.g., documents, photos, people, countries)?

_Are there multiple types of instances (e.g., movies, users, and ratings; people and
interactions between them; nodes and edges)? Please provide a description._

All instances in this dataset represent demographic and socio-economic characteristics of individuals or households living in the US in the period 2005-2023 as collected by the U.S. Census.
The data is partially anonymized. This dataset in particular includes data from individuals in the year 2018.

### How many instances are there in total (of each type, if appropriate)?

The present dataset has 1144252 rows across all states represented in the ACS PUMS. We assume here that every state represents a client in a cross-device federated learning task. State data is each in its own CSV, each with the following number of rows:

```
Row Counts for Each File:
AK_0.csv: 1155
AK_1.csv: 1154
AK_2.csv: 1154
AL_0.csv: 7423
AL_2.csv: 7422
AR_0.csv: 4642
AR_1.csv: 4642
AR_2.csv: 4641
AZ_0.csv: 11074
AZ_1.csv: 11074
AZ_2.csv: 11073
CA_1.csv: 65222
CA_2.csv: 65221
CO_1.csv: 10435
CO_2.csv: 10435
CT_1.csv: 6595
CT_2.csv: 6595
DE_0.csv: 1571
DE_1.csv: 1571
DE_2.csv: 1570
FL_0.csv: 32975
GA_0.csv: 16972
GA_1.csv: 16972
GA_2.csv: 16971
HI_0.csv: 2577
HI_1.csv: 2577
HI_2.csv: 2577
ID_0.csv: 2755
ID_1.csv: 2755
ID_2.csv: 2755
IL_0.csv: 22339
IL_1.csv: 22339
IN_0.csv: 11674
IN_2.csv: 11674
KS_0.csv: 5269
KS_2.csv: 5269
KY_0.csv: 7336
KY_2.csv: 7335
LA_1.csv: 6874
MA_0.csv: 13372
MD_0.csv: 11014
MD_1.csv: 11014
MD_2.csv: 11014
ME_1.csv: 2334
ME_2.csv: 2334
MI_1.csv: 16669
MI_2.csv: 16669
MN_0.csv: 10331
MN_1.csv: 10330
MN_2.csv: 10330
MO_0.csv: 10555
MO_1.csv: 10555
MO_2.csv: 10554
MS_0.csv: 4396
MS_1.csv: 4395
MT_0.csv: 1821
MT_1.csv: 1821
MT_2.csv: 1821
NC_0.csv: 17356
NC_1.csv: 17356
NC_2.csv: 17355
ND_0.csv: 1485
ND_1.csv: 1485
NE_0.csv: 3594
NE_1.csv: 3593
NE_2.csv: 3593
NH_2.csv: 2655
NJ_1.csv: 15927
NJ_2.csv: 15927
NM_0.csv: 2904
NV_0.csv: 4936
NV_2.csv: 4935
NY_0.csv: 34341
NY_1.csv: 34340
NY_2.csv: 34340
OH_0.csv: 20712
PA_0.csv: 22770
PA_2.csv: 22769
RI_0.csv: 1904
RI_2.csv: 1904
SC_1.csv: 8293
SC_2.csv: 8293
SD_1.csv: 1633
SD_2.csv: 1633
TN_0.csv: 11335
TN_2.csv: 11334
TX_0.csv: 45308
TX_1.csv: 45308
TX_2.csv: 45308
VA_2.csv: 15381
VT_0.csv: 1256
VT_1.csv: 1256
WA_1.csv: 13315
WA_2.csv: 13314
WI_0.csv: 10897
WI_1.csv: 10897
WI_2.csv: 10896
WV_0.csv: 2700
WV_2.csv: 2699
WY_0.csv: 1022
```

### Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set?

_If the dataset is a sample, then what is the larger set? Is the sample representative
of the larger set (e.g., geographic coverage)? If so, please describe how this
representativeness was validated/verified. If it is not representative of the larger set,
please describe why not (e.g., to cover a more diverse range of instances, because
instances were withheld or unavailable)._

The ACS PUMS is a partially anonymized sample and is currently designed to include one percent of the households and inviduals in the U.S.
and Puerto Rico. A full documentation of the 2023 ACS PUMS is available [here](https://www2.census.gov/programs-surveys/acs/tech_docs/pums/accuracy/2023AccuracyPUMS.pdf)
as of 16/04/2025. To avoid re-identification of individuals or households, various techniques have been employed on the data by the U.S. Census Bureau.
Among these, certain feature values might be swapped across rows or synthetic data might have been employed in place of the actual ones measured.
These treatments are done at the level of the "first release" of the ACS PUMS and are not part of the treatment of FeDa4Fair or of the creators of this
present dataset.

### What data does each instance consist of? 

_“Raw” data (e.g., unprocessed text or images) or features? In either case, please
provide a description._

The data column names are as follows: 

```
AGEP,COW,SCHL,MAR,OCCP,POBP,RELP,WKHP,SEX,RAC1P,PINCP. 
```

We refer to the ACS PUMS documentation from the year 2018 for a human-readable description.


### Is there a label or target associated with each instance?

_If so, please provide a description._

This dataset is based on the ACSIncome folktables task. As such, the ground truth is whether an individual or household has
earned more than 50 thousand dollars the year prior to the census survey.

### Is any information missing from individual instances?

_If so, please provide a description, explaining why this information is missing (e.g.,
because it was unavailable). This does not include intentionally removed information,
but might include, e.g., redacted text._

The ACS PUMS contains many demographic and socioeconomic features which have not been included in this dataset.
FeDa4Fair follows the design choices made by the authors of the folktables library which have established patterns of "unfairness" in 
machine learning and statistical models trained on the present variables.

### Are relationships between individual instances made explicit (e.g., users’ movie ratings, social network links)?

_If so, please describe how these relationships are made explicit._
N/A

### Are there recommended data splits (e.g., training, development/validation, testing)?

_If so, please provide a description of these splits, explaining the rationale behind them._
No.

### Are there any errors, sources of noise, or redundancies in the dataset?

_If so, please provide a description._

Yes: the ACS PUMS contains factual inaccuracies inserted for the purpose of maintaining privacy and is in general affected by systematic and non-systematic
errors. For reference, see [here](https://www2.census.gov/programs-surveys/acs/tech_docs/pums/accuracy/2023AccuracyPUMS.pdf) for the year 2023.

### Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets)?

_If it links to or relies on external resources, a) are there guarantees that they will
exist, and remain constant, over time; b) are there official archival versions of the
complete dataset (i.e., including the external resources as they existed at the time the
dataset was created); c) are there any restrictions (e.g., licenses, fees) associated with
any of the external resources that might apply to a future user? Please provide descriptions
of all external resources and any restrictions associated with them, as well as links or other
access points, as appropriate._

It relies on the ACS PUMS and is a subset of that.

### Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by doctor-patient confidentiality, data that includes the content of individuals’ non-public communications)?

_If so, please provide a description._

Yes, the ACS PUMS contains potentially identifiable information. Significant attention has been put by the U.S. Census Bureau to prevent re-identification
of individuals and/or households. As of April 2025, we do not know of publicly reported incidents related to re-identification.

### Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety?

_If so, please describe why._

No.

### Does the dataset relate to people? 

_If not, you may skip the remaining questions in this section._

Yes.

### Does the dataset identify any subpopulations (e.g., by age, gender)?

_If so, please describe how these subpopulations are identified and provide a description of
their respective distributions within the dataset._

Yes. 

RAC1P Proportions in Each File:
```
AK_0.csv: {'1.0': 685, '4.0': 250, '5.0': 115, '3.0': 77, '2.0': 28}
AK_1.csv: {'1.0': 712, '4.0': 256, '5.0': 95, '3.0': 66, '2.0': 25}
AK_2.csv: {'4.0': 261, '1.0': 722, '5.0': 94, '3.0': 58, '2.0': 19}
AL_0.csv: {'1.0': 5666, '2.0': 1432, '5.0': 183, '3.0': 103, '4.0': 39}
AL_2.csv: {'1.0': 5670, '2.0': 1467, '3.0': 98, '5.0': 156, '4.0': 31}
AR_0.csv: {'1.0': 3867, '5.0': 210, '4.0': 28, '2.0': 472, '3.0': 65}
AR_1.csv: {'1.0': 3886, '5.0': 187, '2.0': 475, '3.0': 61, '4.0': 33}
AR_2.csv: {'1.0': 3876, '5.0': 183, '3.0': 71, '2.0': 482, '4.0': 29}
AZ_0.csv: {'1.0': 8756, '5.0': 898, '2.0': 442, '4.0': 567, '3.0': 411}
AZ_1.csv: {'5.0': 928, '1.0': 8722, '2.0': 466, '4.0': 553, '3.0': 405}
AZ_2.csv: {'1.0': 8714, '2.0': 457, '5.0': 961, '3.0': 403, '4.0': 538}
CA_1.csv: {'3.0': 10905, '1.0': 40203, '5.0': 10659, '2.0': 2865, '4.0': 590}
CA_2.csv: {'3.0': 10875, '1.0': 40440, '5.0': 10483, '2.0': 2845, '4.0': 578}
CO_1.csv: {'1.0': 9192, '5.0': 567, '3.0': 320, '2.0': 262, '4.0': 94}
CO_2.csv: {'1.0': 9195, '5.0': 556, '3.0': 319, '2.0': 273, '4.0': 92}
CT_1.csv: {'1.0': 5361, '2.0': 531, '3.0': 316, '5.0': 373, '4.0': 14}
CT_2.csv: {'1.0': 5434, '2.0': 505, '5.0': 360, '3.0': 278, '4.0': 18}
DE_0.csv: {'2.0': 258, '1.0': 1203, '3.0': 62, '5.0': 42, '4.0': 6}
DE_1.csv: {'1.0': 1196, '2.0': 257, '4.0': 9, '3.0': 49, '5.0': 60}
DE_2.csv: {'1.0': 1210, '3.0': 57, '2.0': 241, '5.0': 53, '4.0': 9}
FL_0.csv: {'1.0': 26361, '2.0': 3844, '5.0': 1679, '3.0': 993, '4.0': 98}
GA_0.csv: {'1.0': 11543, '2.0': 3988, '5.0': 660, '3.0': 740, '4.0': 41}
GA_1.csv: {'1.0': 11411, '3.0': 734, '2.0': 4116, '5.0': 668, '4.0': 43}
GA_2.csv: {'2.0': 4047, '1.0': 11487, '3.0': 714, '5.0': 670, '4.0': 53}
HI_0.csv: {'1.0': 707, '3.0': 977, '5.0': 828, '2.0': 52, '4.0': 13}
HI_1.csv: {'1.0': 700, '5.0': 866, '3.0': 948, '2.0': 60, '4.0': 3}
HI_2.csv: {'3.0': 962, '5.0': 903, '1.0': 663, '2.0': 45, '4.0': 4}
ID_0.csv: {'1.0': 2531, '5.0': 130, '4.0': 40, '3.0': 39, '2.0': 15}
ID_1.csv: {'1.0': 2549, '3.0': 29, '5.0': 120, '2.0': 15, '4.0': 42}
ID_2.csv: {'1.0': 2524, '5.0': 126, '4.0': 55, '2.0': 18, '3.0': 32}
IL_0.csv: {'1.0': 18094, '3.0': 1154, '4.0': 40, '5.0': 1182, '2.0': 1869}
IL_1.csv: {'1.0': 18110, '5.0': 1184, '2.0': 1891, '3.0': 1113, '4.0': 41}
IN_0.csv: {'1.0': 10359, '5.0': 364, '2.0': 695, '3.0': 225, '4.0': 31}
IN_2.csv: {'2.0': 667, '1.0': 10397, '3.0': 213, '5.0': 380, '4.0': 17}
KS_0.csv: {'1.0': 4670, '5.0': 234, '3.0': 113, '4.0': 51, '2.0': 201}
KS_2.csv: {'1.0': 4674, '5.0': 243, '4.0': 47, '2.0': 197, '3.0': 108}
KY_0.csv: {'1.0': 6633, '2.0': 432, '5.0': 171, '4.0': 12, '3.0': 88}
KY_2.csv: {'1.0': 6650, '2.0': 405, '5.0': 160, '3.0': 111, '4.0': 9}
LA_1.csv: {'1.0': 4878, '2.0': 1620, '3.0': 154, '5.0': 185, '4.0': 37}
MA_0.csv: {'1.0': 10907, '3.0': 933, '2.0': 774, '5.0': 728, '4.0': 30}
MD_0.csv: {'1.0': 6982, '2.0': 2614, '3.0': 742, '5.0': 657, '4.0': 19}
MD_1.csv: {'1.0': 6985, '2.0': 2617, '3.0': 723, '5.0': 654, '4.0': 35}
MD_2.csv: {'3.0': 745, '1.0': 7032, '2.0': 2531, '5.0': 682, '4.0': 24}
ME_1.csv: {'1.0': 2208, '5.0': 57, '3.0': 25, '2.0': 29, '4.0': 15}
ME_2.csv: {'1.0': 2221, '5.0': 44, '2.0': 25, '4.0': 16, '3.0': 28}
MI_1.csv: {'1.0': 14418, '2.0': 1247, '3.0': 463, '5.0': 433, '4.0': 108}
MI_2.csv: {'1.0': 14414, '4.0': 120, '2.0': 1235, '3.0': 446, '5.0': 454}
MN_0.csv: {'1.0': 9434, '4.0': 100, '2.0': 269, '3.0': 303, '5.0': 225}
MN_1.csv: {'1.0': 9438, '2.0': 272, '5.0': 237, '3.0': 282, '4.0': 101}
MN_2.csv: {'1.0': 9371, '5.0': 241, '4.0': 108, '3.0': 275, '2.0': 335}
MO_0.csv: {'1.0': 9377, '2.0': 712, '4.0': 32, '3.0': 194, '5.0': 240}
MO_1.csv: {'1.0': 9359, '5.0': 235, '2.0': 745, '3.0': 168, '4.0': 48}
MO_2.csv: {'1.0': 9326, '3.0': 189, '2.0': 737, '5.0': 261, '4.0': 41}
MS_0.csv: {'5.0': 83, '1.0': 2895, '3.0': 69, '2.0': 1334, '4.0': 15}
MS_1.csv: {'1.0': 2906, '2.0': 1309, '5.0': 89, '3.0': 64, '4.0': 27}
MT_0.csv: {'1.0': 1676, '5.0': 49, '3.0': 10, '4.0': 83, '2.0': 3}
MT_1.csv: {'1.0': 1664, '4.0': 95, '5.0': 47, '3.0': 11, '2.0': 4}
MT_2.csv: {'1.0': 1675, '4.0': 83, '5.0': 47, '2.0': 7, '3.0': 9}
NC_0.csv: {'1.0': 13021, '4.0': 207, '2.0': 2837, '5.0': 749, '3.0': 542}
NC_1.csv: {'1.0': 12986, '2.0': 2856, '4.0': 212, '3.0': 539, '5.0': 763}
NC_2.csv: {'1.0': 12976, '2.0': 2882, '3.0': 498, '4.0': 220, '5.0': 779}
ND_0.csv: {'1.0': 1346, '3.0': 19, '5.0': 30, '4.0': 60, '2.0': 30}
ND_1.csv: {'1.0': 1358, '4.0': 57, '5.0': 39, '2.0': 18, '3.0': 13}
NE_0.csv: {'1.0': 3287, '2.0': 80, '5.0': 126, '3.0': 68, '4.0': 33}
NE_1.csv: {'1.0': 3276, '5.0': 124, '2.0': 89, '3.0': 68, '4.0': 36}
NE_2.csv: {'1.0': 3296, '4.0': 30, '2.0': 97, '3.0': 58, '5.0': 112}
NH_2.csv: {'1.0': 2494, '3.0': 57, '5.0': 64, '2.0': 34, '4.0': 6}
NJ_1.csv: {'1.0': 11521, '3.0': 1701, '2.0': 1629, '5.0': 1033, '4.0': 43}
NJ_2.csv: {'1.0': 11693, '5.0': 1079, '3.0': 1643, '2.0': 1469, '4.0': 43}
NM_0.csv: {'4.0': 412, '1.0': 2130, '5.0': 247, '3.0': 62, '2.0': 53}
NV_0.csv: {'1.0': 3360, '2.0': 338, '5.0': 704, '3.0': 443, '4.0': 91}
NV_2.csv: {'3.0': 467, '1.0': 3308, '2.0': 356, '5.0': 706, '4.0': 98}
NY_0.csv: {'1.0': 24573, '3.0': 3131, '2.0': 3885, '5.0': 2634, '4.0': 118}
NY_1.csv: {'1.0': 24473, '5.0': 2639, '3.0': 3201, '2.0': 3915, '4.0': 112}
NY_2.csv: {'1.0': 24619, '2.0': 3847, '3.0': 3121, '5.0': 2632, '4.0': 121}
OH_0.csv: {'1.0': 18075, '2.0': 1722, '5.0': 454, '3.0': 423, '4.0': 38}
PA_0.csv: {'1.0': 20192, '2.0': 1304, '3.0': 642, '5.0': 601, '4.0': 31}
PA_2.csv: {'1.0': 20145, '3.0': 648, '5.0': 654, '2.0': 1299, '4.0': 23}
RI_0.csv: {'1.0': 1661, '5.0': 89, '3.0': 66, '2.0': 81, '4.0': 7}
RI_2.csv: {'1.0': 1642, '5.0': 92, '2.0': 99, '4.0': 7, '3.0': 64}
SC_1.csv: {'1.0': 6135, '2.0': 1701, '5.0': 283, '3.0': 142, '4.0': 32}
SC_2.csv: {'1.0': 6166, '2.0': 1684, '5.0': 265, '3.0': 159, '4.0': 19}
SD_1.csv: {'1.0': 1468, '4.0': 107, '2.0': 13, '5.0': 26, '3.0': 19}
SD_2.csv: {'1.0': 1463, '4.0': 113, '5.0': 29, '3.0': 12, '2.0': 16}
TN_0.csv: {'1.0': 9500, '5.0': 311, '2.0': 1303, '3.0': 188, '4.0': 33}
TN_2.csv: {'1.0': 9426, '2.0': 1385, '5.0': 302, '3.0': 202, '4.0': 19}
TX_0.csv: {'1.0': 35347, '3.0': 2305, '2.0': 4145, '5.0': 3248, '4.0': 263}
TX_1.csv: {'1.0': 35140, '5.0': 3326, '2.0': 4259, '3.0': 2308, '4.0': 275}
TX_2.csv: {'2.0': 4125, '1.0': 35348, '5.0': 3275, '3.0': 2289, '4.0': 271}
VA_2.csv: {'1.0': 11327, '2.0': 2198, '5.0': 733, '3.0': 1076, '4.0': 47}
VT_0.csv: {'1.0': 1216, '5.0': 21, '3.0': 11, '2.0': 7, '4.0': 1}
VT_1.csv: {'1.0': 1205, '5.0': 22, '2.0': 10, '3.0': 19}
WA_1.csv: {'1.0': 10515, '5.0': 1091, '3.0': 1106, '2.0': 376, '4.0': 227}
WA_2.csv: {'5.0': 1108, '1.0': 10441, '3.0': 1158, '4.0': 200, '2.0': 407}
WI_0.csv: {'1.0': 10086, '5.0': 269, '2.0': 298, '3.0': 163, '4.0': 81}
WI_1.csv: {'3.0': 182, '1.0': 10129, '4.0': 76, '5.0': 237, '2.0': 273}
WI_2.csv: {'3.0': 182, '1.0': 10110, '2.0': 273, '4.0': 86, '5.0': 245}
WV_0.csv: {'1.0': 2554, '2.0': 90, '5.0': 39, '3.0': 14, '4.0': 3}
WV_2.csv: {'1.0': 2557, '2.0': 78, '3.0': 21, '5.0': 42, '4.0': 1}
WY_0.csv: {'1.0': 935, '3.0': 4, '5.0': 28, '4.0': 46, '2.0': 9}
```

Total RAC1P Proportions Across All Files:
```
1.0: 0.7729
4.0: 0.0085
5.0: 0.0652
3.0: 0.0604
2.0: 0.0929
```

SEX Proportions in Each File:
```
AK_0.csv: {'2.0': 520, '1.0': 635}
AK_1.csv: {'1.0': 644, '2.0': 510}
AK_2.csv: {'2.0': 515, '1.0': 639}
AL_0.csv: {'1.0': 3880, '2.0': 3543}
AL_2.csv: {'1.0': 3919, '2.0': 3503}
AR_0.csv: {'2.0': 2237, '1.0': 2405}
AR_1.csv: {'1.0': 2437, '2.0': 2205}
AR_2.csv: {'2.0': 2288, '1.0': 2353}
AZ_0.csv: {'1.0': 5815, '2.0': 5259}
AZ_1.csv: {'2.0': 5227, '1.0': 5847}
AZ_2.csv: {'1.0': 5828, '2.0': 5245}
CA_1.csv: {'2.0': 30598, '1.0': 34624}
CA_2.csv: {'2.0': 30747, '1.0': 34474}
CO_1.csv: {'2.0': 4954, '1.0': 5481}
CO_2.csv: {'2.0': 4908, '1.0': 5527}
CT_1.csv: {'1.0': 3308, '2.0': 3287}
CT_2.csv: {'1.0': 3354, '2.0': 3241}
DE_0.csv: {'2.0': 793, '1.0': 778}
DE_1.csv: {'1.0': 803, '2.0': 768}
DE_2.csv: {'1.0': 795, '2.0': 775}
FL_0.csv: {'1.0': 17010, '2.0': 15965}
GA_0.csv: {'2.0': 8156, '1.0': 8816}
GA_1.csv: {'2.0': 8248, '1.0': 8724}
GA_2.csv: {'1.0': 8838, '2.0': 8133}
HI_0.csv: {'2.0': 1260, '1.0': 1317}
HI_1.csv: {'1.0': 1375, '2.0': 1202}
HI_2.csv: {'2.0': 1196, '1.0': 1381}
ID_0.csv: {'2.0': 1295, '1.0': 1460}
ID_1.csv: {'1.0': 1520, '2.0': 1235}
ID_2.csv: {'1.0': 1524, '2.0': 1231}
IL_0.csv: {'1.0': 11472, '2.0': 10867}
IL_1.csv: {'1.0': 11464, '2.0': 10875}
IN_0.csv: {'1.0': 6119, '2.0': 5555}
IN_2.csv: {'1.0': 6153, '2.0': 5521}
KS_0.csv: {'1.0': 2777, '2.0': 2492}
KS_2.csv: {'2.0': 2480, '1.0': 2789}
KY_0.csv: {'2.0': 3526, '1.0': 3810}
KY_2.csv: {'2.0': 3501, '1.0': 3834}
LA_1.csv: {'1.0': 3541, '2.0': 3333}
MA_0.csv: {'2.0': 6646, '1.0': 6726}
MD_0.csv: {'2.0': 5553, '1.0': 5461}
MD_1.csv: {'2.0': 5446, '1.0': 5568}
MD_2.csv: {'1.0': 5587, '2.0': 5427}
ME_1.csv: {'1.0': 1230, '2.0': 1104}
ME_2.csv: {'1.0': 1195, '2.0': 1139}
MI_1.csv: {'2.0': 7864, '1.0': 8805}
MI_2.csv: {'2.0': 7959, '1.0': 8710}
MN_0.csv: {'2.0': 4775, '1.0': 5556}
MN_1.csv: {'2.0': 4932, '1.0': 5398}
MN_2.csv: {'1.0': 5437, '2.0': 4893}
MO_0.csv: {'2.0': 5080, '1.0': 5475}
MO_1.csv: {'1.0': 5415, '2.0': 5140}
MO_2.csv: {'2.0': 5056, '1.0': 5498}
MS_0.csv: {'2.0': 2183, '1.0': 2213}
MS_1.csv: {'1.0': 2220, '2.0': 2175}
MT_0.csv: {'1.0': 1007, '2.0': 814}
MT_1.csv: {'1.0': 979, '2.0': 842}
MT_2.csv: {'2.0': 864, '1.0': 957}
NC_0.csv: {'1.0': 8847, '2.0': 8509}
NC_1.csv: {'1.0': 8919, '2.0': 8437}
NC_2.csv: {'2.0': 8374, '1.0': 8981}
ND_0.csv: {'2.0': 656, '1.0': 829}
ND_1.csv: {'1.0': 828, '2.0': 657}
NE_0.csv: {'2.0': 1710, '1.0': 1884}
NE_1.csv: {'1.0': 1905, '2.0': 1688}
NE_2.csv: {'1.0': 1898, '2.0': 1695}
NH_2.csv: {'2.0': 1257, '1.0': 1398}
NJ_1.csv: {'2.0': 7647, '1.0': 8280}
NJ_2.csv: {'2.0': 7633, '1.0': 8294}
NM_0.csv: {'1.0': 1504, '2.0': 1400}
NV_0.csv: {'2.0': 2270, '1.0': 2666}
NV_2.csv: {'1.0': 2578, '2.0': 2357}
NY_0.csv: {'1.0': 17393, '2.0': 16948}
NY_1.csv: {'1.0': 17447, '2.0': 16893}
NY_2.csv: {'2.0': 17002, '1.0': 17338}
OH_0.csv: {'2.0': 10018, '1.0': 10694}
PA_0.csv: {'2.0': 10958, '1.0': 11812}
PA_2.csv: {'1.0': 11855, '2.0': 10914}
RI_0.csv: {'1.0': 955, '2.0': 949}
RI_2.csv: {'1.0': 939, '2.0': 965}
SC_1.csv: {'2.0': 4023, '1.0': 4270}
SC_2.csv: {'1.0': 4127, '2.0': 4166}
SD_1.csv: {'1.0': 833, '2.0': 800}
SD_2.csv: {'1.0': 854, '2.0': 779}
TN_0.csv: {'1.0': 5928, '2.0': 5407}
TN_2.csv: {'1.0': 5846, '2.0': 5488}
TX_0.csv: {'2.0': 21241, '1.0': 24067}
TX_1.csv: {'2.0': 21183, '1.0': 24125}
TX_2.csv: {'1.0': 24037, '2.0': 21271}
VA_2.csv: {'2.0': 7475, '1.0': 7906}
VT_0.csv: {'1.0': 624, '2.0': 632}
VT_1.csv: {'2.0': 608, '1.0': 648}
WA_1.csv: {'1.0': 7145, '2.0': 6170}
WA_2.csv: {'2.0': 6261, '1.0': 7053}
WI_0.csv: {'1.0': 5793, '2.0': 5104}
WI_1.csv: {'2.0': 5240, '1.0': 5657}
WI_2.csv: {'2.0': 5150, '1.0': 5746}
WV_0.csv: {'2.0': 1264, '1.0': 1436}
WV_2.csv: {'2.0': 1313, '1.0': 1386}
WY_0.csv: {'1.0': 519, '2.0': 503}
```

Total SEX Proportions Across All Files:

```
2.0: 0.4790
1.0: 0.5210
```

We refer the reader to the ACS PUMS documentation for the year 2018 for a human-readable meaning of these numerical encodings. 

### Is it possible to identify individuals (i.e., one or more natural persons), either directly or indirectly (i.e., in combination with other data) from the dataset?

_If so, please describe how._

Yes, the ACS PUMS contains potentially identifiable information. Significant attention has been put by the U.S. Census Bureau to prevent re-identification
of individuals and/or households. As of April 2025, we do not know of publicly reported incidents related to re-identification.

### Does the dataset contain data that might be considered sensitive in any way (e.g., data that reveals racial or ethnic origins, sexual orientations, religious beliefs, political opinions or union memberships, or locations; financial or health data; biometric or genetic data; forms of government identification, such as social security numbers; criminal history)?

_If so, please provide a description._

Yes. Please refer to the question about sub-populations.

### Any other comments?

## Collection process

_\[T\]he answers to questions here may provide information that allow others to
reconstruct the dataset without access to it._

### How was the data associated with each instance acquired?

_Was the data directly observable (e.g., raw text, movie ratings), reported by subjects (e.g.,
survey responses), or indirectly inferred/derived from other data (e.g., part-of-speech tags,
model-based guesses for age or language)? If data was reported by subjects or indirectly
inferred/derived from other data, was the data validated/verified? If so, please describe how._

Detailed information about the 2023 ACS Survey techniques is available, as of April 2025, at [this link](https://www2.census.gov/programs-surveys/acs/tech_docs/accuracy/ACS_Accuracy_of_Data_2023.pdf).
Please note that this information might differ to the actual microdata sample employed to obtain the present dataset, which is from 2018.
Detailed information about every year of the ACS Survey is available [here](https://www.census.gov/programs-surveys/acs/technical-documentation/code-lists.html).

### What mechanisms or procedures were used to collect the data (e.g., hardware apparatus or sensor, manual human curation, software program, software API)?

_How were these mechanisms or procedures validated?_

The ACS relies on three modes of data collection:
1. Internet  
2. Mailout/Mailback 
3. Computer Assisted Personal Interview (CAPI)

### If the dataset is a sample from a larger set, what was the sampling strategy (e.g., deterministic, probabilistic with specific sampling probabilities)?

We refer to the documentation of the ACS PUMS available [here](https://www.census.gov/programs-surveys/acs/microdata/documentation.html).

As for the present dataset, FeDa4Fair offers researchers the opportunity to modify the data itself by dropping or removing group/label combinations. The rationale is to test bias mitigation in federated learning settings with disparate distributions of resources and biases". For the present dataset, the state-level data has been modified as follows:

| State        | Drop Rate | Attribute | Value |
|--------------|--------------------|--------------------|----------------|
| AZ           | 0.1                | RAC1P              | 5              |
| OH           | 0.1                | RAC1P              | 4              |
| AR           | 0.2                | RAC1P              | 4              |
| MN           | 0.2                | RAC1P              | 5              |
| OR           | 0.2                | RAC1P              | 2              |
| WV           | 0.2                | RAC1P              | 5              |
| DE           | 0.3                | RAC1P              | 4              |
| LA           | 0.3                | RAC1P              | 5              |
| NE           | 0.3                | RAC1P              | 4              |
| AK           | 0.5                | RAC1P              | 4              |
| MS           | 0.5                | RAC1P              | 4              |
| PR           | 0.6                | RAC1P              | 4              |

The "drop rate" refers to how many data rows with a PINCP value of True and a SEX or RAC1P value of 2 were removed from the data obtained from the ACS PUMS to obtain this specific dataset.

### Who was involved in the data collection process (e.g., students, crowdworkers, contractors) and how were they compensated (e.g., how much were crowdworkers paid)?

The U.S. Census Bureau was involved in the creation of the ACS and the creation and release of the ACS PUMS.

### Over what timeframe was the data collected?

_Does this timeframe match the creation timeframe of the data associated with the instances (e.g.
recent crawl of old news articles)? If not, please describe the timeframe in which the data
associated with the instances was created._
This dataset refers to individuals living in the U.S. in the year 2018.

### Were any ethical review processes conducted (e.g., by an institutional review board)?

_If so, please provide a description of these review processes, including the outcomes, as well as
a link or other access point to any supporting documentation._

Not for FeDa4Fair or Folktables as far as we know.

### Does the dataset relate to people?

_If not, you may skip the remainder of the questions in this section._

Yes.

### Did you collect the data from the individuals in question directly, or obtain it via third parties or other sources (e.g., websites)?

We present data was obtained from the Folktables library and processed again with FeDa4Fair. Folktables sources it from the ACS PUMS.

### Were the individuals in question notified about the data collection?

_If so, please describe (or show with screenshots or other information) how notice was provided,
and provide a link or other access point to, or otherwise reproduce, the exact language of the
notification itself._

Yes, via mail and phone call by the U.S. Census Bureau. The individuals are not re-identifiable and thus it is not
possible to notify them of their inclusion in the present dataset.

### Did the individuals in question consent to the collection and use of their data?

_If so, please describe (or show with screenshots or other information) how consent was
requested and provided, and provide a link or other access point to, or otherwise reproduce, the
exact language to which the individuals consented._

Yes. We were not able to obtain the specific consent request forwarded by the U.S. Census Bureau.

### If consent was obtained, were the consenting individuals provided with a mechanism to revoke their consent in the future or for certain uses?

_If so, please provide a description, as well as a link or other access point to the mechanism
(if appropriate)._

Not that we know of, at the ACS PUMS level.

### Has an analysis of the potential impact of the dataset and its use on data subjects (e.g., a data protection impact analysis) been conducted?

_If so, please provide a description of this analysis, including the outcomes, as well as a link
or other access point to any supporting documentation._

Yes. We refer to the ACS PUMS documentation for an analysis of the privacy measures in place.

### Any other comments?

## Preprocessing/cleaning/labeling

_The questions in this section are intended to provide dataset consumers with the information
they need to determine whether the “raw” data has been processed in ways that are compatible
with their chosen tasks. For example, text that has been converted into a “bag-of-words” is
not suitable for tasks involving word order._

### Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)?

_If so, please provide a description. If not, you may skip the remainder of the questions in
this section._

Please refer to the question about subpopulations.

### Was the “raw” data saved in addition to the preprocessed/cleaned/labeled data (e.g., to support unanticipated future uses)?

_If so, please provide a link or other access point to the “raw” data._

No.

### Is the software used to preprocess/clean/label the instances available?

_If so, please provide a link or other access point._

Yes, this is the [FeDa4Fair](https://github.com/xheilmann/FeDa4Fair) library.

### Any other comments?

## Uses

_These questions are intended to encourage dataset creators to reflect on the tasks
for which the dataset should and should not be used. By explicitly highlighting these tasks,
dataset creators can help dataset consumers to make informed decisions, thereby avoiding
potential risks or harms._

### Has the dataset been used for any tasks already?

_If so, please provide a description._

This dataset is intended for the purpose of training and testing bias mitigation techniques in a federated learning scenario.

### Is there a repository that links to any or all papers or systems that use the dataset?

_If so, please provide a link or other access point._

No.

### What (other) tasks could the dataset be used for?

The dataset could be reasonably employed to test out other techniques in multi-party training such as ones secure multi-party computation.

### Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses?

_For example, is there anything that a future user might need to know to avoid uses that
could result in unfair treatment of individuals or groups (e.g., stereotyping, quality of
service issues) or other undesirable harms (e.g., financial harms, legal risks) If so, please
provide a description. Is there anything a future user could do to mitigate these undesirable
harms?_

It has to be kept in mind that the present dataset has been developed exclusively to test out bias mitigation strategies, especially in federated learning scenarios.

### Are there tasks for which the dataset should not be used?

_If so, please provide a description._

This dataset should not be employed to discuss U.S. demographic trends, as it relies on several manipulations and subsamplings of the ACS data.

We discourage any other use than research and development of machine learning techniques and bias mitigation techniques.

### Any other comments?

## Distribution

### Will the dataset be distributed to third parties outside of the entity (e.g., company, institution, organization) on behalf of which the dataset was created? 

_If so, please provide a description._
This dataset is hosted on github and huggingface.

### How will the dataset will be distributed (e.g., tarball on website, API, GitHub)?

_Does the dataset have a digital object identifier (DOI)?_

Github and Huggingface.

### When will the dataset be distributed?

For the foreseeable future.

### Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)?

_If so, please describe this license and/or ToU, and provide a link or other access point to,
or otherwise reproduce, any relevant licensing terms or ToU, as well as any fees associated
with these restrictions._

No.

### Have any third parties imposed IP-based or other restrictions on the data associated with the instances?

_If so, please describe these restrictions, and provide a link or other access point to, or
otherwise reproduce, any relevant licensing terms, as well as any fees associated with these
restrictions._

No.

### Do any export controls or other regulatory restrictions apply to the dataset or to individual instances?

_If so, please describe these restrictions, and provide a link or other access point to, or otherwise
reproduce, any supporting documentation._

No.

### Any other comments?

## Maintenance

_These questions are intended to encourage dataset creators to plan for dataset maintenance
and communicate this plan with dataset consumers._

### Who is supporting/hosting/maintaining the dataset?

This dataset is maintained and hosted by its authors: Luca Corbucci, Xenia Heilmann, Mattia Cerrato and Anna Monreale.

### How can the owner/curator/manager of the dataset be contacted (e.g., email address)?

You may open an issue on the FeDa4Fair github page.

### Is there an erratum?

_If so, please provide a link or other access point._

No.

### Will the dataset be updated (e.g., to correct labeling errors, add new instances, delete instances)?

We do not plan to.

_If so, please describe how often, by whom, and how updates will be communicated to users (e.g., mailing list, GitHub)?_

### If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances (e.g., were individuals in question told that their data would be retained for a fixed period of time and then deleted)?

_If so, please describe these limits and explain how they will be enforced._

No.

### Will older versions of the dataset continue to be supported/hosted/maintained?

_If so, please describe how. If not, please describe how its obsolescence will be communicated to users._

No.

### If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so?

_If so, please provide a description. Will these contributions be validated/verified? If so,
please describe how. If not, why not? Is there a process for communicating/distributing these
contributions to other users? If so, please provide a description._

FeDa4Fair, the library employed to create this dataset, is hosted on [Github](https://github.com/xheilmann/FeDa4Fair).

### Any other comments?

Not at this time.
