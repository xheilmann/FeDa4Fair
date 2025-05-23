# Datasheet for dataset FeDa4Fair-cross-device-attribute-bias

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
This dataset was created so that it represents **attribute bias** in a federated learning context, by which we mean that clients have data biased toward different sensitive attributes.
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

The present dataset has 740932 rows across all states represented in the ACS PUMS. We assume here that every state represents a client in a cross-device federated learning task. State data is each in its own CSV, each with the following number of rows:

```
AL_2.csv: 3694
AL_3.csv: 3694
AL_4.csv: 3694
AZ_0.csv: 5546
AZ_5.csv: 5546
CA_1.csv: 32166
CO_0.csv: 5217
CO_3.csv: 5217
CT_2.csv: 3297
CT_5.csv: 3297
FL_5.csv: 16487
HI_0.csv: 1288
IA_0.csv: 2957
IA_1.csv: 2957
ID_0.csv: 1331
ID_2.csv: 1331
IL_2.csv: 10676
IL_3.csv: 10676
IL_4.csv: 10676
IN_0.csv: 5725
IN_2.csv: 5725
IN_3.csv: 5725
KS_2.csv: 2529
KS_4.csv: 2529
LA_2.csv: 3407
LA_4.csv: 3407
MA_0.csv: 6685
MA_3.csv: 6685
MA_4.csv: 6685
ME_1.csv: 1167
MI_0.csv: 7847
MI_1.csv: 7847
MI_2.csv: 7847
MI_3.csv: 7847
MI_4.csv: 7847
MN_0.csv: 5170
MN_3.csv: 5170
MN_4.csv: 5170
MN_5.csv: 5170
MO_1.csv: 4926
MO_4.csv: 4926
MS_4.csv: 2173
MS_5.csv: 2173
NC_1.csv: 8677
ND_1.csv: 734
ND_4.csv: 734
NE_3.csv: 1797
NE_4.csv: 1797
NE_5.csv: 1797
NH_0.csv: 1270
NH_1.csv: 1270
NH_4.csv: 1270
NH_5.csv: 1270
NJ_1.csv: 7963
NJ_3.csv: 7963
NJ_4.csv: 7963
NM_1.csv: 1451
NM_5.csv: 1451
NV_0.csv: 2467
NV_1.csv: 2467
NV_3.csv: 2467
NY_0.csv: 17170
NY_1.csv: 17170
NY_2.csv: 17170
NY_3.csv: 17170
NY_4.csv: 17170
NY_5.csv: 17170
OH_0.csv: 9722
OH_1.csv: 9722
OH_2.csv: 9722
OH_3.csv: 9722
OH_4.csv: 9722
OK_0.csv: 2885
OK_2.csv: 2885
OK_3.csv: 2885
OR_1.csv: 3618
OR_3.csv: 3618
OR_4.csv: 3618
PA_0.csv: 10792
PA_1.csv: 10792
PA_2.csv: 10792
PA_3.csv: 10792
PA_5.csv: 10792
RI_2.csv: 952
SC_0.csv: 4146
SC_1.csv: 4146
SC_2.csv: 4146
SC_4.csv: 4146
SD_3.csv: 816
TN_2.csv: 5360
TN_4.csv: 5360
TN_5.csv: 5360
TX_0.csv: 21454
TX_1.csv: 21454
TX_3.csv: 21454
TX_4.csv: 21454
TX_5.csv: 21454
UT_0.csv: 2722
UT_1.csv: 2722
UT_4.csv: 2722
UT_5.csv: 2722
VA_0.csv: 7035
VA_1.csv: 7035
VA_3.csv: 7035
VA_4.csv: 7035
WA_3.csv: 6342
WA_5.csv: 6342
WI_2.csv: 5440
WI_3.csv: 5440
WV_0.csv: 1301
WV_5.csv: 1301
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

We refer to the ACS PUMS documentation for a human-readable description.


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
AL_2.csv: {'2.0': 862, '1.0': 2832}
AL_3.csv: {'1.0': 2817, '2.0': 877}
AL_4.csv: {'1.0': 2832, '2.0': 862}
AZ_0.csv: {'1.0': 4358, '2.0': 1188}
AZ_5.csv: {'1.0': 4348, '2.0': 1198}
CA_1.csv: {'2.0': 12147, '1.0': 20019}
CO_0.csv: {'2.0': 606, '1.0': 4611}
CO_3.csv: {'2.0': 618, '1.0': 4599}
CT_2.csv: {'1.0': 2655, '2.0': 642}
CT_5.csv: {'1.0': 2728, '2.0': 569}
FL_5.csv: {'1.0': 13299, '2.0': 3188}
HI_0.csv: {'1.0': 359, '2.0': 929}
IA_0.csv: {'1.0': 2800, '2.0': 157}
IA_1.csv: {'1.0': 2794, '2.0': 163}
ID_0.csv: {'1.0': 1227, '2.0': 104}
ID_2.csv: {'2.0': 110, '1.0': 1221}
IL_2.csv: {'1.0': 8639, '2.0': 2037}
IL_3.csv: {'2.0': 1987, '1.0': 8689}
IL_4.csv: {'1.0': 8629, '2.0': 2047}
IN_0.csv: {'1.0': 5049, '2.0': 676}
IN_2.csv: {'1.0': 5089, '2.0': 636}
IN_3.csv: {'1.0': 5107, '2.0': 618}
KS_2.csv: {'1.0': 2244, '2.0': 285}
KS_4.csv: {'2.0': 279, '1.0': 2250}
LA_2.csv: {'1.0': 2443, '2.0': 964}
LA_4.csv: {'1.0': 2414, '2.0': 993}
MA_0.csv: {'1.0': 5468, '2.0': 1217}
MA_3.csv: {'2.0': 1152, '1.0': 5533}
MA_4.csv: {'2.0': 1127, '1.0': 5558}
ME_1.csv: {'1.0': 1116, '2.0': 51}
MI_0.csv: {'1.0': 6800, '2.0': 1047}
MI_1.csv: {'1.0': 6837, '2.0': 1010}
MI_2.csv: {'1.0': 6768, '2.0': 1079}
MI_3.csv: {'1.0': 6716, '2.0': 1131}
MI_4.csv: {'1.0': 6815, '2.0': 1032}
MN_0.csv: {'1.0': 4715, '2.0': 455}
MN_3.csv: {'1.0': 4705, '2.0': 465}
MN_4.csv: {'1.0': 4724, '2.0': 446}
MN_5.csv: {'1.0': 4709, '2.0': 461}
MO_1.csv: {'1.0': 4351, '2.0': 575}
MO_4.csv: {'1.0': 4395, '2.0': 531}
MS_4.csv: {'1.0': 1480, '2.0': 693}
MS_5.csv: {'2.0': 745, '1.0': 1428}
NC_1.csv: {'2.0': 2192, '1.0': 6485}
ND_1.csv: {'1.0': 671, '2.0': 63}
ND_4.csv: {'1.0': 654, '2.0': 80}
NE_3.csv: {'1.0': 1642, '2.0': 155}
NE_4.csv: {'1.0': 1658, '2.0': 139}
NE_5.csv: {'1.0': 1642, '2.0': 155}
NH_0.csv: {'1.0': 1199, '2.0': 71}
NH_1.csv: {'1.0': 1186, '2.0': 84}
NH_4.csv: {'1.0': 1205, '2.0': 65}
NH_5.csv: {'1.0': 1186, '2.0': 84}
NJ_1.csv: {'1.0': 5873, '2.0': 2090}
NJ_3.csv: {'2.0': 2202, '1.0': 5761}
NJ_4.csv: {'1.0': 5845, '2.0': 2118}
NM_1.csv: {'1.0': 1065, '2.0': 386}
NM_5.csv: {'2.0': 373, '1.0': 1078}
NV_0.csv: {'1.0': 1715, '2.0': 752}
NV_1.csv: {'2.0': 824, '1.0': 1643}
NV_3.csv: {'1.0': 1672, '2.0': 795}
NY_0.csv: {'1.0': 12279, '2.0': 4891}
NY_1.csv: {'1.0': 12293, '2.0': 4877}
NY_2.csv: {'1.0': 12166, '2.0': 5004}
NY_3.csv: {'1.0': 12307, '2.0': 4863}
NY_4.csv: {'1.0': 12336, '2.0': 4834}
NY_5.csv: {'2.0': 4887, '1.0': 12283}
OH_0.csv: {'2.0': 1321, '1.0': 8401}
OH_1.csv: {'1.0': 8457, '2.0': 1265}
OH_2.csv: {'1.0': 8422, '2.0': 1300}
OH_3.csv: {'1.0': 8476, '2.0': 1246}
OH_4.csv: {'1.0': 8454, '2.0': 1268}
OK_0.csv: {'2.0': 725, '1.0': 2160}
OK_2.csv: {'1.0': 2125, '2.0': 760}
OK_3.csv: {'1.0': 2171, '2.0': 714}
OR_1.csv: {'1.0': 3144, '2.0': 474}
OR_3.csv: {'1.0': 3152, '2.0': 466}
OR_4.csv: {'1.0': 3166, '2.0': 452}
PA_0.csv: {'1.0': 9549, '2.0': 1243}
PA_1.csv: {'1.0': 9534, '2.0': 1258}
PA_2.csv: {'1.0': 9560, '2.0': 1232}
PA_3.csv: {'1.0': 9595, '2.0': 1197}
PA_5.csv: {'1.0': 9517, '2.0': 1275}
RI_2.csv: {'1.0': 805, '2.0': 147}
SC_0.csv: {'1.0': 3069, '2.0': 1077}
SC_1.csv: {'1.0': 3130, '2.0': 1016}
SC_2.csv: {'1.0': 3074, '2.0': 1072}
SC_4.csv: {'2.0': 1063, '1.0': 3083}
SD_3.csv: {'1.0': 728, '2.0': 88}
TN_2.csv: {'2.0': 919, '1.0': 4441}
TN_4.csv: {'2.0': 865, '1.0': 4495}
TN_5.csv: {'1.0': 4465, '2.0': 895}
TX_0.csv: {'1.0': 16629, '2.0': 4825}
TX_1.csv: {'1.0': 16535, '2.0': 4919}
TX_3.csv: {'1.0': 16781, '2.0': 4673}
TX_4.csv: {'1.0': 16770, '2.0': 4684}
TX_5.csv: {'1.0': 16607, '2.0': 4847}
UT_0.csv: {'2.0': 273, '1.0': 2449}
UT_1.csv: {'1.0': 2418, '2.0': 304}
UT_4.csv: {'2.0': 291, '1.0': 2431}
UT_5.csv: {'2.0': 289, '1.0': 2433}
VA_0.csv: {'2.0': 1819, '1.0': 5216}
VA_1.csv: {'1.0': 5152, '2.0': 1883}
VA_3.csv: {'1.0': 5168, '2.0': 1867}
VA_4.csv: {'1.0': 5142, '2.0': 1893}
WA_3.csv: {'2.0': 1347, '1.0': 4995}
WA_5.csv: {'1.0': 4986, '2.0': 1356}
WI_2.csv: {'1.0': 5057, '2.0': 383}
WI_3.csv: {'1.0': 5052, '2.0': 388}
WV_0.csv: {'1.0': 1227, '2.0': 74}
WV_5.csv: {'1.0': 1236, '2.0': 65}
```

Total RAC1P Proportions Across All Files:
```
2.0: 0.2017
1.0: 0.7983
```

SEX Proportions in Each File:
```
AL_2.csv: {'2.0': 1774, '1.0': 1920}
AL_3.csv: {'2.0': 1751, '1.0': 1943}
AL_4.csv: {'1.0': 1875, '2.0': 1819}
AZ_0.csv: {'1.0': 2853, '2.0': 2693}
AZ_5.csv: {'1.0': 2997, '2.0': 2549}
CA_1.csv: {'2.0': 15287, '1.0': 16879}
CO_0.csv: {'1.0': 2813, '2.0': 2404}
CO_3.csv: {'2.0': 2450, '1.0': 2767}
CT_2.csv: {'1.0': 1666, '2.0': 1631}
CT_5.csv: {'2.0': 1623, '1.0': 1674}
FL_5.csv: {'2.0': 8011, '1.0': 8476}
HI_0.csv: {'2.0': 636, '1.0': 652}
IA_0.csv: {'1.0': 1512, '2.0': 1445}
IA_1.csv: {'2.0': 1395, '1.0': 1562}
ID_0.csv: {'1.0': 744, '2.0': 587}
ID_2.csv: {'2.0': 581, '1.0': 750}
IL_2.csv: {'1.0': 5782, '2.0': 4894}
IL_3.csv: {'2.0': 4880, '1.0': 5796}
IL_4.csv: {'1.0': 5737, '2.0': 4939}
IN_0.csv: {'1.0': 3054, '2.0': 2671}
IN_2.csv: {'2.0': 2684, '1.0': 3041}
IN_3.csv: {'1.0': 3084, '2.0': 2641}
KS_2.csv: {'1.0': 1378, '2.0': 1151}
KS_4.csv: {'2.0': 1150, '1.0': 1379}
LA_2.csv: {'1.0': 1790, '2.0': 1617}
LA_4.csv: {'1.0': 1766, '2.0': 1641}
MA_0.csv: {'2.0': 3342, '1.0': 3343}
MA_3.csv: {'2.0': 3300, '1.0': 3385}
MA_4.csv: {'2.0': 3357, '1.0': 3328}
ME_1.csv: {'1.0': 574, '2.0': 593}
MI_0.csv: {'2.0': 3438, '1.0': 4409}
MI_1.csv: {'2.0': 3516, '1.0': 4331}
MI_2.csv: {'2.0': 3409, '1.0': 4438}
MI_3.csv: {'2.0': 3458, '1.0': 4389}
MI_4.csv: {'2.0': 3532, '1.0': 4315}
MN_0.csv: {'1.0': 2754, '2.0': 2416}
MN_3.csv: {'2.0': 2419, '1.0': 2751}
MN_4.csv: {'1.0': 2719, '2.0': 2451}
MN_5.csv: {'1.0': 2753, '2.0': 2417}
MO_1.csv: {'2.0': 2227, '1.0': 2699}
MO_4.csv: {'2.0': 2149, '1.0': 2777}
MS_4.csv: {'1.0': 1075, '2.0': 1098}
MS_5.csv: {'1.0': 1078, '2.0': 1095}
NC_1.csv: {'1.0': 4388, '2.0': 4289}
ND_1.csv: {'1.0': 413, '2.0': 321}
ND_4.csv: {'1.0': 414, '2.0': 320}
NE_3.csv: {'2.0': 859, '1.0': 938}
NE_4.csv: {'1.0': 962, '2.0': 835}
NE_5.csv: {'1.0': 960, '2.0': 837}
NH_0.csv: {'2.0': 564, '1.0': 706}
NH_1.csv: {'2.0': 583, '1.0': 687}
NH_4.csv: {'1.0': 697, '2.0': 573}
NH_5.csv: {'1.0': 690, '2.0': 580}
NJ_1.csv: {'1.0': 4121, '2.0': 3842}
NJ_3.csv: {'1.0': 4179, '2.0': 3784}
NJ_4.csv: {'1.0': 4141, '2.0': 3822}
NM_1.csv: {'1.0': 767, '2.0': 684}
NM_5.csv: {'2.0': 695, '1.0': 756}
NV_0.csv: {'2.0': 1125, '1.0': 1342}
NV_1.csv: {'2.0': 1144, '1.0': 1323}
NV_3.csv: {'2.0': 1164, '1.0': 1303}
NY_0.csv: {'1.0': 8770, '2.0': 8400}
NY_1.csv: {'1.0': 8623, '2.0': 8547}
NY_2.csv: {'2.0': 8401, '1.0': 8769}
NY_3.csv: {'1.0': 8678, '2.0': 8492}
NY_4.csv: {'2.0': 8535, '1.0': 8635}
NY_5.csv: {'1.0': 8703, '2.0': 8467}
OH_0.csv: {'1.0': 5277, '2.0': 4445}
OH_1.csv: {'2.0': 4352, '1.0': 5370}
OH_2.csv: {'1.0': 5362, '2.0': 4360}
OH_3.csv: {'1.0': 5442, '2.0': 4280}
OH_4.csv: {'1.0': 5419, '2.0': 4303}
OK_0.csv: {'2.0': 1327, '1.0': 1558}
OK_2.csv: {'1.0': 1630, '2.0': 1255}
OK_3.csv: {'1.0': 1569, '2.0': 1316}
OR_1.csv: {'2.0': 1748, '1.0': 1870}
OR_3.csv: {'1.0': 1894, '2.0': 1724}
OR_4.csv: {'2.0': 1686, '1.0': 1932}
PA_0.csv: {'1.0': 5846, '2.0': 4946}
PA_1.csv: {'1.0': 5956, '2.0': 4836}
PA_2.csv: {'2.0': 4856, '1.0': 5936}
PA_3.csv: {'1.0': 5879, '2.0': 4913}
PA_5.csv: {'1.0': 6003, '2.0': 4789}
RI_2.csv: {'2.0': 461, '1.0': 491}
SC_0.csv: {'1.0': 2128, '2.0': 2018}
SC_1.csv: {'2.0': 2057, '1.0': 2089}
SC_2.csv: {'2.0': 2021, '1.0': 2125}
SC_4.csv: {'2.0': 2103, '1.0': 2043}
SD_3.csv: {'2.0': 409, '1.0': 407}
TN_2.csv: {'1.0': 2944, '2.0': 2416}
TN_4.csv: {'2.0': 2439, '1.0': 2921}
TN_5.csv: {'2.0': 2394, '1.0': 2966}
TX_0.csv: {'1.0': 12068, '2.0': 9386}
TX_1.csv: {'1.0': 11989, '2.0': 9465}
TX_3.csv: {'2.0': 9487, '1.0': 11967}
TX_4.csv: {'1.0': 12103, '2.0': 9351}
TX_5.csv: {'1.0': 11997, '2.0': 9457}
UT_0.csv: {'1.0': 1478, '2.0': 1244}
UT_1.csv: {'1.0': 1449, '2.0': 1273}
UT_4.csv: {'2.0': 1262, '1.0': 1460}
UT_5.csv: {'1.0': 1499, '2.0': 1223}
VA_0.csv: {'1.0': 3965, '2.0': 3070}
VA_1.csv: {'1.0': 4012, '2.0': 3023}
VA_3.csv: {'1.0': 3994, '2.0': 3041}
VA_4.csv: {'1.0': 3940, '2.0': 3095}
WA_3.csv: {'1.0': 3573, '2.0': 2769}
WA_5.csv: {'1.0': 3529, '2.0': 2813}
WI_2.csv: {'1.0': 2876, '2.0': 2564}
WI_3.csv: {'2.0': 2524, '1.0': 2916}
WV_0.csv: {'1.0': 714, '2.0': 587}
WV_5.csv: {'1.0': 708, '2.0': 593}
```

Total SEX Proportions Across All Files:
```
2.0: 0.4638
1.0: 0.5362
```

We refer the reader to the ACS PUMS documentation for the year 2018 for a human-readable meaning of these numerical encodings. Compared to the documentation, we note here that we have binarized the RAC1P attribute to represent "white" and "non-white".

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

| State | Drop Rate | Attribute | Value |
|--------------|--------------------|--------------------|----------------|
| WY           | 0.1                | SEX                | 2              |
| WI           | 0.1                | RAC1P              | 2              |
| ND           | 0.1                | SEX                | 2              |
| CA           | 0.1                | RAC1P              | 2              |
| MT           | 0.1                | RAC1P              | 2              |
| LA           | 0.1                | SEX                | 2              |
| KY           | 0.1                | RAC1P              | 2              |
| ME           | 0.1                | RAC1P              | 2              |
| AL           | 0.2                | RAC1P              | 2              |
| IN           | 0.2                | SEX                | 2              |
| MS           | 0.2                | RAC1P              | 2              |
| GA           | 0.2                | RAC1P              | 2              |
| VT           | 0.2                | RAC1P              | 2              |
| IL           | 0.3                | SEX                | 2              |
| WA           | 0.3                | SEX                | 2              |
| NH           | 0.3                | SEX                | 2              |
| PA           | 0.4                | SEX                | 2              |
| WV           | 0.4                | SEX                | 2              |
| AR           | 0.4                | SEX                | 2              |
| KS           | 0.4                | SEX                | 2              |
| OR           | 0.4                | RAC1P              | 2              |
| TX           | 0.4                | SEX                | 2              |
| DE           | 0.4                | RAC1P              | 2              |
| OK           | 0.4                | SEX                | 2              |
| ID           | 0.4                | SEX                | 2              |
| MI           | 0.5                | SEX                | 2              |
| VA           | 0.5                | SEX                | 2              |
| TN           | 0.5                | SEX                | 2              |
| OH           | 0.5                | SEX                | 2              |
| MO           | 0.6                | SEX                | 2              |
| PR           | 0.6                | SEX                | 2              |

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
