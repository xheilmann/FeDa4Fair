# Datasheet for dataset FeDa4Fair-cross-silo-attribute-bias

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

The present dataset has 1626658 rows across all states represented in the ACS PUMS. We assume here that every state represents a client in a cross-silo federated learning task. State data is each in its own CSV, each with the following number of rows:

```
AK_0.csv: 3546
AL_0.csv: 22167
AR_0.csv: 13416
AZ_0.csv: 33277
CA_0.csv: 192997
CO_0.csv: 31306
CT_0.csv: 19785
DE_0.csv: 4576
FL_0.csv: 98925
GA_0.csv: 50473
HI_0.csv: 7731
IA_0.csv: 17745
ID_0.csv: 7990
IL_0.csv: 64060
IN_0.csv: 34352
KS_0.csv: 15177
KY_0.csv: 21964
LA_0.csv: 20443
MA_0.csv: 40114
MD_0.csv: 33042
ME_0.csv: 7002
MI_0.csv: 47083
MN_0.csv: 31021
MO_0.csv: 29558
MS_0.csv: 13043
MT_0.csv: 5463
NC_0.csv: 52067
ND_0.csv: 4409
NE_0.csv: 10785
NH_0.csv: 7624
NJ_0.csv: 47781
NM_0.csv: 8711
NV_0.csv: 14807
NY_0.csv: 103021
OH_0.csv: 58333
OK_0.csv: 17310
OR_0.csv: 21712
PA_0.csv: 64754
PR_0.csv: 8866
RI_0.csv: 5712
SC_0.csv: 24879
SD_0.csv: 4899
TN_0.csv: 32162
TX_0.csv: 128727
UT_0.csv: 16337
VA_0.csv: 42210
VT_0.csv: 3761
WA_0.csv: 38056
WI_0.csv: 32641
WV_0.csv: 7807
WY_0.csv: 3031
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

RAC1P Proportions in Each CSV file/client:
```
AK_0.csv: {'1.0': 2119, '2.0': 1427}
AL_0.csv: {'1.0': 16966, '2.0': 5201}
AR_0.csv: {'1.0': 11180, '2.0': 2236}
AZ_0.csv: {'1.0': 26192, '2.0': 7085}
CA_0.csv: {'1.0': 121006, '2.0': 71991}
CO_0.csv: {'2.0': 3717, '1.0': 27589}
CT_0.csv: {'1.0': 16159, '2.0': 3626}
DE_0.csv: {'1.0': 3609, '2.0': 967}
FL_0.csv: {'1.0': 79215, '2.0': 19710}
GA_0.csv: {'2.0': 16032, '1.0': 34441}
HI_0.csv: {'1.0': 2070, '2.0': 5661}
IA_0.csv: {'1.0': 16792, '2.0': 953}
ID_0.csv: {'1.0': 7343, '2.0': 647}
IL_0.csv: {'1.0': 51852, '2.0': 12208}
IN_0.csv: {'1.0': 30479, '2.0': 3873}
KS_0.csv: {'1.0': 13426, '2.0': 1751}
KY_0.csv: {'1.0': 19868, '2.0': 2096}
LA_0.csv: {'1.0': 14471, '2.0': 5972}
MA_0.csv: {'1.0': 33004, '2.0': 7110}
MD_0.csv: {'1.0': 20999, '2.0': 12043}
ME_0.csv: {'1.0': 6657, '2.0': 345}
MI_0.csv: {'1.0': 40738, '2.0': 6345}
MN_0.csv: {'1.0': 28243, '2.0': 2778}
MO_0.csv: {'1.0': 26197, '2.0': 3361}
MS_0.csv: {'2.0': 4341, '1.0': 8702}
MT_0.csv: {'1.0': 5015, '2.0': 448}
NC_0.csv: {'1.0': 38983, '2.0': 13084}
ND_0.csv: {'1.0': 4008, '2.0': 401}
NE_0.csv: {'1.0': 9859, '2.0': 926}
NH_0.csv: {'1.0': 7183, '2.0': 441}
NJ_0.csv: {'2.0': 12798, '1.0': 34983}
NM_0.csv: {'2.0': 2339, '1.0': 6372}
NV_0.csv: {'1.0': 10011, '2.0': 4796}
NY_0.csv: {'1.0': 73665, '2.0': 29356}
OH_0.csv: {'2.0': 7689, '1.0': 50644}
OK_0.csv: {'2.0': 4428, '1.0': 12882}
OR_0.csv: {'1.0': 18938, '2.0': 2774}
PA_0.csv: {'1.0': 57347, '2.0': 7407}
PR_0.csv: {'1.0': 5886, '2.0': 2980}
RI_0.csv: {'1.0': 4922, '2.0': 790}
SC_0.csv: {'1.0': 18501, '2.0': 6378}
SD_0.csv: {'1.0': 4403, '2.0': 496}
TN_0.csv: {'1.0': 26860, '2.0': 5302}
TX_0.csv: {'1.0': 100047, '2.0': 28680}
UT_0.csv: {'2.0': 1716, '1.0': 14621}
VA_0.csv: {'2.0': 11188, '1.0': 31022}
VT_0.csv: {'1.0': 3622, '2.0': 139}
WA_0.csv: {'1.0': 29788, '2.0': 8268}
WI_0.csv: {'1.0': 30325, '2.0': 2316}
WV_0.csv: {'1.0': 7377, '2.0': 430}
WY_0.csv: {'1.0': 2765, '2.0': 266}
```

Total RAC1P Proportions Across All states/clients:
```
1.0: 0.7803
2.0: 0.2197
```

SEX Proportions in Each state/client:
```
AK_0.csv: {'2.0': 1592, '1.0': 1954}
AL_0.csv: {'2.0': 10603, '1.0': 11564}
AR_0.csv: {'1.0': 7198, '2.0': 6218}
AZ_0.csv: {'1.0': 17525, '2.0': 15752}
CA_0.csv: {'1.0': 101804, '2.0': 91193}
CO_0.csv: {'1.0': 16579, '2.0': 14727}
CT_0.csv: {'1.0': 10044, '2.0': 9741}
DE_0.csv: {'1.0': 2311, '2.0': 2265}
FL_0.csv: {'1.0': 50822, '2.0': 48103}
GA_0.csv: {'1.0': 26154, '2.0': 24319}
HI_0.csv: {'2.0': 3658, '1.0': 4073}
IA_0.csv: {'1.0': 9351, '2.0': 8394}
ID_0.csv: {'1.0': 4504, '2.0': 3486}
IL_0.csv: {'1.0': 34596, '2.0': 29464}
IN_0.csv: {'1.0': 18364, '2.0': 15988}
KS_0.csv: {'1.0': 8413, '2.0': 6764}
KY_0.csv: {'2.0': 10509, '1.0': 11455}
LA_0.csv: {'2.0': 9966, '1.0': 10477}
MA_0.csv: {'2.0': 19901, '1.0': 20213}
MD_0.csv: {'2.0': 16426, '1.0': 16616}
ME_0.csv: {'2.0': 3429, '1.0': 3573}
MI_0.csv: {'2.0': 20827, '1.0': 26256}
MN_0.csv: {'1.0': 16413, '2.0': 14608}
MO_0.csv: {'1.0': 16388, '2.0': 13170}
MS_0.csv: {'2.0': 6486, '1.0': 6557}
MT_0.csv: {'1.0': 2943, '2.0': 2520}
NC_0.csv: {'1.0': 26747, '2.0': 25320}
ND_0.csv: {'1.0': 2470, '2.0': 1939}
NE_0.csv: {'2.0': 5094, '1.0': 5691}
NH_0.csv: {'2.0': 3499, '1.0': 4125}
NJ_0.csv: {'2.0': 22952, '1.0': 24829}
NM_0.csv: {'1.0': 4512, '2.0': 4199}
NV_0.csv: {'2.0': 6999, '1.0': 7808}
NY_0.csv: {'1.0': 52178, '2.0': 50843}
OH_0.csv: {'1.0': 32219, '2.0': 26114}
OK_0.csv: {'2.0': 7828, '1.0': 9482}
OR_0.csv: {'1.0': 11337, '2.0': 10375}
PA_0.csv: {'1.0': 35480, '2.0': 29274}
PR_0.csv: {'1.0': 4848, '2.0': 4018}
RI_0.csv: {'1.0': 2851, '2.0': 2861}
SC_0.csv: {'1.0': 12614, '2.0': 12265}
SD_0.csv: {'1.0': 2539, '2.0': 2360}
TN_0.csv: {'1.0': 17632, '2.0': 14530}
TX_0.csv: {'1.0': 72229, '2.0': 56498}
UT_0.csv: {'1.0': 8816, '2.0': 7521}
VA_0.csv: {'1.0': 23834, '2.0': 18376}
VT_0.csv: {'2.0': 1835, '1.0': 1926}
WA_0.csv: {'1.0': 21213, '2.0': 16843}
WI_0.csv: {'1.0': 17164, '2.0': 15477}
WV_0.csv: {'1.0': 4285, '2.0': 3522}
WY_0.csv: {'2.0': 1402, '1.0': 1629}
```

Total SEX Proportions Across All Clients:
```
2.0: 0.4685
1.0: 0.5315
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
