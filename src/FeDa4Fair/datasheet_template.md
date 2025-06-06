# Datasheet for dataset [tag:name]"add dataset name here"[/tag]

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
FeDa4Fair is built on top of the [ACS PUMS](https://www.census.gov/programs-surveys/acs/microdata.html) subset of the US census data 
obtained via the [folktables](https://github.com/social-foundations/folktables) APIs.
To clarify the relationship between the present dataset and the U.S. census data:
* The American Community Survey (ACS) is a yearly initiative undertaken by the U.S. Census Bureau with the objective to represent demographics and social status in the U.S.
* The ACS Public Use Microdata Sample (ACS PUMS) is a subset of the ACS (about 1 percent) which is released to the general public.
* Folktables is a library that uses API endpoints offered by the Census Bureau to download PUMS data and test it in the space of algorithmic fairness.
* FeDa4Fair is a library that employs Folktables to offer data, divided at the U.S. state level and further, to employ it in the space of federated learning 
and fairness. FeDa4Fair offers some flexibility in the sense that researchers might employ different partitioning of the data beyond the state-level, for instance
to test client-level federated learning techniques.
* The present dataset was obtained with FeDa4Fair, commit id [tag:commit]to be filled...[/tag] and remote [tag:remote][/tag] for the purpose of further testing.

### Who created the dataset (e.g., which team, research group) and on behalf of which entity (e.g., company, institution, organization)?

FeDa4Fair is created and maintained by Xenia Heilmann (JGU Mainz), Luca Corbucci (University of Pisa), Anna Monreale (University of Pisa) and Mattia Cerrato (JGU Mainz).
[tag:who]to be filled[/tag].

### Who funded the creation of the dataset? 

_If there is an associated grant, please provide the name of the grantor and the grant
name and number._
For FeDa4Fair: XH and MC were funded by the “TOPML: Trading Off Non-Functional Properties of Machine Learning” project funded by the
Carl-Zeiss-Stiftung in the Förderprogramm “Durchbrüche”, identifying code P2021-02-014.
LC was funded by The European Union Horizon 2020 program under grant agreement No. 101120763 (TANGO).
AM was funded by the National Recovery and Resilience Plan (PNRR), under agreements: PNRR - M4C2 - Investimento 1.3, Partenariato Esteso PE00000013 - "FAIR - Future Artificial Intelligence Research" - Spoke 1 "Human-centered AI"
For the specific present dataset: [tag:funding]to be filled[/tag]

### Any other comments?

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
The data is partially anonymized. This dataset in particular includes data from [tag:individuals]individuals or households?[/tag] in the year [tag:year]which year?[/tag].

### How many instances are there in total (of each type, if appropriate)?
The present dataset has [tag:nrows]to be filled[/tag].

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

The data column names are as follows: [tag:columns]to be filled[/tag]. 

We refer to the ACS PUMS documentation for a human-readable description.


### Is there a label or target associated with each instance?

_If so, please provide a description._
[tag:income]This dataset is based on the ACSIncome folktables task. As such, the ground truth is whether an individual or household has
earned more than 50 thousand dollars the year prior to the census survey.[/tag]
[tag:employment]This dataset is based on the ACSEmployment folktables task. As such, the ground truth is whether an individual or reference
person in the household was employed at the time of the census survey.[/tag]

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
At the global federated level (that is, with all the data pooled together), the relevant subpopulations are as follows:
[tag:sens_overall][/tag]

Before further data splits, the state-level groups are as follows:
[tag:sens_by_split][/tag]

For each client, the subpopulations are as follows:
[tag:sens_by_partition][/tag]

We refer the reader to the ACS PUMS documentation for the year [tag:year][/tag] for a human-readable meaning of these numerical encodings.

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
Please note that this information might differ to the actual microdata sample employed to obtain the present dataset, which is from [tag:year][/tag].
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

[tag:modification][/tag]

### Who was involved in the data collection process (e.g., students, crowdworkers, contractors) and how were they compensated (e.g., how much were crowdworkers paid)?
The U.S. Census Bureau was involved in the creation of the ACS and the creation and release of the ACS PUMS.

### Over what timeframe was the data collected?

_Does this timeframe match the creation timeframe of the data associated with the instances (e.g.
recent crawl of old news articles)? If not, please describe the timeframe in which the data
associated with the instances was created._
This dataset refers to individuals living in the U.S. in the year [tag:year][/tag].

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
[tag:whodistribution]How will the dataset be distributed?[/tag]

### How will the dataset will be distributed (e.g., tarball on website, API, GitHub)?

_Does the dataset have a digital object identifier (DOI)?_
[tag:howdistribution]And how?[/tag]

### When will the dataset be distributed?
[tag:whendistributed]And when?[/tag]

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
This dataset is maintained and hosted by its authors: [tag:who][/tag]

### How can the owner/curator/manager of the dataset be contacted (e.g., email address)?

### Is there an erratum?

_If so, please provide a link or other access point._
No.

### Will the dataset be updated (e.g., to correct labeling errors, add new instances, delete instances)?

_If so, please describe how often, by whom, and how updates will be communicated to users (e.g., mailing list, GitHub)?_

### If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances (e.g., were individuals in question told that their data would be retained for a fixed period of time and then deleted)?

_If so, please describe these limits and explain how they will be enforced._
No.

### Will older versions of the dataset continue to be supported/hosted/maintained?

_If so, please describe how. If not, please describe how its obsolescence will be communicated to users._

### If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so?

_If so, please provide a description. Will these contributions be validated/verified? If so,
please describe how. If not, why not? Is there a process for communicating/distributing these
contributions to other users? If so, please provide a description._
FeDa4Fair, the library employed to create this dataset, is hosted on [Github](https://github.com/xheilmann/FeDa4Fair).

### Any other comments?
