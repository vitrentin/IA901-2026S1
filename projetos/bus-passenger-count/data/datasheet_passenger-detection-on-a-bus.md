# Datasheet for Dataset: Passenger Detection on a Bus (Roboflow)

## Motivation
- **For what purpose was the dataset created? Was there a specific task in mind? Was there a specific gap that needed to be filled?**  
  The dataset was created for passenger detection inside buses using object detection models.
- **Who created the dataset and on behalf of which entity?**  
  Created by the user/workspace `bus-project-frdgz` (Bus Project) on Roboflow Universe.
- **Who funded the creation of the dataset?**  
  Unknown.
- **Any other comments?**  
  Public dataset hosted on Roboflow Universe.

## Composition
- **What do the instances represent? (e.g., documents, photos, people, countries)**  
  Photos (images) of bus interiors with passengers.
- **How many instances are there in total?**  
  170 images.
- **Does the dataset contain all possible instances or is it a sample? If a sample, is it representative? How was representativeness validated?**  
  It is a sample focused on the bus interior scenario. Representativeness is high for the target domain according to the project needs.
- **What does each instance consist of? Raw data or features? Please describe.**  
  Raw .jpg images with associated bounding box annotations.
- **Is there a label or target associated with each instance?**  
  Yes, bounding boxes for the class "passenger".
- **Is any information missing from individual instances?**  
  No.
- **Are relationships between individual instances made explicit?**  
  No.
- **Are there recommended data splits? (e.g., train/validation/test)**  
  Unknown (official splits not provided).
- **Are there any errors, sources of noise, or redundancies?**  
  Unknown.
- **Is the dataset self-contained, or does it rely on external resources? If external: are there guarantees they will persist? Are there archival versions? Any restrictions?**  
  Self-contained and hosted on Roboflow.
- **Does the dataset contain data that might be considered confidential?**  
  No.
- **Does the dataset contain data that might be offensive, insulting, threatening, or anxiety-inducing?**  
  No.

**The following questions apply only if the dataset relates to people:**
- **Does the dataset identify any subpopulations? (e.g., by age, gender)**  
  No.
- **Is it possible to identify individuals, directly or indirectly, from the dataset?**  
  Faces may be visible in some images, but no personal metadata is provided.
- **Does the dataset contain data that might be considered sensitive? (e.g., race, sexual orientation, religion, health, financial, biometric, government ID, criminal history)**  
  No.
- **Any other comments?**  
  N/A.

## Collection Process
- **How was the data associated with each instance acquired? Was it directly observable, reported by subjects, or inferred/derived? Was it validated?**  
  Images were collected (likely from video footage) and manually annotated.
- **What mechanisms or procedures were used to collect the data? How were these validated?**  
  Collected and annotated using the Roboflow platform.
- **If the dataset is a sample, what was the sampling strategy?**  
  Unknown.
- **Who was involved in the data collection process and how were they compensated?**  
  Unknown.
- **Over what timeframe was the data collected? Does this match the creation timeframe of the data?**  
  Unknown.
- **Were any ethical review processes conducted? (e.g., IRB) If so, what were the outcomes?**  
  Unknown.

**The following questions apply only if the dataset relates to people:**
- **Was data collected from individuals directly or via third parties?**  
  Via third parties / public collection (unknown details).
- **Were individuals notified about the data collection?**  
  Unknown.
- **Did individuals consent to collection and use of their data?**  
  Unknown.
- **If consent was obtained, can individuals revoke it?**  
  Unknown.
- **Has a data protection impact analysis been conducted?**  
  Unknown.
- **Any other comments?**  
  N/A.

## Preprocessing / Cleaning / Labeling
- **Was any preprocessing, cleaning, or labeling done? If so, please describe.**  
  Yes. Manual bounding box annotation for the "passenger" class.
- **Was the raw data saved in addition to the processed data?**  
  Yes.
- **Is the software used to preprocess/clean/label the data available?**  
  Roboflow annotation tools (platform-dependent).
- **Any other comments?**  
  Annotations are described as clean and suitable for the target task.

## Uses
- **Has the dataset been used for any tasks already?**  
  Yes, for training object detection models on Roboflow.
- **Is there a repository linking to papers or systems that use the dataset?**  
  Available on the Roboflow Universe page.
- **What (other) tasks could the dataset be used for?**  
  Passenger counting, bus occupancy monitoring, domain adaptation for public transport vision systems.
- **Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses? Could it lead to unfair treatment of individuals or groups? How could risks be mitigated?**  
  Small dataset size may lead to overfitting if used alone. No evident bias for unfair treatment.
- **Are there tasks for which the dataset should not be used?**  
  Tasks requiring large-scale training or demographic analysis.
- **Any other comments?**  
  N/A.

## Distribution
- **Will the dataset be distributed to third parties?**  
  Yes, it is already publicly available.
- **How will the dataset be distributed? Does it have a DOI?**  
  Via Roboflow Universe: https://universe.roboflow.com/bus-project-frdgz/passenger-detection-on-a-bus-qgljh (no DOI).
- **When will the dataset be distributed?**  
  Already available.
- **Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)?**  
  Public Domain (CC0).
- **Have any third parties imposed IP-based or other restrictions on the data associated with the instances?**  
  No.
- **Do any export controls or other regulatory restrictions apply to the dataset or to individual instances?**  
  No.
- **Any other comments?**  
  N/A.

## Maintenance
- **Who will be supporting/hosting/maintaining the dataset?**  
  Roboflow and the original creator (`bus-project-frdgz`).
- **How can the owner/curator/manager of the dataset be contacted (e.g., email address)?**  
  Through the Roboflow platform.
- **Is there an erratum? If so, please provide a link or other access point.**  
  Unknown.
- **Will the dataset be updated (e.g., to correct labeling errors, add new instances, delete instances)? If so, please describe how often, by whom, and how updates will be communicated to dataset consumers (e.g., mailing list, GitHub).**  
  Unknown.
- **If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances?**  
  N/A.
- **Will older versions of the dataset continue to be supported/hosted/maintained? If so, please describe how. If not, please describe how its obsolescence will be communicated to dataset consumers.**  
  Unknown.
- **If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so? If so, please provide a description. Will these contributions be validated/verified? If so, please describe how. If not, why not? Is there a process for communicating/distributing these contributions to dataset consumers? If so, please provide a description.**  
  Possible via Roboflow fork/clone features.
- **Any other comments?**  
  N/A.