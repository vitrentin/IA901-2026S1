# Datasheet for Dataset: Inside Bus View (Roboflow)

## Motivation
- **For what purpose was the dataset created? Was there a specific task in mind? Was there a specific gap that needed to be filled?**  
  Seat occupancy detection inside buses (occupied vs unoccupied seats).
- **Who created the dataset and on behalf of which entity?**  
  Created by the `seat-occupancy` workspace on Roboflow Universe.
- **Who funded the creation of the dataset?**  
  Unknown (likely internal academic project).
- **Any other comments?**  
  Public dataset on Roboflow.

## Composition
- **What do the instances represent?**  
  Images of bus seat views.
- **How many instances are there in total?**  
  1,400 images (.jpg).
- **Does the dataset contain all possible instances or is it a sample? If a sample, is it representative? How was representativeness validated?**  
  Sample focused on bus interior seat views.
- **What does each instance consist of? Raw data or features? Please describe.**  
  Raw .jpg images with bounding box annotations.
- **Is there a label or target associated with each instance?**  
  Yes, classes "occupied" and "unoccupied". The "occupied" class bounds the passenger.
- **Is any information missing from individual instances?**  
  No.
- **Are relationships between individual instances made explicit?**  
  No.
- **Are there recommended data splits? (e.g., train/validation/test)**  
  Yes, there are: Train: 70% Valid: 20% Test: 10%.
- **Are there any errors, sources of noise, or redundancies?**  
  No, there are not.
- **Is the dataset self-contained, or does it rely on external resources? If external: are there guarantees they will persist? Are there archival versions? Any restrictions?**  
  The dataset is self-contained.
- **Does the dataset contain data that might be considered confidential?**  
  No.
- **Does the dataset contain data that might be offensive, insulting, threatening, or anxiety-inducing?**  
  No.

**The following questions apply only if the dataset relates to people:**
- **Does the dataset identify any subpopulations? (e.g., by age, gender)**  
  No.
- **Is it possible to identify individuals, directly or indirectly, from the dataset?**  
  Faces may be visible, no metadata.
- **Does the dataset contain data that might be considered sensitive? (e.g., race, sexual orientation, religion, health, financial, biometric, government ID, criminal history)**  
  No.
- **Any other comments?**  
  N/A.

## Collection Process
- **How was the data associated with each instance acquired? Was it directly observable, reported by subjects, or inferred/derived? Was it validated?**  
  Collected from bus video/images and annotated.
- **What mechanisms or procedures were used to collect the data? How were these validated?**  
  Roboflow platform.
- **If the dataset is a sample, what was the sampling strategy?**  
  Unknown.
- **Who was involved in the data collection process and how were they compensated?**  
  Unknown.
- **Over what timeframe was the data collected? Does this match the creation timeframe of the data?**  
  Updated approximately 1 year ago(2025) on Roboflow.
- **Were any ethical review processes conducted? (e.g., IRB) If so, what were the outcomes?**  
  Unknown.

**The following questions apply only if the dataset relates to people:**
- **Was data collected from individuals directly or via third parties?**  
  Unknown.
- **Were individuals notified about the data collection?**  
  Unknown.
- **Did individuals consent to collection and use of their data?**  
  Unknown.
- **If consent was obtained, can individuals revoke it?**  
  Unknown.
- **Has a data protection impact analysis been conducted?**  
  Unknown/not conducted by this project; license: Public Domain (CC0).
- **Any other comments?**  
  N/A.

## Preprocessing / Cleaning / Labeling
- **Was any preprocessing, cleaning, or labeling done? If so, please describe.**  
  Yes, bounding boxes for seat occupancy. In this project "occupied" will be remapped to "person" and "unoccupied" discarded.
- **Was the raw data saved in addition to the processed data?**  
  Yes.
- **Is the software used to preprocess/clean/label the data available?**  
  Roboflow.
- **Any other comments?**  
  N/A.

## Uses
- **Has the dataset been used for any tasks already?**  
  Yes, seat occupancy detection.
- **Is there a repository linking to papers or systems that use the dataset?**  
  Roboflow page.
- **What (other) tasks could the dataset be used for?**  
  Passenger detection, bus monitoring.
- **Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses? Could it lead to unfair treatment of individuals or groups? How could risks be mitigated?**  
  Seat-centric view may limit general person detection use.
- **Are there tasks for which the dataset should not be used?**  
  Demographic analysis or sensitive attribute prediction.
- **Any other comments?**  
  N/A.

## Distribution
- **Will the dataset be distributed to third parties?**  
  Yes (already public).
- **How will the dataset be distributed? Does it have a DOI?**  
  Roboflow: https://universe.roboflow.com/seat-occupancy/inside-bus-view
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
  Roboflow and the original `seat-occupancy` workspace, also for the Google Drive: Maintenance and updates will be managed by the project authors during the IA901 course period, until the end of July 2026.
- **How can the owner/curator/manager of the dataset be contacted (e.g., email address)?**  
  Via Roboflow platform.
- **Is there an erratum? If so, please provide a link or other access point.**  
  No.
- **Will the dataset be updated (e.g., to correct labeling errors, add new instances, delete instances)? If so, please describe how often, by whom, and how updates will be communicated to dataset consumers (e.g., mailing list, GitHub).**  
  No.
- **If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances?**  
  N/A.
- **Will older versions of the dataset continue to be supported/hosted/maintained? If so, please describe how. If not, please describe how its obsolescence will be communicated to dataset consumers.**  
  Yes, all data will be on Google Drive or Roboflow.
- **If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so? If so, please provide a description. Will these contributions be validated/verified? If so, please describe how. If not, why not? Is there a process for communicating/distributing these contributions to dataset consumers? If so, please provide a description.**  
  Yes, via Roboflow.
- **Any other comments?**  
  N/A.