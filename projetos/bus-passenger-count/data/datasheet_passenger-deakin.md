# Datasheet for Dataset: Passenger (Deakin), Roboflow

## Motivation
- **For what purpose was the dataset created? Was there a specific task in mind? Was there a specific gap that needed to be filled?**  
  Created for passenger object detection, likely in public transport environments.
- **Who created the dataset and on behalf of which entity?**  
  Created by the Deakin workspace (`deakin-07shj`) on Roboflow Universe (Deakin University, Australia).
- **Who funded the creation of the dataset?**  
  Unknown (likely internal academic project).
- **Any other comments?**  
  Public dataset on Roboflow.

## Composition
- **What do the instances represent?**  
  Images containing passengers (primarily inside vehicles/public transport).
- **How many instances are there in total?**  
  4,181 images (.jpg). Approximately 100 images will be used after subsampling (1 every 40 frames) for testing.
- **Does the dataset contain all possible instances or is it a sample? If a sample, is it representative? How was representativeness validated?**  
  Sample derived from video footage.
- **What does each instance consist of? Raw data or features? Please describe.**  
  Raw .jpg images with bounding box annotations.
- **Is there a label or target associated with each instance?**  
  Yes, class "passenger".
- **Is any information missing from individual instances?**  
  No.
- **Are relationships between individual instances made explicit?**  
  No (though originally from video sequences).
- **Are there recommended data splits? (e.g., train/validation/test)**  
  Yes, there are: Train: 70% Valid: 20% Test: 10%.
- **Are there any errors, sources of noise, or redundancies?**  
  Yes, there are (possible video-frame redundancy before subsampling).
- **Is the dataset self-contained, or does it rely on external resources? If external: are there guarantees they will persist? Are there archival versions? Any restrictions?**  
  Self-contained on Roboflow.
- **Does the dataset contain data that might be considered confidential?**  
  No.
- **Does the dataset contain data that might be offensive, insulting, threatening, or anxiety-inducing?**  
  No.

**The following questions apply only if the dataset relates to people:**
- **Does the dataset identify any subpopulations?**  
  No.
- **Is it possible to identify individuals, directly or indirectly, from the dataset?**  
  Faces may be visible, but no metadata.
- **Does the dataset contain data that might be considered sensitive?**  
  No.
- **Any other comments?**  
  N/A.

## Collection Process
- **How was the data associated with each instance acquired?**  
  Extracted from video footage and manually annotated.
- **What mechanisms or procedures were used to collect the data? How were these validated?**  
  Roboflow platform for annotation.
- **If the dataset is a sample, what was the sampling strategy?**  
  Video frame extraction.
- **Who was involved in the data collection process and how were they compensated?**  
  Unknown.
- **Over what timeframe was the data collected?**  
  Updated approximately 2 years ago(2024) on Roboflow.
- **Were any ethical review processes conducted?**  
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
  The data is Creative Commons(CC BY 4.0)
- **Any other comments?**  
  N/A.

## Preprocessing / Cleaning / Labeling
- **Was any preprocessing, cleaning, or labeling done? If so, please describe.**  
  Yes, bounding box annotations for "passenger". Subsampling (1 every 40 frames) will be applied in this project.
- **Was the raw data saved in addition to the processed data?**  
  Yes.
- **Is the software used to preprocess/clean/label the data available?**  
  Roboflow tools.
- **Any other comments?**  
  N/A.

## Uses
- **Has the dataset been used for any tasks already?**  
  Yes, passenger detection models on Roboflow.
- **Is there a repository linking to papers or systems that use the dataset?**  
  Roboflow Universe page.
- **What (other) tasks could the dataset be used for?**  
  Public transport monitoring, passenger counting, domain adaptation.
- **Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses?**  
  Video-derived nature may introduce frame similarity if not subsampled.
- **Are there tasks for which the dataset should not be used?**  
  Demographic analysis or sensitive applications.
- **Any other comments?**  
  N/A.

## Distribution
- **Will the dataset be distributed to third parties?**  
  Yes (already public).
- **How will the dataset be distributed? Does it have a DOI?**  
  Via Roboflow: https://universe.roboflow.com/deakin-07shj/passenger-mmpbi (no DOI).
- **When will the dataset be distributed?**  
  Already available.
- **Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)?**  
  CC BY 4.0.
- **Have any third parties imposed IP-based or other restrictions on the data associated with the instances?**  
  No.
- **Do any export controls or other regulatory restrictions apply to the dataset or to individual instances?**  
  No.
- **Any other comments?**  
  N/A.

## Maintenance
- **Who will be supporting/hosting/maintaining the dataset?**  
  Roboflow(the Deakin workspace), also for the Google Drive: Maintenance and updates will be managed by the project authors during the IA901 course period, until the end of July 2026.
- **How can the owner/curator/manager of the dataset be contacted (e.g., email address)?**  
  Through the Roboflow platform.
- **Is there an erratum? If so, please provide a link or other access point.**  
  No.
- **Will the dataset be updated (e.g., to correct labeling errors, add new instances, delete instances)? If so, please describe how often, by whom, and how updates will be communicated to dataset consumers (e.g., mailing list, GitHub).**  
  No.
- **If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances?**  
  N/A.
- **Will older versions of the dataset continue to be supported/hosted/maintained? If so, please describe how. If not, please describe how its obsolescence will be communicated to dataset consumers.**  
  Yes, all data will be on Google Drive or Roboflow.
- **If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so? If so, please provide a description. Will these contributions be validated/verified? If so, please describe how. If not, why not? Is there a process for communicating/distributing these contributions to dataset consumers? If so, please provide a description.**  
  Yes, via Roboflow fork/clone.
- **Any other comments?**  
  N/A.
