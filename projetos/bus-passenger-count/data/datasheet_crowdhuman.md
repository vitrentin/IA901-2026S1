# Datasheet for Dataset: CrowdHuman

## Motivation
- **For what purpose was the dataset created? Was there a specific task in mind? Was there a specific gap that needed to be filled?**  
  Created as a benchmark dataset to better evaluate human detectors in highly crowded scenarios with heavy occlusions.
- **Who created the dataset and on behalf of which entity?**  
  Created by Shuai Shao, Zijian Zhao, Boxun Li, Tete Xiao, Gang Yu, Xiangyu Zhang, and Jian Sun (primarily Megvii Inc.).
- **Who funded the creation of the dataset?**  
  Megvii Technology (Megvii Inc.) and collaborating institutions.
- **Any other comments?**  
  Highly influential academic benchmark published in 2018 (arXiv:1805.00123).

## Composition
- **What do the instances represent? (e.g., documents, photos, people, countries)**  
  Images containing multiple people in crowded real-world scenes.
- **How many instances are there in total?**  
  15,000 (train) + 4,370 (validation) + 5,000 (test) = 24,370 images. Approximately 470K human instances in train+val.
- **Does the dataset contain all possible instances or is it a sample? If a sample, is it representative? How was representativeness validated?**  
  Curated benchmark dataset designed to be diverse and representative of crowded scenes.
- **What does each instance consist of? Raw data or features? Please describe.**  
  Raw .jpg images with rich bounding box annotations (full-body, visible-region, and head boxes).
- **Is there a label or target associated with each instance?**  
  Yes, human instances with three types of bounding boxes per person.
- **Is any information missing from individual instances?**  
  No.
- **Are relationships between individual instances made explicit?**  
  No.
- **Are there recommended data splits? (e.g., train/validation/test)**  
  Yes, official train/validation/test splits provided.
- **Are there any errors, sources of noise, or redundancies?**  
  Annotations are high-quality; occlusions are intentionally present as part of the challenge.
- **Is the dataset self-contained, or does it rely on external resources? If external: are there guarantees they will persist? Are there archival versions? Any restrictions?**  
  Self-contained. Available from official website.
- **Does the dataset contain data that might be considered confidential?**  
  No.
- **Does the dataset contain data that might be offensive, insulting, threatening, or anxiety-inducing?**  
  No.

**The following questions apply only if the dataset relates to people:**
- **Does the dataset identify any subpopulations? (e.g., by age, gender)**  
  No explicit subpopulation labels.
- **Is it possible to identify individuals, directly or indirectly, from the dataset?**  
  Faces may be visible in some images, but no personal metadata is attached.
- **Does the dataset contain data that might be considered sensitive? (e.g., race, sexual orientation, religion, health, financial, biometric, government ID, criminal history)**  
  No explicit sensitive attributes.
- **Any other comments?**  
  N/A.

## Collection Process
- **How was the data associated with each instance acquired? Was it directly observable, reported by subjects, or inferred/derived? Was it validated?**  
  Images collected from the internet and manually annotated.
- **What mechanisms or procedures were used to collect the data? How were these validated?**  
  Manual annotation with rigorous protocols for handling occlusions.
- **If the dataset is a sample, what was the sampling strategy?**  
  Curated selection for diversity and density of crowds.
- **Who was involved in the data collection process and how were they compensated?**  
  Unknown (professional annotators likely used by Megvii).
- **Over what timeframe was the data collected? Does this match the creation timeframe of the data?**  
  Prior to 2018.
- **Were any ethical review processes conducted? (e.g., IRB) If so, what were the outcomes?**  
  Unknown (standard for 2018 academic dataset).

**The following questions apply only if the dataset relates to people:**
- **Was data collected from individuals directly or via third parties?**  
  Via third parties (images sourced from the internet).
- **Were individuals notified about the data collection?**  
  Unknown.
- **Did individuals consent to collection and use of their data?**  
  Unknown (web-sourced images).
- **If consent was obtained, can individuals revoke it?**  
  Unknown.
- **Has a data protection impact analysis been conducted?**  
  Unknown.
- **Any other comments?**  
  N/A.

## Preprocessing / Cleaning / Labeling
- **Was any preprocessing, cleaning, or labeling done? If so, please describe.**  
  Yes, extensive manual labeling of full-body, visible-region, and head bounding boxes.
- **Was the raw data saved in addition to the processed data?**  
  Yes.
- **Is the software used to preprocess/clean/label the data available?**  
  Not publicly released.
- **Any other comments?**  
  Annotations are considered high-quality and rigorous.

## Uses
- **Has the dataset been used for any tasks already?**  
  Yes, widely used for human detection, especially in crowded scenes. Cited in hundreds of papers.
- **Is there a repository linking to papers or systems that use the dataset?**  
  Official website: https://www.crowdhuman.org/
- **What (other) tasks could the dataset be used for?**  
  Pedestrian detection, occlusion handling, crowded scene understanding, multi-person tracking.
- **Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses? Could it lead to unfair treatment of individuals or groups? How could risks be mitigated?**  
  Web-sourced images may contain societal biases. Best used for detection robustness rather than fairness-critical applications.
- **Are there tasks for which the dataset should not be used?**  
  Tasks involving demographic profiling or sensitive attribute inference.
- **Any other comments?**  
  N/A.

## Distribution
- **Will the dataset be distributed to third parties?**  
  Yes, publicly available for research.
- **How will the dataset be distributed? Does it have a DOI?**  
  Download from official site https://www.crowdhuman.org/ (no DOI).
- **When will the dataset be distributed?**  
  Available since 2018.
- **Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)?**  
  Research-only / non-commercial use. Images cannot be redistributed.
- **Have any third parties imposed IP-based or other restrictions on the data associated with the instances?**  
  Yes, Megvii imposes strict terms (no image redistribution).
- **Do any export controls or other regulatory restrictions apply to the dataset or to individual instances?**  
  No.
- **Any other comments?**  
  Users must agree to terms before downloading.

## Maintenance
- **Who will be supporting/hosting/maintaining the dataset?**  
  Megvii Inc. (original creators).
- **How can the owner/curator/manager of the dataset be contacted (e.g., email address)?**  
  Via the official website https://www.crowdhuman.org/
- **Is there an erratum? If so, please provide a link or other access point.**  
  No known erratum.
- **Will the dataset be updated (e.g., to correct labeling errors, add new instances, delete instances)? If so, please describe how often, by whom, and how updates will be communicated to dataset consumers (e.g., mailing list, GitHub).**  
  Unlikely (stable benchmark dataset).
- **If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances?**  
  N/A.
- **Will older versions of the dataset continue to be supported/hosted/maintained? If so, please describe how. If not, please describe how its obsolescence will be communicated to dataset consumers.**  
  Yes (original version remains available).
- **If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so? If so, please provide a description. Will these contributions be validated/verified? If so, please describe how. If not, why not? Is there a process for communicating/distributing these contributions to dataset consumers? If so, please provide a description.**  
  No official mechanism.
- **Any other comments?**  
  N/A.
