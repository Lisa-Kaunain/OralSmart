# OralSmart AI — Dental Dataset Documentation

## 1. Dataset Overview

* **Dataset Title:** Annotated intraoral image dataset for dental caries detection
* **DOI:** [10.5281/zenodo.14827784](https://doi.org/10.5281/zenodo.14827784)
* **Access URL:** https://zenodo.org/records/14827784
* **License:** Creative Commons Attribution 4.0 International (CC-BY-4.0)
* **Authors:** Syed Muhammad Faizan Ahmed, Huzaifa Ghori, Aamna Khalid, Ayesha Nooruddin, Niha Adnan, Abhishek Lal, Fahad Umer
* **Original Publication:** *Scientific Data* (Nature Portfolio), 2025 (DOI: 10.1038/s41597-025-05647-9)
* **Image Modality:** Ordinary intraoral camera and mobile phone color photographs (NOT dental X-rays or radiographs)

---

## 2. Clinical Scope & Patient Demographics

* **Demographics:** Subjects aged 10–24 years presenting for dental examination.
* **Dentition Stages:** Primary (deciduous/baby teeth), mixed dentition, and permanent (secondary) dentition.
* **Intraoral Perspectives:**
  - Frontal View
  - Left Lateral View
  - Right Lateral View
  - Mandibular View (lower arch occlusal)
  - Maxillary Occlusal View (upper arch occlusal)
* **Real-world Variability:** Variations in clinical lighting, saliva, angle, resolution, and both retractor-assisted and unassisted photography.

---

## 3. Class Definitions & Disease Taxonomy

The dataset clinician annotations are mapped to the internationally recognized WHO Oral Health Surveys standard (DMFT / dmft Index criteria):

| Class ID | Internal Code | Screening Finding Class | Clinical Description |
| :---: | :---: | :--- | :--- |
| **0** | `normal` | `No Visible Dental Caries` | Sound, healthy tooth surface without visible cavitation or carious breakdown. |
| **1** | `D` | `Dental Caries (Permanent Dentition)` | Visible enamel/dentinal cavitation or decay in secondary (permanent) teeth. |
| **2** | `d` | `Dental Caries (Primary Dentition)` | Visible carious lesion or cavitation in primary (deciduous/baby) teeth. |

---

## 4. Dataset Partitioning & Preprocessing

The dataset was partitioned following the benchmark protocol into train, validation, and test splits with zero patient/image leakage across splits:

* **Training Set:** 120 images (36 normal, 54 permanent caries `D`, 30 primary caries `d`).
* **Validation Set:** 29 images (10 normal, 13 permanent caries `D`, 6 primary caries `d`).
* **Held-out Test Set:** 30 images (10 normal, 13 permanent caries `D`, 7 primary caries `d`).

### Data Preparation Steps:
1. **HTTP Range-Request Extraction:** Selective stream downloading directly from the official 1.46 GB Zenodo archive using the remote ZIP Central Directory table (`cd.bin`), bypassing bandwidth bottlenecks.
2. **File Integrity Verification:** Each image opened and verified via Pillow (`Image.verify()`) to reject corrupted or truncated payloads.
3. **Duplicate Detection:** Content-based MD5 hashing across all partitions.
4. **Data Augmentation (Training):** Random horizontal flips ($p=0.5$), subtle rotations ($\pm 15^\circ$), and color jitter (brightness, contrast, saturation) to simulate variable phone camera capture conditions.
5. **Standardization:** Resized to $224 \times 224$ pixels, RGB channel order, normalized using standard ImageNet mean ($[0.485, 0.456, 0.406]$) and standard deviation ($[0.229, 0.224, 0.225]$).

---

## 5. Dataset Limitations

1. **Surface Vision vs. Deep Radiographs:** Intraoral photographs only capture visible coronal surfaces. Interproximal decay (between tight teeth contacts) and subgingival or root decay cannot be diagnosed from photographs alone.
2. **Age Distribution:** The benchmark cohort focuses on pediatric and young adult subjects (ages 10–24). Senior patients with root surface caries, severe attrition, or extensive prosthodontic work are less represented.
3. **Clinical Context:** Photographic screening is strictly preliminary and does not replace in-person dental probing, pulp vitality testing, and bitewing X-rays.
