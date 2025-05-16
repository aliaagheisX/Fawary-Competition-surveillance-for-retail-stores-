<div id="top">

<!-- HEADER STYLE: CLASSIC -->
<div align="center">

# Fawry Competition Surveillance for Retail Stores 

<em></em>

<!-- BADGES -->
<img src="https://img.shields.io/github/license/aliaagheisX/Fawary-Competition-surveillance-for-retail-stores-?style=default&logo=opensourceinitiative&logoColor=white&color=0080ff" alt="license">
<img src="https://img.shields.io/github/last-commit/aliaagheisX/Fawary-Competition-surveillance-for-retail-stores-?style=default&logo=git&logoColor=white&color=0080ff" alt="last-commit">
<img src="https://img.shields.io/github/languages/top/aliaagheisX/Fawary-Competition-surveillance-for-retail-stores-?style=default&color=0080ff" alt="repo-top-language">
<img src="https://img.shields.io/github/languages/count/aliaagheisX/Fawary-Competition-surveillance-for-retail-stores-?style=default&color=0080ff" alt="repo-language-count">

<!-- default option, no dependency badges. -->

</div>
<br>

---

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project is a comprehensive surveillance system developed for the Fawary Competition, focusing on retail store environments. It integrates object detection, multi-object tracking, and facial recognition to enhance customer behavior analysis and security monitoring. The system leverages state-of-the-art deep learning models, including YOLO, DETR, Faster R-CNN, and FaceNet, to accurately detect, track, and identify individuals across video frames. Designed with modularity and scalability in mind, this solution supports real-time processing, robust validation pipelines, and performance evaluation to ensure deployment readiness for real-world retail applications.

---

## Project Structure

```sh
└── Fawary-Competition-surveillance-for-retail-stores-/
    ├── LICENSE
    ├── README.md
    ├── eval_notebook.ipynb
    ├── eval_notebook_test.ipynb
    ├── external
    ├── nbs
    │   ├── FaceIdentification.ipynb
    │   ├── ZeroShot.ipynb
    │   └── facenet512.ipynb
    ├── src
    │   ├── DETR
    │   ├── Facenet
    │   ├── FasterRcnn
    │   ├── YOLO
    │   ├── constants.py
    │   ├── dataset.py
    │   └── utils.py
    ├── submission_file.csv
    ├── submission_tracking.ipynb
```

### Project Index
## 📁 __root__
| File                                                    | Summary                                                       |
| ------------------------------------------------------- | ------------------------------------------------------------- |
| [t.ipynb](t.ipynb)                                      | Core notebook for data analysis and visualization.            |
| [submission\_tracking.ipynb](submission_tracking.ipynb) | Prepares competition submission file and shows dataset stats. |
| [LICENSE](LICENSE)                                      | Apache License 2.0 terms and conditions.                      |
| [eval\_notebook\_test.ipynb](eval_notebook_test.ipynb)  | Validates TrackEval with project-specific data.               |
| [test\_gt.txt](test_gt.txt)                             | Ground truth dataset used for model evaluation.               |
| [eval\_notebook.ipynb](eval_notebook.ipynb)             | Assesses model performance and guides optimization.           |

## 📁  __nbs__
| File                                                     | Summary                                                   |
| -------------------------------------------------------- | --------------------------------------------------------- |
| [FaceIdentification.ipynb](nbs/FaceIdentification.ipynb) | Implements face detection and recognition pipeline.       |
| [ZeroShot.ipynb](nbs/ZeroShot.ipynb)                     | Enables rapid prototyping with zero-shot learning models. |
| [facenet512.ipynb](nbs/facenet512.ipynb)                 | Uses FaceNet for accurate face recognition.               |

## 📁 `src`

| File Name                                                                                                   | Summary |
|-------------------------------------------------------------------------------------------------------------|---------|
| [`utils.py`](https://github.com/aliaagheisX/Fawary-Competition-surveillance-for-retail-stores-/blob/master/src/utils.py) | - Extracts sequence metadata and frame data with optional bounding box visualization<br>- Builds Weights & Biases tracking for experiment logging<br>- A utility hub for visual inspection and dataset debugging |
| [`dataset.py`](https://github.com/aliaagheisX/Fawary-Competition-surveillance-for-retail-stores-/blob/master/src/dataset.py) | - Converts CrowdHuman data into a unified, usable format<br>- Prepares dataset dictionaries and lists for downstream tasks<br>- Simplifies dataset preprocessing across experiments |
| [`constants.py`](https://github.com/aliaagheisX/Fawary-Competition-surveillance-for-retail-stores-/blob/master/src/constants.py) | - Centralized definition of paths (dataset, annotations, models)<br>- Simplifies path and environment consistency across codebase |

---

## 📁 `src/DETR`

| File Name                                                                                                   | Summary |
|-------------------------------------------------------------------------------------------------------------|---------|
| [`expirement.ipynb`](https://github.com/aliaagheisX/Fawary-Competition-surveillance-for-retail-stores-/blob/master/src/DETR/expirement.ipynb) | - Jupyter notebook to run and analyze experiments with DETR<br>- Tracks model settings, performance, and outputs<br>- Facilitates quick iterations and ablation tests |
| [`inference.py`](https://github.com/aliaagheisX/Fawary-Competition-surveillance-for-retail-stores-/blob/master/src/DETR/inference.py) | - Loads pretrained DETR model and processes input images<br>- Generates detection results with bounding boxes and confidence scores<br>- Annotates frames and visualizes predictions |

---

## 📁 `src/FasterRcnn`

| File Name                                                                                                   | Summary |
|-------------------------------------------------------------------------------------------------------------|---------|
| [`eval_tracking.py`](https://github.com/aliaagheisX/Fawary-Competition-surveillance-for-retail-stores-/blob/master/src/FasterRcnn/eval_tracking.py) | - Evaluates tracking results by comparing to ground truth<br>- Computes metrics like MOTA and MOTP for tracking performance<br>- Core evaluation script for retail surveillance scenario |
| [`training.ipynb`](https://github.com/aliaagheisX/Fawary-Competition-surveillance-for-retail-stores-/blob/master/src/FasterRcnn/training.ipynb) | - Training workflow notebook for Faster R-CNN<br>- Integrates dataset loading, model initialization, training loop, and loss plots |
| [`data.ipynb`](https://github.com/aliaagheisX/Fawary-Competition-surveillance-for-retail-stores-/blob/master/src/FasterRcnn/data.ipynb) | - Prepares and visualizes dataset samples<br>- Verifies correct path setup and dataset content integrity before training |
| [`eval_tracking_yolo_boosttrack copy.py`](https://github.com/aliaagheisX/Fawary-Competition-surveillance-for-retail-stores-/blob/master/src/FasterRcnn/eval_tracking_yolo_boosttrack%20copy.py) | - Performs object tracking using YOLO + BoostTrack method<br>- Outputs detection and tracking results in CSV and MOT format for analysis |
| [`CustomDataset.py`](https://github.com/aliaagheisX/Fawary-Competition-surveillance-for-retail-stores-/blob/master/src/FasterRcnn/CustomDataset.py) | - Custom PyTorch dataset for Faster R-CNN training<br>- Handles image loading, label parsing, and real-time augmentation |
| [`CrowdHumanDataset.py`](https://github.com/aliaagheisX/Fawary-Competition-surveillance-for-retail-stores-/blob/master/src/FasterRcnn/CrowdHumanDataset.py) | - Dataset class specifically for parsing CrowdHuman annotations<br>- Transforms .odgt files to bounding box format usable by the model |
| [`training.py`](https://github.com/aliaagheisX/Fawary-Competition-surveillance-for-retail-stores-/blob/master/src/FasterRcnn/training.py) | - Standalone training script for Faster R-CNN<br>- Manages data loaders, model architecture, optimization loop, and validation |
| [`eval_tracking_test.py`](https://github.com/aliaagheisX/Fawary-Competition-surveillance-for-retail-stores-/blob/master/src/FasterRcnn/eval_tracking_test.py) | - Runs inference and evaluates object tracking on test data<br>- Outputs prediction files and optionally visualizes frame-by-frame tracking |
| [`tempCodeRunnerFile.py`](https://github.com/aliaagheisX/Fawary-Competition-surveillance-for-retail-stores-/blob/master/src/FasterRcnn/tempCodeRunnerFile.py) | - Temporary scratchpad used for testing code snippets or debugging<br>- Not intended for production or main experiments |
| [`tracking.ipynb`](https://github.com/aliaagheisX/Fawary-Competition-surveillance-for-retail-stores-/blob/master/src/FasterRcnn/tracking.ipynb) | - Demonstrates tracking using the trained Faster R-CNN model<br>- Visualizes tracking performance frame-by-frame in videos |
| [`try_boost_tack.ipynb`](https://github.com/aliaagheisX/Fawary-Competition-surveillance-for-retail-stores-/blob/master/src/FasterRcnn/try_boost_tack.ipynb) | - Exploratory notebook to integrate BoostTrack tracking logic<br>- Tests enhancements to standard Faster R-CNN tracking |
| [`inference.py`](https://github.com/aliaagheisX/Fawary-Competition-surveillance-for-retail-stores-/blob/master/src/FasterRcnn/inference.py) | - Loads trained Faster R-CNN model and performs inference<br>- Applies detection and saves annotated results to disk |


## 📁 `src/Yolo`

| **File Name**                                                                                                                           | **Summary**                                                                                                                                                                                                                                               |
| --------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`train.py`](https://github.com/aliaagheisX/Fawary-Competition-surveillance-for-retail-stores-/blob/master/src/YOLO/train.py)           | Trains a YOLO model on a custom dataset using the Ultralytics library. Configures model architecture, training parameters, and data loading. Runs for 100 epochs with periodic checkpointing and validation. Trained models are saved in the `MODEL_DIR`. |
| [`yolo_utils.py`](https://github.com/aliaagheisX/Fawary-Competition-surveillance-for-retail-stores-/blob/master/src/YOLO/yolo_utils.py) | Converts MOT (Multi-Object Tracking) datasets to YOLO format. Applies image augmentations (flipping, rotation, brightness, contrast, noise) and generates bounding box annotations compatible with YOLO training.                                         |


## 📁 `src/Facenet`

| **File Name**                                                                                                                                                                  | **Summary**                                                                                                                                                        |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [`validation_averaging.ipynb`](https://github.com/aliaagheisX/Fawary-Competition-surveillance-for-retail-stores-/blob/master/src/Facenet/validation_averaging.ipynb)           | Implements validation averaging to boost recognition accuracy by averaging predictions across models. Enhances robustness for security and verification scenarios. |
| [`face_id_dataset.py`](https://github.com/aliaagheisX/Fawary-Competition-surveillance-for-retail-stores-/blob/master/src/Facenet/face_id_dataset.py)                           | Efficiently loads and processes image datasets in batches. Supports image resizing, normalization, and embedding loading for training and validation.              |
| [`validation_eculidean_dist.ipynb`](https://github.com/aliaagheisX/Fawary-Competition-surveillance-for-retail-stores-/blob/master/src/Facenet/validation_eculidean_dist.ipynb) | Validates facial features using Euclidean distances on FaceNet embeddings. Ensures accuracy of feature extraction, crucial for recognition performance.            |
| [`training.ipynb`](https://github.com/aliaagheisX/Fawary-Competition-surveillance-for-retail-stores-/blob/master/src/Facenet/training.ipynb)                                   | Runs training using a triplet loss on labeled face data. Outputs embeddings alongside identity labels for further analysis.                                        |
| [`submission.ipynb`](https://github.com/aliaagheisX/Fawary-Competition-surveillance-for-retail-stores-/blob/master/src/Facenet/submission.ipynb)                               | Generates submission predictions by computing facial feature distances. Leverages preprocessed data and embeddings for model evaluation.                           |
| [`scv_clf.ipynb`](https://github.com/aliaagheisX/Fawary-Competition-surveillance-for-retail-stores-/blob/master/src/Facenet/scv_clf.ipynb)                                     | Implements classification and identification using FaceNet embeddings. Forms the backbone for deploying face recognition in practical scenarios.                   |
| [`validation.ipynb`](https://github.com/aliaagheisX/Fawary-Competition-surveillance-for-retail-stores-/blob/master/src/Facenet/validation.ipynb)                               | Evaluates model performance using precision, recall, and F1-score. Helps identify dataset quality issues and assess recognition accuracy.                          |
| [`face_id_embeddings.py`](https://github.com/aliaagheisX/Fawary-Competition-surveillance-for-retail-stores-/blob/master/src/Facenet/face_id_embeddings.py)                     | Extracts and saves facial embeddings using a pre-trained FaceNet model. Outputs structured CSVs for downstream model training and analysis.                        |
| [`face_id_utils.py`](https://github.com/aliaagheisX/Fawary-Competition-surveillance-for-retail-stores-/blob/master/src/Facenet/face_id_utils.py)                               | Utilities for reading, processing, and averaging facial embeddings. Converts pickled data to structured DataFrames for easy integration.                           |
| [`kmean.ipynb`](https://github.com/aliaagheisX/Fawary-Competition-surveillance-for-retail-stores-/blob/master/src/Facenet/kmean.ipynb)                                         | Applies K-means clustering to facial embeddings for grouping similar faces. Useful for both identification and anomaly detection.                                  |




---

## Getting Started

### Prerequisites

This project requires the following dependencies:

- **Programming Language:** Python

### Installation

Build Fawary-Competition-surveillance-for-retail-stores- from the source and intsall dependencies:

1. **Clone the repository:**

    ```sh
     git clone https://github.com/aliaagheisX/Fawary-Competition-surveillance-for-retail-stores-
    ```

2. **Navigate to the project directory:**

    ```sh
    cd Fawary-Competition-surveillance-for-retail-stores-
    ```

3. **Install the dependencies:**
	```sh
	pip install -r requirments.txt
	```

---


## Contributing

<p align="left">
   <a href="https://github.com{/aliaagheisX/Fawary-Competition-surveillance-for-retail-stores-/}graphs/contributors">
      <img src="https://contrib.rocks/image?repo=aliaagheisX/Fawary-Competition-surveillance-for-retail-stores-">
   </a>
</p>

---

## License

Fawary-competition-surveillance-for-retail-stores- is protected under the [LICENSE](https://choosealicense.com/licenses) License. For more details, refer to the [LICENSE](https://choosealicense.com/licenses/) file.


