**Dataset:** [Insert Dataset Link Here](YOUR_DATASET_LINK)

# YOLOv8 Animal Object Detection Pipeline

## Overview
This repository provides a complete, end-to-end pipeline for training, evaluating, and visualizing a custom **YOLOv8** object detection model. It was specifically built to train on a 54-class animal detection dataset. 

The project addresses common computer vision challenges such as class imbalance (via data augmentation) and provides modular scripts to easily configure training, run inference on videos, and generate beautiful evaluation metrics.

---

## 🛠️ How the Project Works

The pipeline is divided into independent, modular scripts orchestrated by a central Jupyter Notebook (`main.ipynb`).

### 1. Data Preparation & Augmentation (`prepare_data.py` & `Augmentation.ipynb`)
- **Structure Validation:** Checks the `data.yaml` file and ensures that all images in the `train`, `valid`, and `test` directories have matching label `.txt` files.
- **Class Balancing:** Analyzes the distribution of the 54 classes. It applies undersampling to majority classes and bounding-box-safe augmentations (like blurring, flipping, and contrast adjustments using `albumentations`) to minority classes to prevent model bias.

### 2. Model Training (`train.py`)
- A robust Python wrapper around the Ultralytics YOLO CLI.
- Dynamically configures hyperparameters (`epochs`, `batch size`, `imgsz`, `patience`, `workers`) and automatically logs the training parameters to a `params.json` file.
- Outputs logs directly to the terminal while saving them to an experiment folder (e.g., `experiments/yolov8_animal_detection`).

### 3. Inference & Evaluation (`inference.py`)
- Loads the best trained weights (`best.pt`).
- Runs prediction on source files (images, webcam, or video files like `vid_test.mp4`).
- Computes detailed evaluation metrics (Precision, Recall, mAP50, mAP50-95) for both the overall model and individual classes, saving the output to a CSV file for reporting.

### 4. Visualization (`visualize.py`)
- Reads the evaluation CSV and utilizes `matplotlib` and `seaborn` to plot the results.
- Generates professional bar charts showing the mAP@.50 and mAP@.50-.95 metrics per class, helping to easily identify which animal classes the model struggles with.
<img width="2190" height="1769" alt="3bd0c3f7-a3b5-4851-b994-8da9af44195f" src="https://github.com/user-attachments/assets/447aab11-e594-471a-a159-c46ea79300de" />

---

## 🚀 Getting Started

### Prerequisites
Make sure you have Python 3.8+ installed. Install the required dependencies:
```bash
pip install ultralytics albumentations pandas matplotlib seaborn opencv-python pyyaml tqdm
