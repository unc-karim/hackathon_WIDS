# Flood Rescue: Real-Time Human Detection

### Using YOLOv11 with Advanced Training and Evaluation Strategies

---

## Overview

This project focuses on detecting **humans in flood environments** using **YOLOv11**, the latest generation of real-time object detection models.
The main goal is to assist **disaster response teams** in identifying people stranded during floods through automated image analysis.

We enhanced the standard YOLO workflow with improved **data preprocessing**, **augmentation**, **confidence threshold tuning**, and **evaluation metrics visualization**.

---

## Key Features

* YOLOv11-based human detection for flood imagery
* Automatic dataset download and preparation (C2A Flood Dataset)
* Advanced augmentation for flood scenarios
* Comprehensive evaluation metrics (Precision, Recall, mAP@0.5, mAP@0.5:0.95, F1-score)
* Visualizations for training curves and predictions
* Exported model in both PyTorch (`.pt`) and ONNX formats for deployment

---

## Dataset

**Dataset:** [C2A Flood Dataset](https://www.kaggle.com/datasets/rgbnihal/c2a-dataset)

This dataset includes real flood images annotated with human bounding boxes.
We filtered and organized it into:

* **Training set**
* **Validation set**

Each image has a YOLO-style `.txt` annotation file.

---

## Installation

### 1. Clone this repository


git clone https://github.com/yourusername/enhanced-human-detection-flood.git
cd enhanced-human-detection-flood


### 2. Install dependencies

pip install ultralytics kagglehub opencv-python-headless seaborn scikit-learn matplotlib pandas

### 3. Download the dataset

import kagglehub
path = kagglehub.dataset_download("rgbnihal/c2a-dataset")
print(f"Dataset downloaded to: {path}")

---

## Model Training

We trained **YOLOv11s** (small variant) for efficiency and speed on the filtered dataset.

**Training setup:**

* Epochs: 20
* Batch size: 64
* Optimizer: SGD
* Image size: 640
* Augmentations: hue/saturation, scaling, flipping, rotation

Run the training:

from ultralytics import YOLO

model = YOLO('yolo11s.pt')
results = model.train(
    data='flood.yaml',
    epochs=20,
    imgsz=640,
    batch=64
)


---

## Evaluation Metrics

After training, our best model achieved:

| Metric           | Score  |
| ---------------- | ------ |
| **Precision**    | 0.8579 |
| **Recall**       | 0.7905 |
| **mAP@0.5**      | 0.8415 |
| **mAP@0.5:0.95** | 0.5805 |
| **F1-score**     | 0.8234 |

These results show strong detection accuracy, even in complex flood scenarios.

---

## Visual Results

The project includes:

* Training/validation loss curves
* Precision–Recall & F1-score plots
* Bounding box visualizations on validation images
* Confidence threshold vs. performance graphs

Example visualization:

results = model.val(data='flood.yaml')
results.show()


---

## Model Export

The best-performing model is saved in multiple formats for deployment:

| Format     | File                      | Description                           |
| ---------- | ------------------------- | ------------------------------------- |
|  PyTorch | `flood_yolov11_best.pt`   | Fine-tuning and retraining            |
|  ONNX    | `flood_yolov11_best.onnx` | Real-time inference / edge deployment |

Export command:


best_model.export(format='onnx', imgsz=1280, dynamic=True)


---

## Confidence Threshold Analysis

We tested multiple confidence thresholds and found:

* **Optimal confidence:** `0.25`
* **Optimal F1-score:** ~`0.83`

This threshold offers the best trade-off between precision and recall.

---

## Next Steps

* Test on real-world flood videos
* Integrate into drones or surveillance systems
* Extend to detect debris, vehicles, and other hazards
* Deploy using TensorRT or ONNX Runtime for real-time performance

---

## Team & Contributions

**Project Title:** Enhanced Human Detection in Flood Images

**Team Members:**

Karim Abdallah - Hassan Hashem - Hussein Mdaihly - Samer Barakat - Carl Wakim

We also used claude ai, and chatgpt5 for writing code, and brainstorming.
---
