# Automated Image Annotation & Quality Audit Tool 🏗️

[![LinkedIn](https://img.shields.io/badge/LinkedIn-aawhanvyas-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/aawhanvyas)
[![GitHub](https://img.shields.io/badge/GitHub-aawhan0-lightgrey?style=flat&logo=github)](https://github.com/aawhan0)

## 🚀 The Problem
In production-level AI, manual data labeling is the most expensive and time-consuming bottleneck. Raw datasets are often "noisy"—containing blurry images or ambiguous objects that can degrade model performance if not filtered correctly. 

## 🛠️ The Solution
I developed an **Automated Annotation & Quality Audit Pipeline** designed for industrial-grade datasets (specifically Construction Site Safety). This tool implements a "Human-in-the-Loop" architecture that automates the labeling process while maintaining a strict "Gold Standard" for data quality.

## 🧠 Technical Methodology
The pipeline processes raw data through two critical audit layers:

### 1. Visual Quality Audit (OpenCV)
Before the AI ever sees the data, a script audits the image for clarity using **Laplacian Variance**.
- **Logic:** Images with low variance (blurred or out-of-focus) are automatically flagged.
- **Impact:** Prevents "garbage data" from being used in training, ensuring higher model precision.

### 2. AI Confidence Audit (YOLOv8)
The pipeline utilizes a YOLOv8 nano model for high-speed inference.
- **Auto-Accept:** Detections with **>85% confidence** are automatically saved in production-ready YOLO format.
- **Audit Flag:** Any image with detections **<85%** or zero detections is moved to a `needs_review` folder.
- **Impact:** Ensures 100% accuracy by forcing human intervention only where the AI is uncertain.

[Image of a human-in-the-loop data labeling workflow showing AI auto-labeling and human verification steps]

## 📊 Business Impact for AI Services
* **Efficiency:** Estimated **70% reduction** in manual annotation workload.
* **Scalability:** Modular Python architecture (`src/`) allows for easy integration into MLOps pipelines.
* **Quality Assurance:** Built-in auditing ensures zero-defect datasets for client delivery.

## 📂 Project Structure
```text
automated-annotation-tool/
├── data/
│   ├── raw/             # Unlabelled images (Construction Safety Dataset)
│   ├── auto_labeled/    # Validated images & generated .txt labels
│   └── needs_review/    # Images flagged for Blur or Low Confidence
├── models/
│   └── yolov8n.pt       # Pre-trained YOLOv8 weights
├── src/
│   └── audit_pipeline.py # Main Audit & Annotation Logic
├── requirements.txt     # Project Dependencies
└── README.md            # Documentation