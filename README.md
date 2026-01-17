# Automated Image Annotation & Quality Audit Tool

[![LinkedIn](https://img.shields.io/badge/LinkedIn-aawhanvyas-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/aawhanvyas)
[![GitHub](https://img.shields.io/badge/GitHub-aawhan0-lightgrey?style=flat&logo=github)](https://github.com/aawhan0)
[![Live Demo](https://img.shields.io/badge/Streamlit-Live_Demo-FF4B4B?style=flat&logo=streamlit)](https://auto-annotation-aawhan0.streamlit.app/)

## 🚀 The Problem
In production level AI, manual data labeling is the most expensive and time consuming bottleneck. Raw datasets are often noisy; they frequently contain blurry images or ambiguous objects that can degrade model performance if not filtered correctly.

## 🛠️ The Solution
I developed an **Automated Annotation & Quality Audit Pipeline** designed for industrial grade datasets (specifically Construction Site Safety). This tool implements a "Human-in-the-Loop" architecture that automates the labeling process while maintaining a strict "Gold Standard" for data quality.

### 🔗 [Click Here for the Live Demo](https://auto-annotation-aawhan0.streamlit.app/)

## 🧠 Technical Methodology
The pipeline processes raw data through two critical audit layers:

### 1. Visual Quality Audit (OpenCV)
Before the AI ever sees the data, a script audits the image for clarity using **Laplacian Variance**.
* **Logic:** Images with low variance (blurred or out-of-focus) are automatically flagged.
* **Impact:** Prevents "garbage data" from being used in training, ensuring higher model precision.

### 2. AI Confidence Audit (YOLOv8)
The pipeline utilizes a YOLOv8 nano model for high-speed inference.
* **Auto-Accept:** Detections with **>85% confidence** are automatically saved in production-ready YOLO format.
* **Audit Flag:** Any image with detections **<85%** or zero detections is moved to a `needs_review` folder.

<p align="center">
  <img src="assets/ui_screenshot.png" width="800" alt="UI Screenshot">
</p>

## 📊 Business Impact
* **Efficiency:** Estimated **70% reduction** in manual annotation workload.
* **Scalability:** Modular Python architecture allows for easy integration into MLOps pipelines.
* **Quality Assurance:** Built-in auditing ensures zero-defect datasets for client delivery.

## 📈 Current Results
Based on the Construction Site Safety dataset (v30), the pipeline achieved the following distribution:
* **Total Processed:** 717 images
* **Auto-Accepted:** ~70% (High Confidence + High Clarity)
* **Flagged for Review:** ~30% (Low Confidence/Ambiguity/Blur)

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
```
## ⚙️ Setup & Usage
1. Clone the repository
Open your terminal and run the following command to download the project:

```bash
git clone https://github.com/aawhan0/Auto-Annotation.git
cd Auto-Annotation
```
2. Install dependencies
Ensure you have Python installed, then install the required libraries:

```bash
pip install -r requirements.txt
```
3. Run the pipeline
Place your raw images in data/raw/ and execute the audit script:

```bash
python src/audit_pipeline.py
```

## 🔗 Resources & References
* **Dataset Source:** [Construction Site Safety (Roboflow)](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety)
* **Core Model:** [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
* **Image Processing:** [OpenCV Laplacian Variance for Blur Detection](https://docs.opencv.org/4.x/d5/db5/tutorial_laplace_operator.html)
* **Data-Centric AI:** [Andrew Ng's Data-Centric AI Resource Hub](https://datacentricai.org/)

---
**Aawhan Vyas** | [LinkedIn](https://www.linkedin.com/in/aawhanvyas) | [GitHub](https://github.com/aawhan0)
