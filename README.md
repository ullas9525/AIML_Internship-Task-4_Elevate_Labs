# Logistic Regression — AIML Internship Task 4 (Elevate Labs)

## 📌 Overview
This task focuses on building a **Logistic Regression-based classification model** to predict whether a tumor is **Benign (0)** or **Malignant (1)** using the Breast Cancer dataset. The goal is to apply a real-world binary classification workflow, evaluate model performance, and interpret outputs using **confusion matrix, probability distribution, and ROC-AUC**.

---

## 🧠 Project Workflow
1. Load dataset and remove irrelevant columns
2. Handle missing values
3. Scale numerical features using StandardScaler
4. Split dataset using stratified train–test split
5. Train Logistic Regression classifier
6. Evaluate model using multiple metrics
7. Generate decision-support visualizations

---

## 📂 Folder Structure
```
AIML_Internship-Task-4_Elevate_Labs
│
├── Dataset
│     └── data.csv
│
├── Output
│     ├── confusion_matrix_heatmap.png
│     ├── roc_curve.png
│     ├── probability_distribution.png
│     ├── model_logistic_regression.joblib
│     └── scaler.joblib
│
└── logistic_regression_task4.py
```

---

## 📌 Model Performance Summary
| Metric            | Value  |
|------------------|--------|
| Accuracy         | 0.9737 |
| Precision        | 0.9756 |
| Recall           | 0.9524 |
| F1-Score         | 0.9639 |
| ROC-AUC Score    | 0.9960 |

---

## 🔍 Visual Outputs (saved in `Output/`)
| File | Insight |
|------|---------|
| `confusion_matrix_heatmap.png` | Shows TP, FP, TN, FN counts |
| `probability_distribution.png` | Probability separation between benign vs malignant |
| `roc_curve.png` | Threshold-independent performance evaluation (AUC score) |

---

## 🛠 Tech Stack
| Component | Technology |
|----------|------------|
| Language | Python |
| Libraries | Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn, Joblib |
| IDE | Visual Studio Code |

---

## 📦 Requirements
Install dependencies before running:
```
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

---

## ▶ Running the Script
Ensure dataset is located inside the **Dataset** folder, then run:
```
python logistic_regression_task4.py
```
All evaluation plots and model files will be generated inside the **Output** folder.

---

## 📌 Outcome
The logistic regression model successfully:
✔ distinguishes malignant and benign cases with high accuracy
✔ provides interpretable probabilities for medical decision support
✔ visualizes performance using ROC, probability density, and confusion matrix

---

## 👤 Author
**Name:** Ullas B R  
**Role:** AIML Internship Participant — Elevate Labs  
**Task 4:** Binary Classification using Logistic Regression

---

## ⭐ Final Note
This project demonstrates **end-to-end deployment-style machine learning**, covering preprocessing, model training, evaluation, threshold tuning, and result visualization — ensuring complete reproducibility for real-world applications.

