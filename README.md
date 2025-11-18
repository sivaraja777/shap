# Interpretable Machine Learning for Credit Risk Modeling using SHAP Values

This project builds, evaluates, and interprets an ensemble machine learning model for **binary credit default prediction**.  
The focus is not only on achieving strong predictive performance but also on providing **clear, data-driven explanations** using **SHAP (SHapley Additive exPlanations)**.  
This ensures transparency and aligns with real-world regulatory requirements such as *Right to Explanation*, GDPR, and fair lending practices.

---

## 📌 Project Overview

The goal of this project is to develop an interpretable credit risk model using XGBoost or LightGBM.  
The workflow includes:

1. **Preprocessing** of a credit dataset  
2. **Handling categorical variables** (one-hot or target encoding)  
3. **Addressing class imbalance** using resampling  
4. **Training and tuning** an ensemble model  
5. **Evaluating performance** (AUC, Precision, Recall)  
6. **Extracting model-native feature importance**  
7. **Generating SHAP global explanations**  
8. **Producing SHAP local explanations** for three selected loan applicants  
9. **Writing an executive summary with policy recommendations**

The notebook produces all required plots and metrics needed for model interpretation and reporting.

---

## 📂 Project Structure

```
project-root/
│
├── credit_risk_shap.ipynb          # Main notebook with full pipeline & outputs
├── results.json                    # Stores evaluation metrics & model parameters
├── requirements.txt                # Python dependencies
│
├── dataset/
│   └── credit_risk.csv             # Credit dataset (real or synthetic)
│
└── plots/
    ├── gain_importance.png
    ├── shap_summary_bar.png
    ├── shap_beeswarm.png
    ├── shap_dependence_<top_feature>.png
    ├── local_shap_case_1.png
    ├── local_shap_case_2.png
    └── local_shap_case_3.png
```

---

## 🧠 Key Methods & Techniques

### 🔹 1. Preprocessing & Class Imbalance
- Handling of categorical features  
- Standardization of numerical variables  
- Oversampling using **SMOTE**  

### 🔹 2. Ensemble Model Training
Models used:
- **XGBoostClassifier** (primary)
- LightGBM (alternative option)

Evaluation metrics:
- **AUC**
- **Precision**
- **Recall**
- **F1-score**
- Confusion Matrix

### 🔹 3. Feature Importance (Native)
- Extracted from XGBoost using **gain importance**
- Sorted and visualized as `gain_importance.png`

### 🔹 4. SHAP Global Explanations
- **SHAP summary bar plot**
- **Beeswarm plot**
- **Dependence plot** for the top contributing feature

These plots explain the *overall* drivers of credit risk in the model.

### 🔹 5. SHAP Local Explanations
Three individuals from the test set are analyzed:

1. **High-risk approval**
2. **Low-risk rejection**
3. **Borderline case**

For each case, a **local SHAP waterfall or force plot** is saved:
```
local_shap_case_1.png  
local_shap_case_2.png  
local_shap_case_3.png
```

These explain *why* the model made each prediction.

---

## 📊 Evaluation Output

All evaluation metrics and model settings are saved in:
```
results.json
```

This includes:
- Best hyperparameters  
- AUC, Precision, Recall, F1  
- Confusion matrix  

---

## 🚀 How to Run the Project

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Run the notebook
Open and run all cells in:

```
credit_risk_shap.ipynb
```

It will:
- Load dataset  
- Train model  
- Generate plots  
- Export evaluation metrics  
- Save SHAP local/global explanations  

### 3️⃣ Check generated outputs
All plots will be stored in:
```
plots/
```

Metrics go to:
```
results.json
```

---

## 📝 Executive Summary (High-Level)

- The ensemble model achieves competitive AUC and balanced precision/recall.  
- SHAP reveals the 3–5 most critical features influencing loan risk.  
- Local explanations provide clear justifications for individual approvals/rejections.  
- Recommendations are based on transparent, interpretable insights.

A full interpretation is provided in `report.md`.

---

## 👤 Author

**Saravanan R**  
GitHub: https://github.com/Saravanan2104

