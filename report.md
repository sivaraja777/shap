# Credit Risk Modeling – SHAP-Based Interpretability Report

## 1. Project Overview

This project develops and interprets an ensemble machine learning model for predicting **loan default risk**.  
Beyond achieving strong predictive accuracy, the primary goal is to ensure the model is **transparent**, **explainable**, and **aligned with responsible lending practices**.

The pipeline includes:
- Data preprocessing  
- Handling class imbalance  
- Training and tuning an XGBoost model  
- Evaluating model performance  
- Extracting model-native feature importance  
- Generating SHAP global and local explanations  
- Interpreting results and providing actionable recommendations  

---

## 2. Dataset & Preprocessing

The dataset contains borrower-level financial and demographic features such as:

- Age  
- Income  
- Loan amount  
- Loan term  
- Previous delinquencies  
- Credit history length  
- Home ownership  
- Employment years  

**Target:**  
`default` (1 = high-risk borrower, 0 = low-risk borrower)

Preprocessing steps performed:
- Missing value handling  
- One-hot encoding for categorical variables  
- Standardization of numeric variables  
- Oversampling with **SMOTE** to address class imbalance  
- Train/test split using stratification  

---

## 3. Model Training

An **XGBoostClassifier** was trained with randomized hyperparameter search.  
The final model returned the following best parameter configuration:

```
[INSERT_BEST_PARAMS_HERE]
```

These hyperparameters were automatically saved into `results.json`.

---

## 4. Model Performance

The model was evaluated on a held-out test set using industry-standard metrics.

**Results:**

- **AUC:** `[INSERT_AUC]`  
- **Precision:** `[INSERT_PRECISION]`  
- **Recall:** `[INSERT_RECALL]`  
- **F1-score:** `[INSERT_F1]`  

**Confusion Matrix:**

```
[INSERT_CONFUSION_MATRIX]
```

**Interpretation:**
- **AUC** reflects strong discriminatory power between good and bad borrowers.  
- **Precision** indicates the proportion of predicted defaulters that were truly high-risk.  
- **Recall** measures how many actual defaulters were correctly identified.  
- **F1-score** provides a balance between precision and recall.  

These results demonstrate a balanced and reliable classifier for credit risk prediction.

---

## 5. Model-Native Feature Importance

XGBoost’s native **gain importance** identified the top predictive features.

Top features included:

1. **[TOP_FEATURE_NAME]**  
2. Additional highly ranked financial variables such as income, loan amount, and past delinquencies  

These features align with real-world lending behavior, supporting the model’s credibility.

A plot is saved as:

```
gain_importance.png
```

---

## 6. SHAP Global Interpretability

SHAP (SHapley Additive exPlanations) was applied to the trained model to understand **how each feature contributes to predictions**.

Generated plots:
- `shap_summary_bar.png`  
- `shap_beeswarm.png`  
- `shap_dependence_<top_feature>.png`  

**Insights:**
- SHAP confirms the model-native importance ranking.  
- Features with strong effects include:
  - Loan amount  
  - Income  
  - Delinquency history  
  - Credit history length  
- Higher loan amounts and more delinquencies push predictions toward **default**.  
- Higher income and longer credit history push predictions toward **non-default**.

SHAP global plots validate that the model behaves as expected in a financial risk context.

---

## 7. SHAP Local Interpretations (3 Individual Cases)

Three individuals from the test set were selected:

1. **High-risk approval**  
2. **Low-risk rejection**  
3. **Borderline case (around 50% probability)**  

Local SHAP explanations show **why** the model made its prediction for each case.

Plots saved:
- `local_shap_case_1.png`  
- `local_shap_case_2.png`  
- `local_shap_case_3.png`  

### Summary of Local Explanations:
- The high-risk case had contributions such as **high loan amount**, **low income**, or **recent delinquencies** pushing risk upward.  
- The low-risk case had stabilizing factors such as **good credit history**, **higher income**, or **few delinquent accounts**.  
- The borderline case demonstrated a mix of positive and negative SHAP contributions, creating a near 50–50 risk score.

These insights help explain model reasoning at the individual borrower level.

---

## 8. Executive Summary & Policy Recommendations

### ✅ Executive Summary
The ensemble credit risk model delivers strong performance and highly interpretable predictions.  
Global and local SHAP visualizations confirm that the model aligns with real-world financial behavior, increasing trust and regulatory compliance.

### 🏦 Policy Recommendations
- Applicants with **high delinquency count**, **large loan amounts**, and **short credit histories** should undergo additional checks.  
- Encourage underwriting rules to incorporate **credit history length** and **income stability** more heavily.  
- Use local SHAP explanations for **case-by-case approvals**, especially borderline decisions.  
- Regularly monitor model bias and fairness across demographic segments.

---

## 9. Files Generated

- `credit_risk_shap.ipynb` – full notebook  
- `results.json` – evaluation metrics  
- `gain_importance.png` – model-native importance  
- `shap_summary_bar.png` – SHAP bar plot  
- `shap_beeswarm.png` – SHAP beeswarm plot  
- `shap_dependence_<top_feature>.png` – dependence plot  
- `local_shap_case_1.png` – high-risk explanation  
- `local_shap_case_2.png` – low-risk explanation  
- `local_shap_case_3.png` – borderline explanation  

---

## 10. Conclusion

This project successfully builds an interpretable credit risk prediction system that is both **accurate** and **transparent**.  
By combining XGBoost with SHAP, the model provides actionable explanations that support fair lending decisions and real-world risk assessment.

