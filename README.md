# Bank Loan Default Prediction: Machine Learning Project 🚀

# Quick Summary 🚀

**Objective:** Predict loan defaults using the bank’s loan dataset to enable proactive risk management.

**Tech Stack:** Python (Pandas, Scikit-learn, XGBoost, LightGBM), imbalanced-learn (SMOTE), Matplotlib/Seaborn for visualization.

**Workflow & Techniques:**
- **Data Cleaning & EDA:** Removed duplicates, handled missing values, outlier detection via IQR.
- **Feature Engineering:** Categorical encoding (ordinal & one-hot), log transformation for skewed features, derived temporal features from issue dates.
- **Class Balancing:** Applied **SMOTE** to address target imbalance.
- **Models Tested:**  
  1. Logistic Regression (baseline + SMOTE + PCA) → Accuracy: 0.6658  
  2. Tree-based Models (Random Forest, XGBoost, LightGBM) → Accuracy: 0.9125  
  3. Tree Models + Time-series features → Accuracy: 0.8561
- **Evaluation:** Accuracy, precision, recall, F1-score; feature importance analysis highlighted `int_rate`, `term`, and `loan_amount` as key predictors.

**Key Insight:** Tree-based models significantly outperform linear models, and incorporating temporal features provides actionable insights into loan default risk progression.


## Project Overview 📊

This project builds upon the existing **Bank Loan Report dataset**, leveraging it to develop machine learning models for **predicting loan defaults**. The goal is to identify high-risk loans before they occur, enabling proactive risk management for the bank.  

The project follows a complete **machine learning pipeline**: from data cleaning and preprocessing, to feature engineering, model building, and evaluation, culminating in multiple predictive models with progressively higher accuracy.

---

## Dataset & Target 🎯

- **Dataset**: Derived from the bank’s loan portfolio used in the Bank Loan Report project.
- **Target Variable**: `loan_status`  
  - Encoded for ML as:
    - `Fully Paid` → 0  
    - `Charged Off` → 1  
- **Excluded for initial models**: `Current` loans (used for later forecasting).  
- **Features**:  
  - **Numeric**: `annual_income`, `dti`, `installment`, `int_rate`, `loan_amount`, `total_acc`  
  - **Categorical**: `emp_length`, `grade`, `sub_grade`, `term`, `verification_status`, `home_ownership`, `purpose`  
  - **Dates** (used in advanced forecasting models): `issue_date`, `last_credit_pull_date`, `last_payment_date`, `next_payment_date`  

---

## Data Preprocessing & Feature Engineering 🛠️

1. **Data Cleaning & EDA**:
   - Checked for **duplicates** and removed them.
   - Imputed missing values where necessary (numeric & categorical columns).
   - Examined **class imbalance** in the target (`Fully Paid` vs `Charged Off`).

2. **Encoding Categorical Features**:
   - **Ordinal Encoding**: `emp_length`, `grade`, `sub_grade`, `term`, `verification_status`  
   - **One-Hot Encoding**: `home_ownership`, `purpose`

3. **Scaling & Transformation**:
   - Standardized numeric features for linear models.
   - Applied **log transformation** to skewed features like `loan_amount` and `annual_income`.

4. **Outlier Handling**:
   - Detected and removed extreme outliers using the **IQR method** for numeric columns.

5. **Balancing Classes**:
   - Used **SMOTE** (Synthetic Minority Oversampling Technique) to balance the target classes in training data.

6. **Feature Engineering**:
   - Extracted temporal features from `issue_date` and `last_credit_pull_date` for later time-series modeling (e.g., days between issue and last credit pull).

---

## Machine Learning Models & Workflow 🤖

The project progressively implements multiple models, improving performance by adding/removing features and advanced preprocessing techniques:

1. **Simple Logistic Regression**  
   - Only numeric and categorical features, no SMOTE.  
   - Accuracy: **0.6421**

2. **SMOTE + PCA Logistic Regression**  
   - Applied SMOTE to handle class imbalance.  
   - Dimensionality reduction using **PCA**.  
   - Accuracy: **0.6658**

3. **Tree-Based Models**  
   - Models: **Random Forest, XGBoost, LightGBM**  
   - Used balanced classes, all categorical & numeric features, no PCA.  
   - Accuracy: **0.9125**  
   - Feature importance analysis identified `int_rate`, `term`, and `loan_amount` as top predictors.

4. **Tree Models + Time-Series Features**  
   - Incorporated **temporal features** (e.g., days since last credit pull, month/quarter of issue).  
   - Built on tree-based models.  
   - Accuracy: **0.8561**  
   - Allowed prediction of loan default risk considering the progression of loan lifecycle.

---

## Evaluation Metrics 📊

- **Primary Metric**: Accuracy  
- **Secondary Metrics**: Classification Report (Precision, Recall, F1-score), Confusion Matrix  
- Progressive improvement observed across models due to **SMOTE, PCA, tree-based methods, and temporal features**.

| Model | Features | Technique | Accuracy |
|-------|----------|-----------|---------|
| Logistic Regression | Numeric + Categorical | Baseline | 0.6421 |
| Logistic Regression | SMOTE + PCA | Class balancing + dimensionality reduction | 0.6658 |
| Tree Models | Numeric + Categorical | Random Forest / XGBoost / LightGBM | 0.9125 |
| Tree Models + Time-Series | Numeric + Categorical + Temporal Features | Advanced Tree + Time-based features | 0.8561 |

---

## Key Insights 🔑

- Tree-based models significantly outperform linear models for this dataset.  
- Temporal features can enhance understanding of **loan default risk over time**.  
- Synthetic oversampling (SMOTE) helps logistic regression but has limited impact on high-performing tree models.  
- Features like **interest rate, loan term, and loan amount** are strong predictors of default.

---

## Technical Stack 💻

- **Data Manipulation & EDA**: Python (Pandas, NumPy)  
- **Machine Learning**: Scikit-learn, XGBoost, LightGBM, imbalanced-learn (SMOTE)  
- **Visualization**: Matplotlib, Seaborn, Plotly (for feature importance and trend analysis)  
- **Reporting**: Integrated with original SQL/PBI dashboards for holistic insights  

---

This project demonstrates a **complete ML workflow**, from raw data to actionable predictive insights, highlighting **feature engineering, preprocessing, model tuning, and evaluation**.

