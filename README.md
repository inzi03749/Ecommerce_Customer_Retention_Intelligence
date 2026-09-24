# E-Commerce Customer Retention Intelligence System

An end-to-end machine learning and analytics project for identifying customers at risk of churn and supporting data-driven retention planning.

The project combines data cleaning, exploratory data analysis, business insight generation, machine learning, threshold optimization, leakage-sensitivity analysis, and an interactive Streamlit dashboard into a single reproducible workflow.

---

## 1. Project Overview

Customer retention is an important business problem in e-commerce because identifying customers who may churn can help organizations prioritize retention efforts.

This project develops a complete customer retention intelligence workflow using an e-commerce customer dataset.

The solution:

- Loads and validates the raw customer dataset.
- Performs data cleaning and exploratory data analysis.
- Generates business-oriented insights related to customer behavior and churn.
- Builds and evaluates multiple machine learning models.
- Uses cross-validation and out-of-fold predictions for model evaluation and threshold selection.
- Applies preprocessing inside scikit-learn pipelines to reduce data leakage risk.
- Performs leakage-sensitivity analysis using selected potentially leakage-prone features.
- Provides an interactive Streamlit dashboard.
- Generates customer-level risk scores and retention-plan recommendations.

The final implementation is consolidated into a single Python file:

```text
Inzimamul_Haq_Ecommerce_Customer_Retention_Intelligence.py
```

---

## 2. Project Objectives

The main objectives of the project are:

1. Understand the structure and quality of the customer dataset.
2. Identify patterns associated with customer churn.
3. Build machine learning models capable of identifying customers at risk of churn.
4. Compare Logistic Regression, Random Forest, and Gradient Boosting.
5. Select an operating classification threshold using training-set out-of-fold predictions.
6. Evaluate the selected model on a held-out test set.
7. Examine sensitivity to potentially leakage-prone features.
8. Present the results through an interactive business dashboard.
9. Translate model outputs into customer retention planning information.

---

## 3. Dataset

The project uses:

```text
E Commerce Dataset.xlsx
```

The dataset contains **5,630 customer records** and **20 columns**.

The target variable is:

```text
Churn
```

The dataset contains customer demographic, transactional, behavioral, satisfaction, and service-related attributes.

Examples of variables used in the analysis include:

- Tenure
- PreferredLoginDevice
- CityTier
- WarehouseToHome
- PreferredPaymentMode
- Gender
- HourSpendOnApp
- NumberOfDeviceRegistered
- PreferedOrderCat
- SatisfactionScore
- MaritalStatus
- NumberOfAddress
- Complain
- OrderAmountHikeFromlastYear
- CouponUsed
- OrderCount
- DaySinceLastOrder
- CashbackAmount

The raw Excel file is treated as the authoritative source for machine learning.

---

## 4. Project Architecture

The project is organized into three main analytical stages and one presentation layer.

```text
                         ┌──────────────────────┐
                         │  E-Commerce Dataset  │
                         │ E Commerce Dataset   │
                         │       .xlsx          │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      Stage 2         │
                         │ Data Cleaning + EDA  │
                         │ Business Insights    │
                         └──────────┬───────────┘
                                    │
                                    │ EDA reference
                                    ▼
                         ┌──────────────────────┐
                         │      Stage 3         │
                         │ Machine Learning     │
                         │ Model Evaluation     │
                         │ Threshold Selection  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      Stage 4         │
                         │ Streamlit Dashboard  │
                         │ Prediction + Planning│
                         └──────────────────────┘
```

### Important data-flow decision

The cleaned CSV generated during Stage 2 is used as an **EDA/business-analysis reference**.

Stage 3 deliberately reloads the original Excel dataset rather than using the Stage 2 cleaned CSV for machine learning.

This keeps the machine-learning preprocessing workflow separate from the exploratory-analysis cleaning workflow.

---

# 5. Stage 2 — Data Cleaning, Validation and EDA

Stage 2 performs exploratory analysis and prepares a cleaned reference dataset for business analysis.

The workflow includes:

- Dataset loading
- Structural inspection
- Missing-value analysis
- Duplicate analysis
- Data-type inspection
- Categorical-value standardization
- Numerical-data handling for EDA
- Churn distribution analysis
- Customer behavior analysis
- Satisfaction analysis
- Business-oriented visualizations
- Export of cleaned EDA reference data

The cleaned reference dataset is saved as:

```text
data/ecomm_cleaned.csv
```

This file is not used as the machine-learning training source.

---

## 6. Stage 2 Business Analysis

The EDA examines customer characteristics and their relationship with churn.

Areas explored include:

- Overall churn distribution
- Customer tenure
- Satisfaction score
- Complaint behavior
- Order behavior
- Login and device preferences
- Preferred order category
- Payment preferences
- Cashback and order-related variables
- Time since last order
- Customer demographic attributes

The analysis is intended to identify patterns and potential business signals rather than establish causal relationships.

Observed associations should therefore be interpreted as relationships in the available dataset, not proof that a particular variable directly causes churn.

---

# 7. Stage 3 — Machine Learning

Stage 3 builds and evaluates three classification models:

1. Logistic Regression
2. Random Forest
3. Gradient Boosting

The target is:

```text
Churn
```

A stratified train/test split is used:

```text
Training set: 80%
Test set:     20%
random_state: 42
```

The test set is held out for final evaluation.

---

## 8. Machine Learning Preprocessing

Numerical missing values are handled inside scikit-learn pipelines using:

```text
SimpleImputer(strategy="median")
```

The preprocessing pipeline is fitted using the training data.

Categorical standardization is performed before the train/test split.

The Stage 2 cleaned CSV is not used for machine-learning preprocessing or model training.

This design helps prevent information from the held-out test set from influencing preprocessing parameters.

---

# 9. Model Evaluation

The project evaluates models using metrics including:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion matrix

Five-fold cross-validation is used on the training set for model comparison.

The final test set remains separate and is used only for held-out evaluation and post-fit diagnostics.

---

## 10. Model Results

The main model results are summarized below.

| Model | CV F1 | CV ROC-AUC | Test F1 | Test ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.5890 | 0.8901 | 0.5781 | 0.8852 |
| Random Forest | 0.8225 | 0.9783 | 0.9266 | 0.9990 |
| Gradient Boosting | 0.8113 | 0.9604 | 0.8827 | 0.9893 |

Random Forest is used by the dashboard for customer-level risk scoring.

---

# 11. Threshold Selection

The project does not rely only on the default classification threshold of 0.50.

Instead, the operating threshold for the selected Random Forest model is chosen using **out-of-fold predictions on the training set**.

The objective is:

```text
Maximize F1-score
```

The selected threshold is:

```text
0.34
```

This threshold is selected without using the held-out test set.

The test set is used afterward for final evaluation.

---

## 12. Random Forest Threshold Comparison

At the default threshold of 0.50:

```text
Accuracy:  0.9769
Precision: 1.0000
Recall:    0.8632
F1:        0.9266
ROC-AUC:   0.9990
PR-AUC:    0.9952

TN: 936
FP:   0
FN:  26
TP: 164
```

At the F1-selected threshold of 0.34:

```text
Accuracy:  0.9813
Precision: 0.9122
Recall:    0.9842
F1:        0.9468

TN: 918
FP: 18
FN:  3
TP: 187
```

The threshold changes the operating point of the classifier.

A lower threshold identifies more customers as potential churners, increasing recall while also increasing the number of false positives.

---

# 13. Risk Scores and Risk Tiers

The Random Forest model produces a model score for each customer.

These scores are used to support customer-level risk segmentation.

The dashboard uses the following business risk boundaries:

```text
High Risk:
score >= 0.60

Medium Risk:
selected threshold <= score < 0.60

Low Risk:
score < selected threshold
```

The F1-selected threshold is loaded dynamically from the Stage 3 model artifact.

The score should be interpreted as a **model risk score**, not as a calibrated probability of churn.

---

# 14. Leakage-Sensitivity Analysis

The project also examines model sensitivity to potentially leakage-prone variables.

A no-leakage sensitivity experiment is performed by retraining models after removing:

```text
Complain
DaySinceLastOrder
```

This is a sensitivity analysis rather than a claim that these variables are necessarily invalid.

The resulting Random Forest test F1-score changes from:

```text
Original:  0.9266
No-leak:   0.8647
```

This demonstrates that model performance is sensitive to the inclusion of these variables.

The no-leakage experiment is included to make the model evaluation more transparent and to highlight the importance of considering feature availability and timing when deploying a churn model in a real business environment.

---

# 15. Stage 4 — Streamlit Dashboard

The project includes an interactive Streamlit dashboard.

Run it using:

```bash
streamlit run Inzimamul_Haq_Ecommerce_Customer_Retention_Intelligence.py
```

The dashboard contains five pages.

---

## Page 1 — Overview

Provides a high-level view of the customer retention problem.

It includes summary information such as:

- Customer population
- Churn distribution
- Key project metrics
- High-level business observations

---

## Page 2 — EDA Explorer

Provides interactive exploration of the dataset and selected churn-related variables.

The page supports analysis of relationships between customer characteristics and churn.

---

## Page 3 — Model Results

Presents the machine-learning evaluation results.

It includes information related to:

- Model comparison
- Cross-validation performance
- Test performance
- Threshold selection
- Confusion matrices
- Leakage-sensitivity results

Cross-validation metrics and held-out test metrics should be interpreted separately.

---

## Page 4 — Churn Predictor

Allows a user to enter customer characteristics and generate a churn-risk score using the trained Random Forest model.

The page provides:

- Customer risk score
- Risk classification
- Relevant prediction information

The score is a model-generated risk score and should not be interpreted as a guaranteed outcome.

---

## Page 5 — Retention Plan

Provides customer-level retention planning information.

Customers can be grouped into:

```text
High Risk
Medium Risk
Low Risk
```

The dashboard provides suggested retention actions associated with the risk segments.

These recommendations are decision-support hypotheses rather than guaranteed interventions or causal prescriptions.

---

# 16. Leakage Control

The project applies several controls to reduce data leakage during model development.

### Train/test separation

The dataset is split into training and held-out test sets using a stratified 80/20 split.

### Pipeline preprocessing

Numerical imputation is performed inside scikit-learn pipelines.

### Training-only threshold selection

The classification threshold is selected using out-of-fold predictions from the training data.

The test set is not used for threshold selection.

### Held-out test evaluation

The test set is reserved for final model evaluation and post-fit diagnostics.

### Stage separation

The Stage 2 cleaned CSV is not used as the machine-learning training source.

Stage 3 reloads the original Excel dataset.

### Sensitivity analysis

A no-leakage sensitivity experiment removes selected potentially leakage-prone variables and retrains the models.

---

# 17. Generated Artifacts

The project generates model and analysis artifacts used by the dashboard and for evaluation.

Important artifacts include:

```text
data/
    ecomm_cleaned.csv

model_random_forest.joblib
model_logistic_regression.joblib
model_gradient_boosting.joblib

selected_threshold.joblib
model_comparison.csv
```

Additional CSV and visualization outputs are generated during the EDA and machine-learning stages.

---

# 18. Project Structure

A typical project structure is:

```text
IBM_Capstone/
│
├── E Commerce Dataset.xlsx
│
├── Inzimamul_Haq_Ecommerce_Customer_Retention_Intelligence.py
│
├── requirements.txt
│
├── README.md
│
├── data/
│   └── ecomm_cleaned.csv
│
├── models/
│   ├── model_random_forest.joblib
│   ├── model_logistic_regression.joblib
│   ├── model_gradient_boosting.joblib
│   └── selected_threshold.joblib
│
├── outputs/
│   ├── model_comparison.csv
│   └── other generated analysis outputs
│
└── charts/
    └── generated visualizations
```

The exact artifact locations may vary depending on how the project is executed.

---

# 19. Installation

## Python Version

The project is intended to run with a modern Python 3 environment.

Create a virtual environment before installing the dependencies.

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## Install Dependencies

Install the required packages using:

```bash
pip install -r requirements.txt
```

The project uses libraries including:

- pandas
- numpy
- scipy
- scikit-learn
- matplotlib
- seaborn
- plotly
- openpyxl
- joblib
- streamlit

The project requirements file pins the package versions used by the implementation.

---

# 20. Running the Project

The consolidated Python file supports stage-based execution.

## Run Stage 2

```bash
python Inzimamul_Haq_Ecommerce_Customer_Retention_Intelligence.py --stage 2
```

This runs:

- Data loading
- Data validation
- Cleaning
- EDA
- Business analysis
- EDA artifact generation

---

## Run Stage 3

```bash
python Inzimamul_Haq_Ecommerce_Customer_Retention_Intelligence.py --stage 3
```

This runs:

- Train/test split
- Preprocessing
- Model training
- Cross-validation
- Threshold selection
- Test evaluation
- Leakage-sensitivity analysis
- Model artifact generation

---

## Run All Analytical Stages

```bash
python Inzimamul_Haq_Ecommerce_Customer_Retention_Intelligence.py --stage all
```

This runs the Stage 2 and Stage 3 workflows sequentially.

---

## Launch the Dashboard

After the required artifacts have been generated:

```bash
streamlit run Inzimamul_Haq_Ecommerce_Customer_Retention_Intelligence.py
```

The dashboard should then open in the browser.

---

# 21. Reproducibility

The machine-learning workflow uses fixed random seeds where applicable, including:

```text
random_state = 42
```

The model development workflow uses:

- A fixed stratified train/test split
- Five-fold cross-validation
- Training-set out-of-fold predictions for threshold selection
- Explicit preprocessing pipelines
- Saved model artifacts
- A saved selected threshold artifact

These design choices support reproducible execution under the same software environment and input data.

Exact numerical outputs may depend on the installed package versions and execution environment.

---

# 22. Important Interpretation Notes

### Model scores are not calibrated probabilities

The dashboard's customer risk score should be interpreted as a model score used for ranking and segmentation.

It should not automatically be interpreted as:

> "This customer has an X% probability of churning."

Calibration would require a separate calibration analysis.

### Association is not causation

The EDA and machine-learning results identify statistical patterns and predictive relationships.

They do not establish that a particular customer characteristic causes churn.

### Retention recommendations are decision-support suggestions

The retention actions shown in the dashboard are intended to support business decision-making.

They should be validated through appropriate business experiments or operational analysis before being treated as causal interventions.

### Threshold selection is business-sensitive

The selected threshold optimizes F1-score.

Different business costs for false positives and false negatives could justify a different operating threshold.

---

# 23. Key Findings

The project produced several important analytical findings.

### Customer churn is a minority class

The dataset contains:

```text
Total customers: 5,630
Retained:        4,682
Churned:           948
Overall churn:   16.8%
```

The class imbalance makes metrics such as precision, recall, F1-score, ROC-AUC, and PR-AUC useful alongside accuracy.

### Random Forest performed strongly on the available dataset

Random Forest produced:

```text
Test F1:     0.9266
Test ROC-AUC: 0.9990
```

at the default threshold.

Using the training-derived F1-selected threshold of 0.34 produced:

```text
Test F1: 0.9468
Recall:  0.9842
```

with the corresponding change in precision and confusion-matrix trade-offs described above.

### Leakage sensitivity matters

Removing `Complain` and `DaySinceLastOrder` reduced Random Forest test F1 from:

```text
0.9266 → 0.8647
```

This indicates that these variables have a substantial effect on predictive performance and should be considered carefully when defining the information available at the intended prediction time.

---

# 24. Limitations

This project has several limitations.

1. The analysis is based on a single provided e-commerce dataset.
2. The dataset may not represent every e-commerce customer population.
3. The model identifies predictive relationships rather than causal relationships.
4. The risk score is not calibrated as a probability.
5. The selected threshold optimizes F1-score rather than an explicit financial cost function.
6. Some variables may represent information that becomes available only after customer behavior has already changed.
7. The leakage-sensitivity analysis removes selected variables but does not prove that all possible forms of temporal or business-process leakage have been eliminated.
8. Retention recommendations require business validation before operational deployment.
9. Model performance on a new production population may differ from the reported test-set performance.

---

# 25. Recommended Production Considerations

Before deploying a similar system in a production environment, additional work should include:

- Defining the exact prediction time point.
- Ensuring every feature is available at prediction time.
- Monitoring data drift.
- Monitoring model performance over time.
- Calibrating risk scores if probability estimates are required.
- Selecting thresholds based on business costs and intervention capacity.
- Testing retention interventions experimentally.
- Establishing model governance and documentation.
- Periodically retraining the model using newly observed customer behavior.
- Monitoring false positives and false negatives after deployment.

---

# 26. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core implementation |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| SciPy | Scientific computing |
| Scikit-learn | Machine learning and preprocessing |
| Matplotlib | Visualization |
| Seaborn | Statistical visualization |
| Plotly | Interactive dashboard visualizations |
| Streamlit | Interactive web dashboard |
| Joblib | Model serialization |
| OpenPyXL | Excel file handling |

---

# 27. Conclusion

The E-Commerce Customer Retention Intelligence System provides an end-to-end workflow for analyzing customer churn and generating model-based retention insights.

The project combines:

```text
Data
  ↓
Cleaning & Validation
  ↓
EDA & Business Insights
  ↓
Machine Learning
  ↓
Cross-Validation
  ↓
OOF Threshold Selection
  ↓
Held-Out Test Evaluation
  ↓
Risk Scoring
  ↓
Interactive Dashboard
  ↓
Retention Planning
```

The implementation emphasizes separation between exploratory data preparation and machine-learning preprocessing, training/test separation, training-only threshold selection, and explicit sensitivity analysis for potentially leakage-prone variables.

The resulting system is designed as a **decision-support prototype** for customer retention analysis rather than a fully deployed production decision system.
