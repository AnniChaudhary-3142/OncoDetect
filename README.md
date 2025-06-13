  **OncoDetect – High-Accuracy Breast Cancer Classification Using ML**

OncoDetect is a machine learning–driven classification system that leverages three robust models—Random Forest, XGBoost, and Logistic Regression—to detect breast cancer using diagnostic data. It achieved an accuracy of up to **98.2%** and an ROC-AUC of **0.996**, making it highly reliable for early-stage diagnosis.



**Dataset**

- **Source**: [Breast Cancer Wisconsin Diagnostic Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
- **Features**: 30 real-valued input features (mean, standard error, and worst of radius, texture, smoothness, etc.)
- **Samples**: 569 instances (357 benign, 212 malignant)
- **Target**: Binary classification – Malignant (0), Benign (1)



** Project Summary**

- Conducted extensive preprocessing using `StandardScaler` and stratified train-test split (80-20).
- Performed hyperparameter tuning using `GridSearchCV` with cross-validation for:
  - `Random Forest`
  - `XGBoost`
  - `Logistic Regression`
- Evaluated models using:
  - Accuracy
  - ROC-AUC Score
  - Confusion Matrix
  - Classification Report
  - Feature Importance Visualizations



**Model Performance**

| Model                | Accuracy | ROC-AUC | Precision (Benign) | Recall (Benign) | F1-score (Benign) |
|---------------------|----------|---------|---------------------|------------------|-------------------|
| Logistic Regression | 98.2%    | 0.996   | 0.99                | 0.99             | 0.99              |
| Random Forest       | 95.6%    | 0.993   | 0.96                | 0.97             | 0.97              |
| XGBoost             | 94.7%    | 0.992   | 0.95                | 0.97             | 0.96              |



## Visualizations

-  ROC Curves for all models
-  Confusion Matrices
-  Feature Importance Plots
-  Classification Reports





##  Requirements

- Python 3.9+
- scikit-learn
- xgboost
- matplotlib
- seaborn
- pandas
- numpy

> You can install dependencies using:
```bash
pip install -r requirements.txt
