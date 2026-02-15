# Bank Marketing Classification App

## Problem Statement
The goal of this project is to build and evaluate machine learning models that predict whether a customer will subscribe to a term deposit based on the Bank Marketing dataset. This helps banks optimize marketing campaigns and improve customer targeting.

## Dataset Description
We used the **Bank Marketing Dataset** from UCI, which contains customer demographic and campaign-related attributes.  
- Rows: ~41,188
- Rows in test_bank.csv: 3000
- Rows in bank-additional-train.csv: 38188
- Rows in bank-additional-full.csv: 41188 
- Columns: 20 features + target column `y` (yes/no)  
- Separator: `;`  
- Target: `y` (subscription outcome)

## Models Used
We implemented six classification models:
- Logistic Regression  
- Decision Tree  
- k-Nearest Neighbors (kNN)  
- Naive Bayes  
- Random Forest (Ensemble)  
- XGBoost (Ensemble)

### Comparison Table of Metrics

| ML Model Name              | Accuracy | AUC    | Precision | Recall | F1    | MCC   |
|----------------------------|----------|--------|-----------|--------|-------|-------|
| Logistic Regression        | 0.9753   | 0.9649 | 0.4681    | 0.3099 | 0.3729| 0.3688|
| Decision Tree              | 0.9520   | 0.6799 | 0.2171    | 0.3944 | 0.2800| 0.2697|
| kNN                        | 0.9737   | 0.8276 | 0.4000    | 0.2254 | 0.2883| 0.2878|
| Naive Bayes                | 0.9677   | 0.7578 | 0.2593    | 0.1972 | 0.2240| 0.2098|
| Random Forest (Ensemble)   | 0.9757   | 0.9372 | 0.4444    | 0.1127 | 0.1798| 0.2151|
| XGBoost (Ensemble)         | 0.9707   | 0.9668 | 0.3651    | 0.3239 | 0.3433| 0.3290|



### Observations

| ML Model Name              | Observation about model performance |
|----------------------------|-------------------------------------|
| Logistic Regression        | Strong accuracy and AUC, with balanced precision, recall is modest due to class imbalance. |
| Decision Tree              | Easy to interpret, but lower precision and weaker overall balance; recall is higher than precision. |
| kNN                        | Good accuracy and precision, but recall is low, showing moderate performance overall. |
| Naive Bayes                | Fast and lightweight, but weaker precision and recall, reflecting its independence assumption limitations. |
| Random Forest (Ensemble)   | Excellent accuracy and AUC, but recall is very low, indicating difficulty in capturing positive cases. |
| XGBoost (Ensemble)         | High accuracy and AUC, with more balanced precision and recall compared to other models, making it the most reliable overall. |
