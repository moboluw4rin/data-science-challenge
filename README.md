# 5CCSAMLF: Machine Learning CW1 - Data Science Challenge

## Overview

This coursework is a data science challenge that addresses a supervised regression problem. The objective is to train a model that predicts the target variable 'outcome' using the remaining features in the diamond dataset.

## Model selection
I've selected the following 5 algorithms and compared them directly:
- Linear Regression
- Ridge
- Random Forest
- XGBoost 
- Support Vector Regression

Performance is evaluated using out-of-sample R² on a held-out test set. The goal is to minimise generalisation error and approach the Bayes optimal model.

## Repository Strcuture
```
cw1/
│
├── README.md
├── requirements.txt
├── src/
│   └── cw1_script.py
├── CW1_train.csv
├── CW1_test.csv
└── CW1_test_predictions.csv   (generated after running)
```
## How to Run
1. Install dependencies


```
 pip install -r requirements.txt
```


2.  Run training and prediction

```
python src/cw1_script.py
```

3. Output


The script generates:

```
CW1_test_predictions.csv
```
