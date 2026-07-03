# Credit Card Fraud Detection 💳🔍

Detecting fraudulent credit card transactions using **Isolation Forest** and **machine learning** techniques.

## Overview

This project analyzes credit card transaction data (PCA-transformed features V1–V28) to identify fraudulent transactions. The dataset is highly imbalanced — fraud cases represent only ~0.17% of all transactions.

## Dataset

- **Source:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Features:** 28 PCA-transformed features (V1–V28), Time, Amount
- **Target:** Class (0 = valid, 1 = fraud)
- **Imbalance:** ~0.17% fraud cases

## Approach

1. **Data Sampling** — 10% stratified sample for efficient processing
2. **Exploratory Analysis** — Distribution plots, correlation heatmap, class imbalance visualization
3. **Anomaly Detection** — Isolation Forest algorithm (unsupervised)
4. **Evaluation** — Confusion matrix, precision, recall, F1-score

## Results

| Metric | Value |
|--------|-------|
| Algorithm | Isolation Forest |
| Approach | Unsupervised anomaly detection |
| Key Challenge | Extreme class imbalance (~0.17%) |

## Tech Stack

- Python 3
- Pandas, NumPy — data manipulation
- Matplotlib, Seaborn — visualization
- Scikit-learn — Isolation Forest, evaluation metrics
- Jupyter Notebook

## Quick Start

```bash
pip install -r requirements.txt
jupyter notebook credit_card_fraud_detection.ipynb
```
