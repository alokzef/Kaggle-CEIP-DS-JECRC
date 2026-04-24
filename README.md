# Sensor Data Classification: CEIP-DS-JECRC

This repository contains a high-performance machine learning solution for classifying sensor-based data. The approach utilizes **LightGBM** with custom windowed features and precision threshold tuning to handle highly imbalanced classes.

---

## 🚀 Overview
The objective is to predict binary targets from a time-series sensor dataset ($X1$ to $X5$). This solution focuses on capturing temporal dependencies and signal volatility through manual feature extraction.

## 🛠️ Feature Engineering
To improve the model's predictive power, the following transformations were applied:
* **Time Features:** Extracted `hour`, `day_of_week`, and `is_weekend` from timestamps.
* **Sensor Aggregates:** Computed row-wise `mean`, `std`, and `max` across all sensors.
* **Windowing & Smoothing:** * **Rolling Mean & Variance:** Captured local trends and volatility.
    * **EWMA:** Exponentially Weighted Moving Averages for noise reduction.
    * **Differencing:** Calculated step-to-step changes ($X_t - X_{t-1}$).
* **Signal Ratios:** Created $X1/X2$ and $X3/X4$ interaction features.

## 📉 Methodology
* **Validation:** A **73/27 Chronological Split** was used to simulate real-world forecasting and prevent data leakage from future timestamps.
* **Scaling:** Features were normalized using `StandardScaler`.
* **Model:** `LGBMClassifier` with `scale_pos_weight` to address the heavy class imbalance.
* **Optimization:** Custom thresholding was implemented. Instead of the default $0.5$, the model searches for the optimal threshold (found at **0.97**) to maximize the **F1-Score**.

## 📊 Performance
* **Validation F1-Score:** ~0.618
* **Optimal Threshold:** 0.97
* **Model Params:** 250 estimators, 0.03 learning rate, max depth of 5.

## 📂 Requirements
```python
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
