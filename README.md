# Term-Deposit Subscription — Practical Application 3: Comparing Classifiers

This repository contains my analysis for **UC Berkeley ML/AI – Practical Application 3** on bank-marketing telemarketing data.  
**Goal:** predict whether a client will subscribe to a term deposit **before the call** and compare multiple classifiers.

> 📒 Open the analysis notebook: **[nic_r-practical_application-3.ipynb](./nic_r-practical_application-3.ipynb)**

---

## Summary of Findings (business-friendly)

- **Best model:** **Linear SVM** delivered the strongest discrimination on the test set (highest ROC-AUC), with **Logistic Regression** a close second. _If probability outputs are required from SVM, calibrate with Platt or Isotonic calibration._
- **Key signals (pre-call):** prior successful outcomes (`poutcome=success`) and **cellular** contact correlate with higher conversion; excessive same-campaign contacts (`campaign`) and very recent re-contacts (`pdays` not 999) reduce odds. Seasonality (**month**) and macro indicators (e.g., `euribor3m`, `nr.employed`) matter.
- **Metric choice:** We optimize **ROC-AUC** due to class imbalance; we also report **PR-AUC**, precision/recall, F1, and accuracy for context.
- **Actionable use:** score leads daily, **rank by predicted probability**, set a business threshold (e.g., target precision or recall), and call from the top down.
- **Important note:** we **exclude `duration`** (known only after the call) to avoid target leakage and ensure the model is usable at dial time.

---

## Data & Methods (brief)

- **Dataset:** UCI Bank Marketing (“bank-additional-full.csv”, Portuguese bank telemarketing outcomes).
- **Prep:** treat `"unknown"` as missing where appropriate; engineer “no prior contact” from `pdays==999`; one-hot encode categoricals; scale numeric features; exclude `duration`.
- **Modeling:** compared **KNN**, **Logistic Regression**, **Decision Tree**, and **SVM** (linear and RBF); used **Stratified 5-fold CV** with small **GridSearchCV** grids; 80/20 train–test split.
- **Evaluation:** primary **ROC-AUC**; secondary **PR-AUC**, precision, recall, F1, accuracy; majority-class baseline included.
- **Interpretability:** interpreted feature effects via **model outputs and EDA** (see notebook).

---

## How to Run

1. `pip install -r requirements.txt` (pandas, scikit-learn, seaborn/matplotlib).
2. Open **nic_r-practical_application-3.ipynb** and run all cells.

---

## Next Steps & Recommendations

- **Threshold tuning** to match business goals (e.g., precision ≥ 2× baseline conversion, or target recall).
- **Probability calibration** (Platt/Isotonic) if deploying Linear SVM with probabilities; **quarterly drift checks**; retrain if lift declines.
- Explore **gradient boosting** (XGBoost/LightGBM) as a follow-on benchmark while maintaining interpretability.

---

## Repository Contents

- `nic_r-practical_application-3.ipynb` — full CRISP-DM analysis (EDA → Prep → Modeling → Evaluation → Recommendations).

---

## Acknowledgments

Course: **UC Berkeley ML/AI (Emeritus)** — Practical Application 3. Built with Python, pandas, scikit-learn, seaborn/matplotlib.
