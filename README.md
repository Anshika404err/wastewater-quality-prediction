# Wastewater Quality Prediction (COD and TN)

Machine learning app that predicts COD and total nitrogen (TN) in wastewater plant inflow, with a Streamlit interface and high-load flagging.

**Live app:** https://wastewater-prediction.streamlit.app/
<img width="1904" height="947" alt="image" src="https://github.com/user-attachments/assets/a90998c9-70ec-4368-8008-5bfd603cc91f" />
<img width="1852" height="853" alt="image" src="https://github.com/user-attachments/assets/5d51c375-1974-4b6a-9019-8e84058edf40" />



## What it does
- Predicts COD and TN from today's flow, ammonia, weather and recent history (lag and 7-day rolling features).
- Flags a day as HIGH LOAD when a prediction exceeds the historical 90th percentile.
- Supports single predictions and CSV batch predictions.

## Data
Daily Melbourne wastewater plant data (Kaggle: Full Scale Waste Water Treatment Plant Data), 1,382 rows. The values are plant **inflow**, not treated discharge.

## Method
- Time-based 80/20 split (no shuffling) to avoid leakage between neighbouring days.
- Compared against two naive baselines: training mean and previous-record.
- Random Forest with lag and rolling-average features; also checked with time-series cross-validation.

## Results (R², higher is better)

| Target | Train-mean baseline | Previous-record baseline | Random Forest (holdout) | Time-series CV |
|---|---|---|---|---|
| COD | -0.168 | -0.345 | **0.125** | 0.069 |
| TN | -0.030 | -0.342 | **0.172** | 0.144 |
| BOD | -0.843 | -0.074 | -0.583 | -0.203 |

BOD could not beat the baselines, so it is not predicted in the app. Signal in this dataset is weak, so scores are modest.

## Limitations
- Inflow data, so it cannot check treated-discharge limits.
- Weak feature-target correlation and drift over time.
- Future work: more years of data, effluent measurements, gradient boosting.

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

**Tech stack:** Python, pandas, NumPy, scikit-learn, Streamlit
