from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Wastewater Quality Predictor", page_icon="💧", layout="wide")


@st.cache_resource
def load_bundle():
    return joblib.load(Path(__file__).parent / "model" / "model.pkl")


bundle = load_bundle()
model, features, targets = bundle["model"], bundle["features"], bundle["targets"]
defaults, thresholds, metrics = bundle["defaults"], bundle["thresholds"], bundle["metrics"]


def predict(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in features:
        if c not in df.columns:
            df[c] = defaults[c]
    return pd.DataFrame(model.predict(df[features]), columns=targets, index=df.index)


st.title("💧 Wastewater Quality Prediction")
st.caption(
    "Predicts COD and Total Nitrogen (TN) of plant inflow from today's flow and weather plus recent history. "
    "Trained on the Melbourne plant dataset (Kaggle) with a time-based split."
)

tab1, tab2, tab3 = st.tabs(["Single prediction", "Batch (CSV)", "Model performance"])

with tab1:
    st.subheader("Enter today's plant readings")
    c1, c2, c3 = st.columns(3)
    avg_inflow = c1.number_input("Avg inflow", value=float(defaults["avg_inflow"]), format="%.3f")
    avg_outflow = c1.number_input("Avg outflow", value=float(defaults["avg_outflow"]), format="%.3f")
    total_grid = c2.number_input("Total grid (Wh)", value=float(defaults["total_grid"]))
    am = c2.number_input("Ammonia (Am)", value=float(defaults["Am"]))
    month = c3.number_input("Month", min_value=1, max_value=12, value=int(defaults["month"]))

    st.markdown("**Recent history**")
    h1, h2, h3, h4 = st.columns(4)
    cod_prev = h1.number_input("Yesterday's COD", value=float(defaults["COD_lag1"]))
    cod_avg = h2.number_input("Last-7-day avg COD", value=float(defaults["COD_roll7"]))
    tn_prev = h3.number_input("Yesterday's TN", value=float(defaults["TN_lag1"]))
    tn_avg = h4.number_input("Last-7-day avg TN", value=float(defaults["TN_roll7"]))

    if st.button("Predict", type="primary"):
        row = dict(defaults)
        row.update({
            "avg_inflow": avg_inflow, "avg_inflow_lag1": avg_inflow, "avg_inflow_roll7": avg_inflow,
            "avg_outflow": avg_outflow, "avg_outflow_lag1": avg_outflow, "avg_outflow_roll7": avg_outflow,
            "total_grid": total_grid, "Am": am, "Am_lag1": am, "Am_roll7": am, "month": month,
            "COD_lag1": cod_prev, "COD_roll7": cod_avg, "TN_lag1": tn_prev, "TN_roll7": tn_avg,
        })
        out = predict(pd.DataFrame([row])).iloc[0]
        cols = st.columns(len(targets))
        for col, t in zip(cols, targets):
            val, lim = out[t], thresholds[t]
            col.metric(f"Predicted {t}", f"{val:.1f}")
            if val > lim:
                col.error(f"HIGH LOAD: above threshold ({lim:.1f})")
            else:
                col.success(f"Normal: below threshold ({lim:.1f})")
        st.caption("Threshold = 90th percentile of historical values. Weather inputs use typical (median) values.")

with tab2:
    st.subheader("Upload a CSV for batch predictions")
    template = pd.DataFrame([defaults])[features]
    st.download_button("Download CSV template", template.to_csv(index=False), "template.csv")
    up = st.file_uploader("CSV file", type="csv")
    if up:
        data = pd.read_csv(up)
        missing = [c for c in features if c not in data.columns]
        if missing:
            st.warning(f"{len(missing)} columns missing; filled with typical values.")
        preds = predict(data)
        result = pd.concat([data, preds.add_prefix("pred_")], axis=1)
        for t in targets:
            result[f"{t}_flag"] = (preds[t] > thresholds[t]).map({True: "HIGH", False: "OK"})
        st.dataframe(result)
        st.line_chart(preds)
        st.download_button("Download results", result.to_csv(index=False), "predictions.csv")

with tab3:
    st.subheader("How good is the model, honestly?")
    st.dataframe(pd.DataFrame(metrics).T.rename(columns={"holdout_r2": "Holdout R²", "cv_r2": "Time-series CV R²"}))
    st.markdown(
        "- Scores are **modest**: daily plant data is noisy and drifts over time.\n"
        "- Evaluated with a **time-based split** (no shuffling) so results are not inflated by leakage.\n"
        "- Both targets beat the naive baselines (training mean and previous-record).\n"
        "- **BOD** was tested too but no model beat the baselines, so it is not predicted here.\n"
        "- The compliance flag marks days above the historical 90th percentile; the data is plant *inflow*, "
        "not treated discharge."
    )