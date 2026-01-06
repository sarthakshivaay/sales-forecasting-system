import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from datetime import timedelta
import pickle
import os

# ------------------ PAGE CONFIG ------------------ #
st.set_page_config(
    page_title="Rossmann Sales Forecasting",
    page_icon="📊",
    layout="wide"
)

# ------------------ DATA LOADING ------------------ #
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/rossmann_prepared.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

df = load_data()
stores = sorted(df["Store"].unique())

# ------------------ SIDEBAR NAVIGATION ------------------ #
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["📊 Forecast", "📈 Model Comparison", "ℹ️ About"]
)

# =========================================================
# PAGE 1: FORECAST
# =========================================================
if page == "📊 Forecast":
    st.title("📊 Rossmann Store Sales Forecasting")
    st.markdown("Simple demo using Prophet to forecast daily sales for the next 90 days.")

    col1, col2 = st.columns(2)
    with col1:
        store_id = st.selectbox("Select Store", stores, index=0)
    with col2:
        forecast_days = st.slider("Forecast Days", 30, 180, 90)

    store_data = df[df["Store"] == store_id].sort_values("Date")

    # -------- Historical Sales Plot -------- #
    st.subheader(f"Store {store_id} – Historical Sales")

    fig_hist, ax_hist = plt.subplots(figsize=(12, 4))
    ax_hist.plot(store_data["Date"], store_data["Sales"], color="tab:blue", linewidth=1.5)
    ax_hist.set_xlabel("Date")
    ax_hist.set_ylabel("Sales")
    ax_hist.grid(alpha=0.3)
    st.pyplot(fig_hist)

    # -------- Forecast with Prophet -------- #
    st.subheader(f"Store {store_id} – {forecast_days}-Day Forecast (Prophet)")

    prophet_df = store_data[["Date", "Sales"]].rename(columns={"Date": "ds", "Sales": "y"})
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)

    hist_end = store_data["Date"].max()
    forecast_future = forecast[forecast["ds"] > hist_end]

    fig_forecast, ax_forecast = plt.subplots(figsize=(12, 4))
    ax_forecast.plot(store_data["Date"], store_data["Sales"],
                     label="Historical", color="tab:blue", linewidth=1.5)
    ax_forecast.plot(forecast_future["ds"], forecast_future["yhat"],
                     label="Forecast", color="tab:red", linewidth=1.5)
    ax_forecast.fill_between(
        forecast_future["ds"],
        forecast_future["yhat_lower"],
        forecast_future["yhat_upper"],
        color="tab:red",
        alpha=0.2,
        label="95% CI",
    )
    ax_forecast.set_xlabel("Date")
    ax_forecast.set_ylabel("Sales")
    ax_forecast.legend()
    ax_forecast.grid(alpha=0.3)
    st.pyplot(fig_forecast)

    col3, col4, col5 = st.columns(3)
    with col3:
        st.metric("Historical avg daily sales", f"€{store_data['Sales'].mean():.0f}")
    with col4:
        st.metric("Forecast avg daily sales", f"€{forecast_future['yhat'].mean():.0f}")
    with col5:
        change = (forecast_future["yhat"].mean() / store_data["Sales"].mean() - 1) * 100
        st.metric("Expected change", f"{change:+.1f}%")


# =========================================================
# PAGE 2: MODEL COMPARISON
# =========================================================
elif page == "📈 Model Comparison":
    st.title("📈 Model Comparison: Prophet vs ARIMA vs LSTM")

    st.markdown(
        """
        This page compares three forecasting approaches on the same store:

        - **Prophet** – Flexible business time-series model (trend + seasonality)
        - **ARIMA(1,1,1)** – Classical statistical time-series model
        - **LSTM** – Deep learning model for sequences
        """
    )

    results_path = "models/results_store_1.pkl"

    if not os.path.exists(results_path):
        st.error("Trained model results not found. Please run `python src/forecast_models.py` first.")
    else:
        with open(results_path, "rb") as f:
            results = pickle.load(f)

        data = []
        for name in ["prophet", "arima", "lstm"]:
            r = results[name]
            data.append({
                "Model": name.upper(),
                "MAE": round(r["mae"], 2),
                "RMSE": round(r["rmse"], 2),
                "MAPE (%)": round(r["mape"] * 100, 2),
            })

        comp_df = pd.DataFrame(data)

        st.subheader("📋 Error Metrics (Store 1)")
        st.dataframe(comp_df, use_container_width=True)

        st.subheader("🏆 Best Model (Lowest MAPE)")
        best = min(data, key=lambda x: x["MAPE (%)"])
        st.success(
            f"**{best['Model']}** has the lowest MAPE: **{best['MAPE (%)']}%** on Store 1."
        )

        st.markdown("### 📌 Interpretation")
        st.markdown(
            """
            - **MAE**: Average absolute error in sales units (how many € off on average).  
            - **RMSE**: Penalizes larger errors more strongly.  
            - **MAPE**: Percentage error; easier to interpret across scales.

            In many business settings, **MAPE** is the most intuitive metric
            because it answers: *“On average, how wrong are we in percentage terms?”*
            """
        )


# =========================================================
# PAGE 3: ABOUT
# =========================================================
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")

    st.markdown(
        """
        ### 🧠 Project Overview

        This app demonstrates an **end-to-end sales forecasting system** using the
        [Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales) dataset from Kaggle.  
        It forecasts daily sales for individual stores and compares different forecasting models.

        ### 📊 Business Problem

        Retailers need to plan:

        - Staffing  
        - Inventory  
        - Promotions  

        Accurate **90-day sales forecasts** help reduce stockouts, overstock, and scheduling issues.

        ### 🧪 Models Used

        - **Prophet** – Captures trend + weekly/yearly seasonality with interpretable components.  
        - **ARIMA(1,1,1)** – Classical linear time-series model.  
        - **LSTM** – Neural network for sequence data, capturing non-linear patterns.

        Each model has trade-offs:

        - Prophet: Great interpretability, fast to train.  
        - ARIMA: Simple, well-understood, but limited with complex patterns.  
        - LSTM: Powerful but needs more data, tuning, and compute.

        ### 📂 Data Source

        - **Dataset:** Rossmann Store Sales (Kaggle Competition)  
        - **Link:** https://www.kaggle.com/competitions/rossmann-store-sales  
        - **Note:** Raw CSV files are **not** included in this project to respect Kaggle's data rules.  
          Users should download the data directly from Kaggle.

        ### 🏗️ Tech Stack

        - Python, pandas, numpy  
        - Prophet, statsmodels (ARIMA), TensorFlow/Keras (LSTM)  
        - Streamlit for the dashboard  
        - Git/GitHub for version control  

        ### ⚠️ Limitations

        - Models are trained on a **single store** for comparison in this demo.  
        - No hyperparameter tuning yet (baseline configurations).  
        - No production monitoring or scheduled retraining implemented.  
        - Evaluation is performed on historical hold-out periods only.

        ### 🚀 Possible Extensions

        - Train and compare models across *all* 1,115 stores.  
        - Add hyperparameter tuning (Prophet, ARIMA, LSTM).  
        - Implement automatic model selection per store.  
        - Add anomaly detection for sudden drops/spikes in sales.  
        - Integrate with a backend API and scheduling for regular retraining.  

        ---

        Created as an educational, portfolio-ready time-series forecasting project.
        """
    )

# ------------------ FOOTER ------------------ #
st.sidebar.markdown("---")
st.sidebar.caption("Created by Sarthak Tyagi · January 2026")
