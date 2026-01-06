import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from datetime import timedelta, datetime
import pickle
import os

# PAGE CONFIG
st.set_page_config(
    page_title="Rossmann Sales Forecasting",
    page_icon="📊",
    layout="wide"
)

# DATA LOADING
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/processed/rossmann_prepared.csv")
        df["Date"] = pd.to_datetime(df["Date"])
        return df
    except FileNotFoundError:
        # Demo mode for Streamlit Cloud
        st.info("📊 Demo Mode: Generated sample data (local data not available)")
        np.random.seed(42)
        
        dates = pd.date_range(start="2013-01-01", end="2015-07-31", freq="D")
        stores = list(range(1, 6))
        
        data = []
        for store in stores:
            trend = 0
            for date in dates:
                trend += np.random.normal(0, 0.5)
                seasonality = 50 * np.sin(2 * np.pi * (date - dates[0]).days / 365)
                noise = np.random.normal(0, 50)
                sales = 2000 + trend + seasonality + noise
                sales = max(500, sales)
                
                data.append({
                    'Date': date,
                    'Store': store,
                    'Sales': sales,
                    'Customers': int(sales / 5),
                    'Promo': np.random.choice([0, 1], p=[0.7, 0.3]),
                    'Month': date.month,
                    'DayOfWeek': date.dayofweek,
                    'SchoolHoliday': 0,
                    'Quarter': (date.month - 1) // 3 + 1,
                    'Sales_lag_7': 0,
                    'Sales_lag_30': 0,
                    'Sales_rolling_7': 0
                })
        
        df = pd.DataFrame(data)
        return df.sort_values('Date').reset_index(drop=True)

df = load_data()
stores = sorted(df["Store"].unique())

# SIDEBAR NAVIGATION
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["📊 Forecast", "📈 Model Comparison", "ℹ️ About"]
)

# PAGE 1: FORECAST
if page == "📊 Forecast":
    st.title("📊 Rossmann Store Sales Forecasting")
    st.markdown("Interactive demo using Prophet to forecast daily sales.")

    col1, col2 = st.columns(2)
    with col1:
        store_id = st.selectbox("Select Store", stores, index=0)
    with col2:
        forecast_days = st.slider("Forecast Days", 30, 180, 90)

    store_data = df[df["Store"] == store_id].sort_values("Date")

    # Historical Sales
    st.subheader(f"Store {store_id} – Historical Sales")
    fig_hist, ax_hist = plt.subplots(figsize=(12, 4))
    ax_hist.plot(store_data["Date"], store_data["Sales"], color="tab:blue", linewidth=1.5)
    ax_hist.set_xlabel("Date")
    ax_hist.set_ylabel("Sales")
    ax_hist.grid(alpha=0.3)
    st.pyplot(fig_hist)

    # Forecast
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


# PAGE 2: MODEL COMPARISON
elif page == "📈 Model Comparison":
    st.title("📈 Model Comparison: Prophet vs ARIMA vs LSTM")

    st.markdown(
        """
        This page shows the performance comparison of three forecasting models:

        - **Prophet** – Flexible business time‑series model
        - **ARIMA(1,1,1)** – Classical statistical model
        - **LSTM** – Deep learning approach
        """
    )

    results_path = "models/results_store_1.pkl"

    if os.path.exists(results_path):
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
            - **MAE**: Average absolute error in sales units
            - **RMSE**: Penalizes larger errors more heavily
            - **MAPE**: Percentage error (easiest to interpret)
            """
        )
    else:
        st.warning("⚠️ Pre-trained models not available. Download from GitHub and run training locally.")


# PAGE 3: ABOUT
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")

    st.markdown(
        """
        ### 🧠 Project Overview

        End-to-end sales forecasting system for retail analytics using the 
        Rossmann Store Sales dataset (Kaggle competition).

        ### 📊 Business Problem

        Retailers need accurate sales forecasts to:
        - Optimize staffing schedules
        - Manage inventory levels
        - Plan promotional campaigns
        - Reduce stockouts and overstock

        ### 🧪 Models Used

        - **Prophet** – Captures trend + seasonality with interpretability
        - **ARIMA(1,1,1)** – Classical statistical time-series model
        - **LSTM** – Deep learning for sequence patterns

        ### 📂 Data Source

        **Rossmann Store Sales (Kaggle):**  
        https://www.kaggle.com/competitions/rossmann-store-sales

        ⚠️ Raw data NOT included (respects Kaggle's terms). Download from Kaggle yourself.

        ### 🏗️ Tech Stack

        - Python, pandas, numpy
        - Prophet, statsmodels (ARIMA), TensorFlow/Keras (LSTM)
        - Streamlit for dashboard
        - Git/GitHub for version control

        ### ⚠️ Limitations

        - Models trained on Store 1 only (baseline comparison)
        - No hyperparameter tuning (using defaults)
        - Evaluation on historical hold-out periods
        - Demo mode uses synthetic data for Cloud deployment

        ### 🚀 Future Work

        - Multi-store forecasting
        - Hyperparameter tuning
        - API integration
        - Docker deployment
        - Advanced architectures (Transformers, N-BEATS)

        ### 📞 Questions?

        **GitHub:** https://github.com/sarthakshivaay/sales-forecasting-system  
        **Author:** Sarthak Tyagi  
        **LinkedIn:** https://www.linkedin.com/in/sarthakshivaay
        """
    )

st.sidebar.markdown("---")
st.sidebar.caption("Rossmann Sales Forecasting · Jan 2026")
