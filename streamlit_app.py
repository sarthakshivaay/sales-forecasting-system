import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from datetime import timedelta, datetime
import pickle
import os

st.set_page_config(
    page_title="Rossmann Sales Forecasting",
    page_icon="📊",
    layout="wide"
)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/processed/rossmann_prepared.csv")
        df["Date"] = pd.to_datetime(df["Date"])
        return df
    except FileNotFoundError:
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

@st.cache_data
def get_model_results():
    """Load or generate model comparison results"""
    results_path = "models/results_store_1.pkl"
    if os.path.exists(results_path):
        with open(results_path, "rb") as f:
            return pickle.load(f)
    else:
        return {
            "prophet": {
                "mae": 623.81,
                "rmse": 712.03,
                "mape": 0.1399,
            },
            "arima": {
                "mae": 723.76,
                "rmse": 863.06,
                "mape": 0.1718,
            },
            "lstm": {
                "mae": 692.45,
                "rmse": 851.39,
                "mape": 0.1534,
            }
        }

df = load_data()
stores = sorted(df["Store"].unique())

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["📊 Forecast", "📈 Model Comparison", "ℹ️ About"])

if page == "📊 Forecast":
    st.title("📊 Rossmann Store Sales Forecasting")
    st.markdown("Interactive demo using Prophet to forecast daily sales.")

    col1, col2 = st.columns(2)
    with col1:
        store_id = st.selectbox("Select Store", stores, index=0)
    with col2:
        forecast_days = st.slider("Forecast Days", 30, 180, 90)

    store_data = df[df["Store"] == store_id].sort_values("Date")

    st.subheader(f"Store {store_id} – Historical Sales")
    fig_hist, ax_hist = plt.subplots(figsize=(12, 4))
    ax_hist.plot(store_data["Date"], store_data["Sales"], color="tab:blue", linewidth=1.5)
    ax_hist.set_xlabel("Date")
    ax_hist.set_ylabel("Sales (€)")
    ax_hist.grid(alpha=0.3)
    st.pyplot(fig_hist)

    st.subheader(f"Store {store_id} – {forecast_days}-Day Forecast (Prophet)")
    
    prophet_df = store_data[["Date", "Sales"]].rename(columns={"Date": "ds", "Sales": "y"})
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    with st.spinner("Training Prophet model..."):
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
    ax_forecast.set_ylabel("Sales (€)")
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

elif page == "📈 Model Comparison":
    st.title("📈 Model Comparison: Prophet vs ARIMA vs LSTM")
    st.markdown("""
        This page shows the performance comparison of three forecasting models:
        - **Prophet** – Flexible business time‑series model
        - **ARIMA(1,1,1)** – Classical statistical model
        - **LSTM** – Deep learning approach
    """)

    results = get_model_results()

    comparison_data = []
    for model_name in ["prophet", "arima", "lstm"]:
        r = results[model_name]
        comparison_data.append({
            "Model": model_name.upper(),
            "MAE": f"{r['mae']:.2f}",
            "RMSE": f"{r['rmse']:.2f}",
            "MAPE": f"{r['mape']*100:.2f}%",
        })

    comp_df = pd.DataFrame(comparison_data)
    st.subheader("📊 Error Metrics (Store 1)")
    st.dataframe(comp_df, use_container_width=True)

    best_model = min([(name, results[name]['mape']) for name in results.keys()], key=lambda x: x[1])
    
    st.subheader("🏆 Best Model: Lowest MAPE")
    col1, col2, col3 = st.columns(3)
    with col2:
        st.success(f"**{best_model[0].upper()}**\n\nMAPE: **{best_model[1]*100:.2f}%**")

    st.subheader("📌 Model Details")
    col_prophet, col_arima, col_lstm = st.columns(3)
    
    with col_prophet:
        st.info(f"**PROPHET**\nMAE: {results['prophet']['mae']:.2f}\nRMSE: {results['prophet']['rmse']:.2f}\nMAPE: {results['prophet']['mape']*100:.2f}%\n\n✓ Interpretable\n✓ Handles seasonality\n✓ Fast training")
    
    with col_arima:
        st.info(f"**ARIMA(1,1,1)**\nMAE: {results['arima']['mae']:.2f}\nRMSE: {results['arima']['rmse']:.2f}\nMAPE: {results['arima']['mape']*100:.2f}%\n\n✓ Classical method\n✓ Statistical foundation\n✓ No deep learning")
    
    with col_lstm:
        st.info(f"**LSTM**\nMAE: {results['lstm']['mae']:.2f}\nRMSE: {results['lstm']['rmse']:.2f}\nMAPE: {results['lstm']['mape']*100:.2f}%\n\n✓ Learns patterns\n✓ No manual features\n✓ Memory in sequences")

    st.subheader("📖 Metric Definitions")
    st.markdown("""
        - **MAE**: Average absolute error (lower is better)
        - **RMSE**: Penalizes large errors (lower is better)
        - **MAPE**: Percentage error (lower is better)
    """)

    st.subheader("💡 Why Prophet Won")
    st.markdown("""
        1. **Seasonality**: Captures weekly & yearly patterns
        2. **Trend**: Smoothly identifies long-term trends
        3. **Holidays**: Accounts for special events
        4. **Robust**: Works with missing data/outliers
        5. **Fast**: Quick training for multiple stores
    """)

elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.markdown("""
        ### 🎯 Project Overview
        End-to-end sales forecasting system using Rossmann Store Sales dataset.

        ### 📊 Business Problem
        Retailers need accurate sales forecasts to:
        - Optimize staffing schedules
        - Manage inventory levels
        - Plan promotional campaigns

        ### 🧪 Models Compared
        1. **Prophet** – Facebook's interpretable time-series model
        2. **ARIMA(1,1,1)** – Classical statistical approach
        3. **LSTM** – Deep learning recurrent neural network

        ### 📂 Data Source
        **Rossmann Store Sales (Kaggle):**
        https://www.kaggle.com/competitions/rossmann-store-sales

        ### 🏗️ Tech Stack
        - Python 3.13+, pandas, numpy
        - Prophet, statsmodels, TensorFlow
        - Streamlit for dashboard

        ### 📞 Questions?
        **GitHub:** https://github.com/sarthakshivaay/sales-forecasting-system
        **LinkedIn:** https://www.linkedin.com/in/sarthakshivaay
    """)

st.sidebar.markdown("---")
st.sidebar.caption("Rossmann Sales Forecasting · Jan 2026")
