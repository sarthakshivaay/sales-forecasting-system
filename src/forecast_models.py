import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import pickle
from datetime import datetime, timedelta

# Time-series models
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

# Neural network
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

class SalesForecaster:
    def __init__(self, data_path='data/processed/rossmann_prepared.csv'):
        """Initialize forecaster with prepared data"""
        print("[INIT] Loading prepared data...")
        self.df = pd.read_csv(data_path)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.store_id = 1  # We'll forecast for Store 1 (largest store)
        self.forecast_days = 90
        self.models = {}
        self.results = {}
        print(f"[INIT] Data loaded. Shape: {self.df.shape}")
    
    def get_store_data(self, store_id=1):
        """Extract data for specific store"""
        store_data = self.df[self.df['Store'] == store_id].copy()
        store_data = store_data.sort_values('Date').reset_index(drop=True)
        return store_data
    
    def train_prophet(self):
        """Train Facebook Prophet model"""
        print("\n" + "="*60)
        print("MODEL 1: PROPHET (Time-Series Forecasting)")
        print("="*60)
        
        store_data = self.get_store_data(self.store_id)
        print(f"[PROPHET] Using {len(store_data)} records from Store {self.store_id}")
        
        # Prepare data for Prophet
        prophet_df = store_data[['Date', 'Sales']].copy()
        prophet_df.columns = ['ds', 'y']
        
        # Train model
        print("[PROPHET] Training model...")
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            interval_width=0.95
        )
        model.fit(prophet_df)
        print("[PROPHET] ✓ Model trained")
        
        # Make future dataframe
        future = model.make_future_dataframe(periods=self.forecast_days)
        forecast = model.predict(future)
        
        # Get test period for evaluation
        test_start = store_data['Date'].max() - timedelta(days=30)
        test_data = store_data[store_data['Date'] >= test_start].copy()
        
        # Evaluate on test set
        test_forecast = forecast[forecast['ds'].isin(test_data['Date'])].copy()
        test_actual = test_data['Sales'].values
        test_pred = test_forecast['yhat'].values[:len(test_actual)]
        
        mae = mean_absolute_error(test_actual, test_pred)
        rmse = np.sqrt(mean_squared_error(test_actual, test_pred))
        mape = mean_absolute_percentage_error(test_actual, test_pred)
        
        print(f"[PROPHET] MAE:  {mae:.2f}")
        print(f"[PROPHET] RMSE: {rmse:.2f}")
        print(f"[PROPHET] MAPE: {mape:.2%}")
        
        self.models['prophet'] = model
        self.results['prophet'] = {
            'forecast': forecast,
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }
        
        return model, forecast
    
    def train_arima(self):
        """Train ARIMA model"""
        print("\n" + "="*60)
        print("MODEL 2: ARIMA (Classical Time-Series)")
        print("="*60)
        
        store_data = self.get_store_data(self.store_id)
        sales = store_data['Sales'].values
        dates = store_data['Date'].values
        
        print(f"[ARIMA] Using {len(sales)} records from Store {self.store_id}")
        
        # Split into train/test
        train_size = int(len(sales) * 0.8)
        train_sales = sales[:train_size]
        test_sales = sales[train_size:]
        test_dates = dates[train_size:]
        
        print("[ARIMA] Training model with order (1,1,1)...")
        
        # Train ARIMA
        model = ARIMA(train_sales, order=(1, 1, 1))
        fitted = model.fit()
        print("[ARIMA] ✓ Model trained")
        
        # Forecast test period
        forecast_result = fitted.get_forecast(steps=len(test_sales))
        # In this statsmodels version, predicted_mean is already a NumPy array
        test_pred = forecast_result.predicted_mean
 
        
        # Metrics
        mae = mean_absolute_error(test_sales, test_pred)
        rmse = np.sqrt(mean_squared_error(test_sales, test_pred))
        mape = mean_absolute_percentage_error(test_sales, test_pred)
        
        print(f"[ARIMA] MAE:  {mae:.2f}")
        print(f"[ARIMA] RMSE: {rmse:.2f}")
        print(f"[ARIMA] MAPE: {mape:.2%}")
        
        # Forecast future
        future_forecast = fitted.get_forecast(steps=self.forecast_days + len(test_sales))
        future_pred = future_forecast.predicted_mean[-self.forecast_days:]

        future_dates = pd.date_range(
            start=store_data['Date'].max() + timedelta(days=1),
            periods=self.forecast_days
        )
        
        self.models['arima'] = fitted
        self.results['arima'] = {
            'forecast_dates': future_dates,
            'forecast_values': future_pred,
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }
        
        return fitted
    
    def train_lstm(self):
        """Train LSTM Neural Network"""
        print("\n" + "="*60)
        print("MODEL 3: LSTM (Deep Learning)")
        print("="*60)
        
        store_data = self.get_store_data(self.store_id)
        sales = store_data['Sales'].values.reshape(-1, 1)
        
        print(f"[LSTM] Using {len(sales)} records from Store {self.store_id}")
        
        # Normalize data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_sales = scaler.fit_transform(sales)
        
        # Create sequences
        lookback = 30  # Use 30 days to predict next day
        X, y = [], []
        
        for i in range(len(scaled_sales) - lookback):
            X.append(scaled_sales[i:i+lookback, 0])
            y.append(scaled_sales[i+lookback, 0])
        
        X = np.array(X).reshape(-1, lookback, 1)
        y = np.array(y)
        
        print(f"[LSTM] Created {len(X)} sequences (lookback={lookback})")
        
        # Split data
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Build model
        print("[LSTM] Building model...")
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(lookback, 1), return_sequences=True),
            Dropout(0.2),
            LSTM(50, activation='relu'),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        # Train
        print("[LSTM] Training model (20 epochs)...")
        history = model.fit(
            X_train, y_train,
            epochs=20,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=0
        )
        print("[LSTM] ✓ Model trained")
        
        # Evaluate
        y_pred = model.predict(X_test, verbose=0)
        y_pred = scaler.inverse_transform(y_pred)
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
        
        mae = mean_absolute_error(y_test_actual, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
        mape = mean_absolute_percentage_error(y_test_actual, y_pred)
        
        print(f"[LSTM] MAE:  {mae:.2f}")
        print(f"[LSTM] RMSE: {rmse:.2f}")
        print(f"[LSTM] MAPE: {mape:.2%}")
        
        self.models['lstm'] = {
            'model': model,
            'scaler': scaler,
            'lookback': lookback
        }
        self.results['lstm'] = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }
        
        return model
    
    def save_models(self):
        """Save all trained models"""
        print("\n" + "="*60)
        print("SAVING MODELS")
        print("="*60)
        
        # Save Prophet
        with open(f'models/prophet_store_{self.store_id}.pkl', 'wb') as f:
            pickle.dump(self.models['prophet'], f)
        print("[SAVE] ✓ Prophet model saved")
        
        # Save ARIMA
        with open(f'models/arima_store_{self.store_id}.pkl', 'wb') as f:
            pickle.dump(self.models['arima'], f)
        print("[SAVE] ✓ ARIMA model saved")
        
        # Save LSTM
        with open(f'models/lstm_store_{self.store_id}.pkl', 'wb') as f:
            pickle.dump(self.models['lstm'], f)
        print("[SAVE] ✓ LSTM model saved")
        
        # Save results
        with open(f'models/results_store_{self.store_id}.pkl', 'wb') as f:
            pickle.dump(self.results, f)
        print("[SAVE] ✓ Results saved")
    
    def print_summary(self):
        """Print model comparison summary"""
        print("\n" + "="*60)
        print("MODEL COMPARISON SUMMARY")
        print("="*60)
        
        print("\n{:<15} {:<12} {:<12} {:<12}".format(
            "Model", "MAE", "RMSE", "MAPE"
        ))
        print("-" * 60)
        
        for model_name in ['prophet', 'arima', 'lstm']:
            result = self.results[model_name]
            mae = result.get('mae', 0)
            rmse = result.get('rmse', 0)
            mape = result.get('mape', 0)
            
            print("{:<15} {:<12.2f} {:<12.2f} {:<12.2%}".format(
                model_name.upper(), mae, rmse, mape
            ))
        
        # Best model
        best_model = min(self.results.items(), key=lambda x: x[1]['mape'])[0]
        print("-" * 60)
        print(f"✅ Best Model: {best_model.upper()} (lowest MAPE)")
        print("="*60 + "\n")

def main():
    """Run all models"""
    forecaster = SalesForecaster()
    
    # Train all 3 models
    forecaster.train_prophet()
    forecaster.train_arima()
    forecaster.train_lstm()
    
    # Save models
    forecaster.save_models()
    
    # Summary
    forecaster.print_summary()
    
    print("✅ ALL MODELS TRAINED AND SAVED!")

if __name__ == '__main__':
    main()
