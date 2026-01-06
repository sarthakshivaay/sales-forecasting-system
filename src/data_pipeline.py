import pandas as pd
import numpy as np
from pathlib import Path

class RossmannPipeline:
    def __init__(self):
        self.train_path = 'data/raw/train.csv'
        self.store_path = 'data/raw/store.csv'
        self.output_path = 'data/processed/rossmann_prepared.csv'
    
    def load_data(self):
        """Load raw Rossmann data"""
        print("[1/5] Loading data...")
        train = pd.read_csv(self.train_path)
        stores = pd.read_csv(self.store_path)
        df = train.merge(stores, on='Store', how='left')
        print(f"✓ Loaded {len(df):,} records from {df['Store'].nunique()} stores")
        return df
    
    def clean_data(self, df):
        """Clean and prepare data"""
        print("[2/5] Cleaning data...")
        
        # Convert Date to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Remove days when store was closed
        df = df[df['Open'] == 1].copy()
        
        # Handle missing values
        df['CompetitionDistance'].fillna(df['CompetitionDistance'].median(), inplace=True)
        df['Promo2SinceWeek'].fillna(0, inplace=True)
        df['Promo2SinceYear'].fillna(0, inplace=True)
        
        print(f"✓ Cleaned data: {len(df):,} records")
        return df
    
    def engineer_features(self, df):
        """Create time-based features"""
        print("[3/5] Engineering features...")
        
        df = df.copy()
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Time features
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Week'] = df['Date'].dt.isocalendar().week
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['Quarter'] = df['Date'].dt.quarter
        df['DayOfMonth'] = df['Date'].dt.day
        
        # Store type encoding
        df['StoreType_A'] = (df['StoreType'] == 'a').astype(int)
        df['StoreType_B'] = (df['StoreType'] == 'b').astype(int)
        df['StoreType_C'] = (df['StoreType'] == 'c').astype(int)
        df['StoreType_D'] = (df['StoreType'] == 'd').astype(int)
        
        # Lag features (previous week sales)
        df['Sales_lag_7'] = df.groupby('Store')['Sales'].shift(7).fillna(0)
        df['Sales_lag_30'] = df.groupby('Store')['Sales'].shift(30).fillna(0)
        
        # Rolling average
        df['Sales_rolling_7'] = df.groupby('Store')['Sales'].transform(
            lambda x: x.rolling(7, min_periods=1).mean()
        )
        
        print(f"✓ Created {len([c for c in df.columns if c.startswith('Sales_') or c in ['Year', 'Month', 'DayOfWeek']])} features")
        return df
    
    def select_features(self, df):
        """Select final features for modeling"""
        print("[4/5] Selecting features...")
        
        # Key columns for time-series forecasting
        features = ['Date', 'Store', 'Sales', 'Customers', 'Promo', 
                   'Month', 'DayOfWeek', 'SchoolHoliday', 'Quarter',
                   'Sales_lag_7', 'Sales_lag_30', 'Sales_rolling_7']
        
        df = df[features].dropna()
        print(f"✓ Selected {len(features)} features")
        return df
    
    def save_data(self, df):
        """Save processed data"""
        print("[5/5] Saving processed data...")
        df.to_csv(self.output_path, index=False)
        print(f"✓ Saved to {self.output_path}")
        print(f"✓ Data shape: {df.shape}")
        print(f"✓ Date range: {df['Date'].min()} to {df['Date'].max()}")
        return df
    
    def run(self):
        """Execute full pipeline"""
        print("=" * 60)
        print("ROSSMANN DATA PIPELINE")
        print("=" * 60)
        
        df = self.load_data()
        df = self.clean_data(df)
        df = self.engineer_features(df)
        df = self.select_features(df)
        df = self.save_data(df)
        
        print("=" * 60)
        print("✅ PIPELINE COMPLETE!")
        print("=" * 60)

if __name__ == '__main__':
    pipeline = RossmannPipeline()
    pipeline.run()
