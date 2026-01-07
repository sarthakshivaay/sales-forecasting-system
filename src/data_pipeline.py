import os
import pandas as pd
import numpy as np


RAW_TRAIN_PATH = "data/raw/train.csv"
RAW_STORE_PATH = "data/raw/store.csv"
PROCESSED_PATH = "data/processed/rossmann_prepared.csv"


def load_raw_data():
    print("=" * 60)
    print("ROSSMANN DATA PIPELINE")
    print("=" * 60)
    print("[1/5] Loading data...")

    if not os.path.exists(RAW_TRAIN_PATH):
        raise FileNotFoundError(f"Missing file: {RAW_TRAIN_PATH}")
    if not os.path.exists(RAW_STORE_PATH):
        raise FileNotFoundError(f"Missing file: {RAW_STORE_PATH}")

    train = pd.read_csv(RAW_TRAIN_PATH)
    store = pd.read_csv(RAW_STORE_PATH)

    print(f"✓ Loaded {len(train):,} train records")
    print(f"✓ Loaded {len(store):,} store records")

    return train, store


def merge_and_clean(train, store):
    print("[2/5] Merging & cleaning data...")

    # Merge store metadata
    df = train.merge(store, on="Store", how="left")

    # Parse date
    df["Date"] = pd.to_datetime(df["Date"])

    # Remove days when stores were closed (Open = 0 or NaN treated as closed)
    if "Open" in df.columns:
        df = df[df["Open"].fillna(0) == 1]

    # Handle missing competition distance & promo columns
    if "CompetitionDistance" in df.columns:
        df["CompetitionDistance"] = df["CompetitionDistance"].fillna(
            df["CompetitionDistance"].median()
        )

    # Some store files have Promo2SinceYear etc.; just fill NAs if present
    for col in ["Promo2SinceYear", "Promo2SinceWeek", "PromoInterval"]:
        if col in df.columns:
            df[col] = df[col].fillna(0 if df[col].dtype != "object" else "None")

    print(f"✓ Cleaned data: {len(df):,} records after filtering closed days")
    return df


def add_time_features(df):
    print("[3/5] Engineering time features...")

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Quarter"] = ((df["Month"] - 1) // 3 + 1).astype(int)

    print("✓ Added basic calendar features")
    return df


def add_lag_features(df, lag_days=(7, 30), rolling_windows=(7,)):
    print("[4/5] Creating lag / rolling features...")

    df = df.sort_values(["Store", "Date"])
    for lag in lag_days:
        df[f"Sales_lag_{lag}"] = (
            df.groupby("Store")["Sales"].shift(lag)
        )

    for window in rolling_windows:
        df[f"Sales_rolling_{window}"] = (
            df.groupby("Store")["Sales"]
            .shift(1)
            .rolling(window=window, min_periods=1)
            .mean()
        )

    print(f"✓ Created lag features: {lag_days}")
    print(f"✓ Created rolling features: {rolling_windows}")
    return df


def select_and_save(df):
    print("[5/5] Selecting features & saving...")

    # Ensure metadata columns exist (they come from store.csv)
    # If they don't, fill with default values so the app never breaks.
    for col, default in [
        ("StoreType", "a"),
        ("Assortment", "basic"),
        ("CompetitionDistance", df["CompetitionDistance"].median() if "CompetitionDistance" in df.columns else 1000),
    ]:
        if col not in df.columns:
            df[col] = default

    # Columns used by the Streamlit app
    cols_to_keep = [
        "Date",
        "Store",
        "Sales",
        "Customers",
        "Promo",
        "StateHoliday",
        "SchoolHoliday",
        "Open",

        # store metadata (for the blue info box)
        "StoreType",
        "Assortment",
        "CompetitionDistance",

        # time features
        "Year",
        "Month",
        "Day",
        "DayOfWeek",
        "WeekOfYear",
        "Quarter",

        # lag / rolling features
        "Sales_lag_7",
        "Sales_lag_30",
        "Sales_rolling_7",
    ]

    # keep only columns that actually exist (in case something is missing)
    cols_final = [c for c in cols_to_keep if c in df.columns]
    df_final = df[cols_final].copy()

    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    df_final.to_csv(PROCESSED_PATH, index=False)

    print(f"✓ Saved to {PROCESSED_PATH}")
    print(f"✓ Data shape: {df_final.shape}")
    print(f"✓ Date range: {df_final['Date'].min().date()} to {df_final['Date'].max().date()}")
    print("=" * 60)
    print("✅ PIPELINE COMPLETE!")
    print("=" * 60)


def main():
    train, store = load_raw_data()
    df = merge_and_clean(train, store)
    df = add_time_features(df)
    df = add_lag_features(df)
    select_and_save(df)


if __name__ == "__main__":
    main()
