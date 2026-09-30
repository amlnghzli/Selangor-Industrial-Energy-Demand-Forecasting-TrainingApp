from pathlib import Path
import numpy as np
import pandas as pd

def load_and_clean_data(file_path: Path) -> pd.DataFrame:
    df = pd.read_excel(file_path, sheet_name="Selangor_Industrial_Demand")
    
    # 1. Datetime Reconstruction & Reindexing
    df["timestamp"] = pd.to_datetime(df["Date"]) + pd.to_timedelta(df["Hour"], unit="h")
    df = df.rename(columns={
        "Temperature_C": "temperature_c",
        "Humidity_%": "humidity_pct",
        "Industrial_Consumption_MWh": "industrial_power",
        "IsWeekend": "is_weekend"
    }).sort_values("timestamp").reset_index(drop=True)
    
    # Check continuous hourly frequency & handle missing grid timestamps
    full_range = pd.date_range(start=df["timestamp"].min(), end=df["timestamp"].max(), freq="h")
    df = df.set_index("timestamp").reindex(full_range)
    df.index.name = "timestamp"
    
    # Safe Forward-Fill Imputation for telemetry gaps
    df["industrial_power"] = df["industrial_power"].ffill()
    df["temperature_c"] = df["temperature_c"].ffill().bfill()
    df["humidity_pct"] = df["humidity_pct"].ffill().bfill()
    df["is_weekend"] = df["is_weekend"].ffill().bfill()
    
    return df.reset_index()

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # 2. Strict Shifted Features (Zero Leakage)
    df["lag_1"] = df["industrial_power"].shift(1)
    df["lag_24"] = df["industrial_power"].shift(24)
    df["lag_168"] = df["industrial_power"].shift(168)  # Seasonal Naive baseline lag
    df["roll_mean_24"] = df["industrial_power"].shift(1).rolling(24).mean()
    
    # Cyclical Time Embeddings
    df["hour_sin"] = np.sin(2 * np.pi * df["timestamp"].dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["timestamp"].dt.hour / 24)
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    
    return df.dropna().reset_index(drop=True)

def train_test_split_chronological(df: pd.DataFrame, test_hours: int = 168):
    split_idx = len(df) - test_hours
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df
