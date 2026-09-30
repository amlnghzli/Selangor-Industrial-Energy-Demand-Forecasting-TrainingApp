import numpy as np
import pandas as pd
import lightgbm as lgb

FEATURES = [
    "lag_1", "lag_24", "lag_168", "roll_mean_24",
    "temperature_c", "humidity_pct", "is_weekend",
    "hour_sin", "hour_cos", "day_of_week"
]

def train_gbdt(train_df: pd.DataFrame):
    # LightGBM Regressor - avoids sklearn _kd_tree C-extension DLL policy blocks
    model = lgb.LGBMRegressor(
        n_estimators=150,
        learning_rate=0.05,
        random_state=42,
        verbosity=-1
    )
    model.fit(train_df[FEATURES], train_df["industrial_power"])
    return model

def forecast_30_days(df: pd.DataFrame, model) -> pd.DataFrame:
    last_timestamp = df["timestamp"].max()
    future_dates = pd.date_range(start=last_timestamp + pd.Timedelta(hours=1), periods=720, freq="h")
    
    future_df = pd.DataFrame({"timestamp": future_dates})
    combined = pd.concat([df, future_df], ignore_index=True)
    
    # Auto-regressive iterative forecasting loop
    for i in range(len(df), len(combined)):
        hist = combined.iloc[:i]
        
        # Shifted lag calculations
        combined.loc[i, "lag_1"] = hist["industrial_power"].iloc[-1]
        combined.loc[i, "lag_24"] = hist["industrial_power"].iloc[-24] if len(hist) >= 24 else hist["industrial_power"].iloc[-1]
        combined.loc[i, "lag_168"] = hist["industrial_power"].iloc[-168] if len(hist) >= 168 else hist["industrial_power"].iloc[-1]
        combined.loc[i, "roll_mean_24"] = hist["industrial_power"].iloc[-24:].mean()
        
        # Time and environmental features
        t = combined.loc[i, "timestamp"]
        combined.loc[i, "hour_sin"] = np.sin(2 * np.pi * t.hour / 24)
        combined.loc[i, "hour_cos"] = np.cos(2 * np.pi * t.hour / 24)
        combined.loc[i, "day_of_week"] = t.dayofweek
        combined.loc[i, "is_weekend"] = 1 if t.dayofweek >= 5 else 0
        combined.loc[i, "temperature_c"] = hist["temperature_c"].iloc[-24:].mean()
        combined.loc[i, "humidity_pct"] = hist["humidity_pct"].iloc[-24:].mean()
        
        # Predict step
        pred = model.predict(combined.loc[[i], FEATURES])[0]
        combined.loc[i, "industrial_power"] = pred

    return combined.iloc[len(df):].copy()