import numpy as np
import pandas as pd

TARIFF_MYR_PER_MWH = 365.00  # Commercial grid energy rate tariff in MYR/MWh

def compute_metrics(actual, predicted):
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    # Currency and Risk Translation
    financial_exposure_myr = mae * TARIFF_MYR_PER_MWH
    grid_reserve_buffer_mwh = 1.96 * np.std(actual - predicted)
    
    return {
        "MAE (MWh)": round(mae, 2),
        "RMSE (MWh)": round(rmse, 2),
        "MAPE (%)": round(mape, 2),
        "MAE Financial Exposure (MYR/h)": round(financial_exposure_myr, 2),
        "95% Safety Reserve Buffer (MWh)": round(grid_reserve_buffer_mwh, 2)
    }
