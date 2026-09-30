from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Selangor Grid Energy Forecast - Industrial Engine",
    page_icon="⚡",
    layout="wide",
)

# Configuration Constants (Ringgit Malaysia)
TARIFF_MYR_PER_MWH = 365.00  # Commercial grid tariff rate in MYR/MWh
DATA_PATH = Path("NEWSelangor_Industrial_Electricity_2025_Dummy_Dataset.xlsx")
SHEET_NAME = "Hourly_2025_Data"


# 1. Data Preparation & Feature Engineering
@st.cache_data
def load_and_prep_data(file_path: Path) -> tuple[pd.DataFrame, str]:
    df_raw = pd.read_excel(file_path, sheet_name=SHEET_NAME)
    df_raw["Datetime"] = pd.to_datetime(df_raw["Datetime"])
    df_raw = df_raw.sort_values("Datetime").reset_index(drop=True)

    # Continuous hourly range alignment
    full_range = pd.date_range(
        start=df_raw["Datetime"].min(),
        end=df_raw["Datetime"].max(),
        freq="1h",
    )
    df = (
        df_raw.set_index("Datetime")
        .reindex(full_range)
        .rename_axis("Datetime")
        .reset_index()
    )

    # Safe Imputation
    target_col = "Industrial_Consumption_MWh"
    df[target_col] = (
        df[target_col].ffill(limit=2).interpolate(method="linear")
    )
    df["Temperature_C"] = (
        df["Temperature_C"].ffill(limit=2).interpolate(method="linear")
    )
    df["Humidity_%"] = (
        df["Humidity_%"].ffill(limit=2).interpolate(method="linear")
    )

    # Time-based Deterministic Features
    df["Hour"] = df["Datetime"].dt.hour
    df["DayOfWeek"] = df["Datetime"].dt.dayofweek
    df["Month"] = df["Datetime"].dt.month
    df["IsWeekend"] = df["DayOfWeek"].apply(lambda x: 1 if x >= 5 else 0)

    # Cyclic Embeddings
    df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24.0)
    df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24.0)
    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12.0)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12.0)

    # Lags
    for lag in [24, 48, 168]:
        df[f"lag_{lag}"] = df[target_col].shift(lag)

    # Rolling Statistics
    df["rolling_mean_24_lag24"] = (
        df[target_col].shift(24).rolling(window=24).mean()
    )
    df["rolling_std_24_lag24"] = (
        df[target_col].shift(24).rolling(window=24).std()
    )
    df["rolling_mean_168_lag24"] = (
        df[target_col].shift(24).rolling(window=168).mean()
    )

    return df.dropna().reset_index(drop=True), target_col


def compute_metrics(actual, predicted, tariff=TARIFF_MYR_PER_MWH):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    financial_exposure_myr = mae * tariff
    safety_buffer_mwh = 1.5 * rmse

    return {
        "MAE (MWh)": round(mae, 2),
        "RMSE (MWh)": round(rmse, 2),
        "MAPE (%)": round(mape, 2),
        "Financial Exposure (MYR/h)": round(financial_exposure_myr, 2),
        "Safety Buffer (MWh)": round(safety_buffer_mwh, 2),
    }


@st.cache_resource
def train_gbdt_model(X_tr, y_tr):
    gbdt = HistGradientBoostingRegressor(
        max_iter=250, learning_rate=0.05, random_state=42
    )
    gbdt.fit(X_tr, y_tr)
    return gbdt


# 2. Multi-Step Iterative Dynamic Forecasting (30 Days / 720 Hours)
def generate_30day_forecast(
    df_history: pd.DataFrame,
    model,
    feature_cols: list[str],
    target_col: str,
    days: int = 30,
) -> pd.DataFrame:
    hours_to_predict = days * 24
    history_df = df_history.copy().sort_values("Datetime").reset_index(drop=True)
    last_timestamp = history_df["Datetime"].iloc[-1]

    # Generate future hourly dates
    future_dates = [
        last_timestamp + pd.Timedelta(hours=i) for i in range(1, hours_to_predict + 1)
    ]
    forecast_results = []

    for dt in future_dates:
        row = {"Datetime": dt}

        # Deterministic Time Features
        row["Hour"] = dt.hour
        row["DayOfWeek"] = dt.dayofweek
        row["Month"] = dt.month
        row["IsWeekend"] = 1 if dt.dayofweek >= 5 else 0

        # Cyclic Embeddings
        row["Hour_sin"] = np.sin(2 * np.pi * row["Hour"] / 24.0)
        row["Hour_cos"] = np.cos(2 * np.pi * row["Hour"] / 24.0)
        row["Month_sin"] = np.sin(2 * np.pi * row["Month"] / 12.0)
        row["Month_cos"] = np.cos(2 * np.pi * row["Month"] / 12.0)

        # Temperature & Humidity Exogenous Features
        recent_temp = history_df[history_df["Hour"] == dt.hour]["Temperature_C"]
        recent_hum = history_df[history_df["Hour"] == dt.hour]["Humidity_%"]
        row["Temperature_C"] = (
            recent_temp.tail(14).mean() if len(recent_temp) > 0 else 28.0
        )
        row["Humidity_%"] = (
            recent_hum.tail(14).mean() if len(recent_hum) > 0 else 75.0
        )

        # Lag Features calculated dynamically
        y_series = history_df[target_col]
        row["lag_24"] = y_series.iloc[-24]
        row["lag_48"] = y_series.iloc[-48]
        row["lag_168"] = y_series.iloc[-168]

        # Dynamic Rolling Statistics
        lag24_window_24 = y_series.iloc[-48:-24]
        lag24_window_168 = y_series.iloc[-192:-24]
        row["rolling_mean_24_lag24"] = lag24_window_24.mean()
        row["rolling_std_24_lag24"] = lag24_window_24.std()
        row["rolling_mean_168_lag24"] = lag24_window_168.mean()

        # Feature Extraction & Model Inference
        X_curr = pd.DataFrame([row])[feature_cols]
        pred_val = model.predict(X_curr)[0]
        row[target_col] = pred_val

        # Append step back into history to calculate upcoming lags recursively
        history_df = pd.concat(
            [history_df, pd.DataFrame([row])], ignore_index=True
        )
        forecast_results.append(row)

    df_forecast = pd.DataFrame(forecast_results)
    return df_forecast[["Datetime", target_col, "Hour", "DayOfWeek"]]


# Sidebar Data Loader
uploaded_file = st.sidebar.file_uploader("Upload Excel Dataset", type=["xlsx"])
current_path = uploaded_file if uploaded_file else DATA_PATH

try:
    df, target_col = load_and_prep_data(current_path)

    # 3. LEAKAGE-FREE VALIDATION & FEATURE ENGINEERING (70 / 20 / 10 Split)
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.90)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    feature_cols = [
        c
        for c in train_df.columns
        if c not in ["Datetime", "Date", "Day", "Peak_Period", target_col]
    ]

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_val, y_val = val_df[feature_cols], val_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    # Train GBDT Model
    gbdt_model = train_gbdt_model(X_train, y_train)

    # Evaluate on Test Set
    test_df["pred_snaive_24"] = test_df["lag_24"]
    test_df["pred_snaive_168"] = test_df["lag_168"]
    test_df["pred_gbdt"] = gbdt_model.predict(X_test)

    m_snaive_24 = compute_metrics(y_test, test_df["pred_snaive_24"])
    m_snaive_168 = compute_metrics(y_test, test_df["pred_snaive_168"])
    m_gbdt = compute_metrics(y_test, test_df["pred_gbdt"])

    # Header
    st.title(
        "⚡ Selangor Industrial Energy Demand: Operations & Forecast Engine"
    )

    # Sidebar Project Framing Summary
    st.sidebar.header("📋 Project Framing")
    st.sidebar.markdown(f"""
    - **Target ($y$)**: `{target_col}` (MWh)
    - **Train Set (70%)**: {len(train_df):,} hours
    - **Val Set (20%)**: {len(val_df):,} hours
    - **Test Set (10%)**: {len(test_df):,} hours
    - **Tariff Rate**: RM {TARIFF_MYR_PER_MWH:.2f} / MWh
    """)

    # Top Metric Banner (Baseline vs GBDT)
    st.markdown("### 📊 Model Performance: Baseline vs GBDT")

    gbdt_mae_diff = round(m_snaive_24["MAE (MWh)"] - m_gbdt["MAE (MWh)"], 2)
    gbdt_exp_diff = round(
        m_snaive_24["Financial Exposure (MYR/h)"] - m_gbdt["Financial Exposure (MYR/h)"], 2
    )
    gbdt_annual_savings = gbdt_mae_diff * 8760 * TARIFF_MYR_PER_MWH

    # Row 1: Baseline Metrics
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Baseline MAE (24h Lag)", f"{m_snaive_24['MAE (MWh)']} MWh")
    b2.metric("Baseline Exposure Risk", f"RM {m_snaive_24['Financial Exposure (MYR/h)']}")
    b3.metric("Baseline Safety Buffer", f"+{m_snaive_24['Safety Buffer (MWh)']} MWh")
    b4.metric("Baseline Annual Exposure", f"RM {m_snaive_24['Financial Exposure (MYR/h)'] * 8760:,.2f}")

    # Row 2: GBDT Metrics
    g1, g2, g3, g4 = st.columns(4)
    g1.metric(
        "GBDT MAE",
        f"{m_gbdt['MAE (MWh)']} MWh",
        delta=f"-{gbdt_mae_diff} MWh vs Baseline",
        delta_color="inverse",
    )
    g2.metric(
        "GBDT Exposure Risk",
        f"RM {m_gbdt['Financial Exposure (MYR/h)']}",
        delta=f"-RM {gbdt_exp_diff} / hr",
        delta_color="inverse",
    )
    g3.metric(
        "GBDT Safety Buffer",
        f"+{m_gbdt['Safety Buffer (MWh)']} MWh",
        delta="1.5x RMSE Rule",
        delta_color="off",
    )
    g4.metric(
        "GBDT Annual Savings",
        f"RM {gbdt_annual_savings:,.2f}",
        delta="Grid Optimization",
        delta_color="normal",
    )

    st.divider()

    # Dashboard Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🚀 24-Hour Operations Dispatch",
        "📅 30-Day Energy Projection",
        "⚔️ Model Scorecard",
        "📈 Historical Data & Buffer Analysis",
    ])

    with tab1:
        st.subheader("24-Hour Ahead Operations Schedule")

        # Sub-tabs for 1 January 2026 and 12 September 2026
        sub_tab_jan, sub_tab_sep = st.tabs([
            "📅 1 January 2026",
            "📅 12 September 2026",
        ])

        with sub_tab_jan:
            st.markdown("### GBDT Model Prediction: 1 January 2026")

            # Generate 30-day forecast using GBDT model
            df_future = generate_30day_forecast(
                df_history=df,
                model=gbdt_model,
                feature_cols=feature_cols,
                target_col=target_col,
                days=30,
            )

            # Extract 1 January 2026 and sort ascending by hour
            jan1_df = (
                df_future[df_future["Datetime"].dt.date == pd.to_datetime("2026-01-01").date()]
                .sort_values("Hour", ascending=True)
                .reset_index(drop=True)
            )

            if not jan1_df.empty:
                jan1_df["Predicted_Industrial_Consumption_MWh"] = round(jan1_df[target_col], 2)
                jan1_df["Safety_Buffer_MWh"] = round(m_gbdt["Safety Buffer (MWh)"], 2)
                jan1_df["Max_Dispatch_Commitment_MWh"] = round(
                    jan1_df["Predicted_Industrial_Consumption_MWh"] + jan1_df["Safety_Buffer_MWh"],
                    2,
                )

                fig_jan = go.Figure()
                fig_jan.add_trace(
                    go.Scatter(
                        x=jan1_df["Hour"],
                        y=jan1_df["Predicted_Industrial_Consumption_MWh"],
                        name="Predicted Industrial Demand (GBDT)",
                        line=dict(color="#10b981", width=2),
                    )
                )
                fig_jan.add_trace(
                    go.Scatter(
                        x=jan1_df["Hour"],
                        y=jan1_df["Max_Dispatch_Commitment_MWh"],
                        name=f"Max Commitment (+{m_gbdt['Safety Buffer (MWh)']} MWh Buffer)",
                        line=dict(color="#f43f5e", dash="dash"),
                    )
                )
                fig_jan.update_layout(
                    title="1 January 2026 Hourly Demand Schedule",
                    xaxis_title="Hour of Day (0 - 23)",
                    yaxis_title="Demand (MWh)",
                    hovermode="x unified",
                    template="plotly_white",
                    height=400,
                )
                st.plotly_chart(fig_jan, use_container_width=True)

                cols_to_show = [
                    "Hour",
                    "Datetime",
                    "Predicted_Industrial_Consumption_MWh",
                    "Safety_Buffer_MWh",
                    "Max_Dispatch_Commitment_MWh",
                ]
                st.dataframe(jan1_df[cols_to_show], use_container_width=True)
            else:
                st.warning("1 January 2026 prediction is not within the forecast horizon.")

        with sub_tab_sep:
            st.markdown("### Sheet Forecast: 12 September 2026")
            try:
                forecast_sheet = pd.read_excel(
                    current_path, sheet_name="Forecast_12Sep2026"
                )
                forecast_sheet["Safety_Buffer_MWh"] = round(
                    m_gbdt["Safety Buffer (MWh)"], 2
                )
                forecast_sheet["Max_Dispatch_Commitment_MWh"] = round(
                    forecast_sheet["Predicted_Industrial_Consumption_MWh"]
                    + m_gbdt["Safety Buffer (MWh)"],
                    2,
                )

                fig1 = go.Figure()
                fig1.add_trace(
                    go.Scatter(
                        x=forecast_sheet["Hour"],
                        y=forecast_sheet["Predicted_Industrial_Consumption_MWh"],
                        name="Predicted Industrial Demand",
                        line=dict(color="#10b981", width=2),
                    )
                )
                fig1.add_trace(
                    go.Scatter(
                        x=forecast_sheet["Hour"],
                        y=forecast_sheet["Max_Dispatch_Commitment_MWh"],
                        name=f"Max Commitment (+{m_gbdt['Safety Buffer (MWh)']} MWh Buffer)",
                        line=dict(color="#f43f5e", dash="dash"),
                    )
                )
                fig1.update_layout(
                    xaxis_title="Hour of Day",
                    yaxis_title="Demand (MWh)",
                    hovermode="x unified",
                    template="plotly_white",
                    height=400,
                )
                st.plotly_chart(fig1, use_container_width=True)
                st.dataframe(forecast_sheet, use_container_width=True)
            except Exception:
                st.info(
                    "Upload a dataset containing the `Forecast_12Sep2026` sheet to visualize the schedule."
                )

    with tab2:
        st.subheader("📅 30-Day Ahead Energy Consumption Demand Forecast (GBDT)")

        with st.spinner("Calculating dynamic 30-day (720-hour) projection..."):
            df_30day_forecast = generate_30day_forecast(
                df_history=df,
                model=gbdt_model,
                feature_cols=feature_cols,
                target_col=target_col,
                days=30,
            )

            df_30day_forecast["Safety_Buffer_MWh"] = m_gbdt["Safety Buffer (MWh)"]
            df_30day_forecast["Upper_Commitment_MWh"] = (
                df_30day_forecast[target_col]
                + df_30day_forecast["Safety_Buffer_MWh"]
            )

        total_mwh = df_30day_forecast[target_col].sum()
        total_cost = total_mwh * TARIFF_MYR_PER_MWH
        daily_avg_mwh = total_mwh / 30

        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("30-Day Total Energy Demand", f"{total_mwh:,.2f} MWh")
        kpi2.metric("Daily Average Demand", f"{daily_avg_mwh:,.2f} MWh/day")
        kpi3.metric("30-Day Forecasted Cost", f"RM {total_cost:,.2f}")

        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=df_30day_forecast["Datetime"],
                y=df_30day_forecast[target_col],
                name="30-Day Predicted Consumption (GBDT)",
                line=dict(color="#2563eb", width=1.5),
            )
        )
        fig2.add_trace(
            go.Scatter(
                x=df_30day_forecast["Datetime"],
                y=df_30day_forecast["Upper_Commitment_MWh"],
                name="Upper Reserve Limit (+1.5x RMSE)",
                line=dict(color="#dc2626", dash="dot", width=1),
            )
        )
        fig2.update_layout(
            xaxis_title="Timeline",
            yaxis_title="Demand (MWh)",
            hovermode="x unified",
            template="plotly_white",
            height=450,
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("30-Day Hourly Table")
        st.dataframe(df_30day_forecast, use_container_width=True)

    with tab3:
        st.subheader("⚔️ Model Scorecard & Benchmark Matrix")

        # 1. Evaluate GBDT multi-step performance over a 7-day (168h) horizon on Test Data
        test_start_dt = test_df["Datetime"].iloc[0]
        test_7day_actual = test_df.iloc[:168]

        # Generate recursive 7-day GBDT prediction from test set history start
        df_hist_test = df.iloc[:val_end].copy()
        gbdt_7day_pred_df = generate_30day_forecast(
            df_history=df_hist_test,
            model=gbdt_model,
            feature_cols=feature_cols,
            target_col=target_col,
            days=7,
        )

        m_gbdt_7day = compute_metrics(
            test_7day_actual[target_col].values,
            gbdt_7day_pred_df[target_col].values,
        )

        # 2. Build Updated Scorecard DataFrame
        scorecard = pd.DataFrame(
            [m_snaive_24, m_gbdt],
            index=[
                "Seasonal Naive (24h Lag - 1-Day Baseline)",
               
                "GBDT Model (1-Day Operational Evaluation)",
               
            ],
        )

        # 3. Display Scorecard Table
        st.dataframe(
            scorecard.style.highlight_min(
                axis=0,
                color="#d1fae5",
                subset=["MAE (MWh)", "RMSE (MWh)", "MAPE (%)"],
            ),
            use_container_width=True,
        )


# 2. Add Tab 4 Implementation
    with tab4:
        st.subheader("📈 Historical Consumption & Reserve Buffer Percentage")

        # Compute dynamic safety buffer percentage for the imported dataset
        df_hist_analysis = df.copy()
        safety_buffer_val = m_gbdt["Safety Buffer (MWh)"]
        
        df_hist_analysis["Upper_Reserve_MWh"] = (
            df_hist_analysis[target_col] + safety_buffer_val
        )
        df_hist_analysis["Buffer_Percentage"] = (
            (safety_buffer_val / df_hist_analysis[target_col]) * 100
        )

        # Overview Metrics
        avg_buffer_pct = df_hist_analysis["Buffer_Percentage"].mean()
        min_buffer_pct = df_hist_analysis["Buffer_Percentage"].min()
        max_buffer_pct = df_hist_analysis["Buffer_Percentage"].max()

        h1, h2, h3 = st.columns(3)
        h1.metric("Average Buffer (% of Load)", f"{avg_buffer_pct:.2f}%")
        h2.metric("Min Buffer Margin", f"{min_buffer_pct:.2f}%")
        h3.metric("Max Buffer Margin", f"{max_buffer_pct:.2f}%")

        # Dual-Axis Plotly Visualization
        fig_hist = go.Figure()

        # Primary Y-Axis: MWh Consumption & Reserve Limit
        fig_hist.add_trace(
            go.Scatter(
                x=df_hist_analysis["Datetime"],
                y=df_hist_analysis[target_col],
                name="Actual Industrial Demand (MWh)",
                line=dict(color="#0284c7", width=1.5),
            )
        )
        fig_hist.add_trace(
            go.Scatter(
                x=df_hist_analysis["Datetime"],
                y=df_hist_analysis["Upper_Reserve_MWh"],
                name="Upper Commitment Limit (+1.5x RMSE)",
                line=dict(color="#ef4444", dash="dash", width=1),
            )
        )

        # Secondary Y-Axis: Buffer Percentage
        fig_hist.add_trace(
            go.Scatter(
                x=df_hist_analysis["Datetime"],
                y=df_hist_analysis["Buffer_Percentage"],
                name="Buffer Ratio (%)",
                yaxis="y2",
                line=dict(color="#f59e0b", width=1),
                opacity=0.6,
            )
        )

        # Layout Configuration with Dual Axes
        fig_hist.update_layout(
            title="Historical Energy Demand vs. Required Buffer Percentage",
            xaxis=dict(title="Timeline"),
            yaxis=dict(title="Energy Demand (MWh)", side="left"),
            yaxis2=dict(
                title="Buffer Margin (%)",
                side="right",
                overlaying="y",
                showgrid=False,
                ticksuffix="%",
            ),
            hovermode="x unified",
            template="plotly_white",
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        st.plotly_chart(fig_hist, use_container_width=True)

        # Data Table View
        with st.expander("🔍 View Detailed Historical Buffer Breakdown Table"):
            st.dataframe(
                df_hist_analysis[
                    [
                        "Datetime",
                        target_col,
                        "Upper_Reserve_MWh",
                        "Buffer_Percentage",
                    ]
                ],
                use_container_width=True,
            )
except Exception as e:
    st.error(f"Error loading dataset or calculating forecast: {e}")