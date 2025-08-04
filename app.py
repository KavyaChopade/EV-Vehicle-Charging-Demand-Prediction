import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt

# ==== Page Configuration ====
st.set_page_config(
    page_title="EV Adoption Forecast",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==== Custom CSS for Dark Theme ====
st.markdown("""
<style>
.stApp {
    background-color: #121212;
    color: #f0f0f0;
}
h1, h2, h3, h4, h5, h6 {
    color: #ffffff;
}
div[data-testid="stSidebar"] {
    background-color: #1e1e1e;
}
.stSelectbox > div {
    background-color: #2b2b2b !important;
}
</style>
""", unsafe_allow_html=True)

# ==== Load Model with Error Handling ====
try:
    model = joblib.load("forecasting_ev_model (1).pkl")
except Exception as e:
    st.error("❌ Model loading failed. Please check the .pkl file.")
    st.stop()

# ==== Load Data with Caching ====
@st.cache_data
def load_data():
    df = pd.read_csv("preprocessed_ev_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

try:
    df = load_data()
except Exception as e:
    st.error("❌ Data loading failed. Please check the CSV file.")
    st.stop()

# ==== Enhanced Sidebar ====
st.sidebar.title("🔧 Forecast Controls")

county_list = sorted(df['County'].dropna().unique().tolist())
county = st.sidebar.selectbox("Select a County", county_list)

# User-controlled forecast range
forecast_horizon = st.sidebar.slider("Forecast Horizon (Months)", 12, 60, 36, step=12)

# Model info
with st.sidebar.expander("📊 Model Info"):
    st.markdown("""
    - **Model**: Random Forest Regressor  
    - **Version**: 1.0  
    - **Trained on**: Washington County EV Data  
    - **Last Data Update**: August 2025
    """)

# Help section
with st.sidebar.expander("❓ How this works"):
    st.markdown("""
    This app forecasts EV adoption trends using a trained machine learning model.
    
    **Steps**:
    - Select a county
    - View 3–5 year forecasts
    - Compare up to 3 counties
    - Download results
    """)

# Feedback placeholder
st.sidebar.markdown("---")
st.sidebar.markdown("📫 Have suggestions?")
st.sidebar.text_input("Type here...", placeholder="Your feedback")

# ==== Title Section ====
st.markdown("""
<div style='text-align: center;'>
    <img src='https://cdn-icons-png.flaticon.com/512/2630/2630547.png' width='80'>
    <h1 style='margin-bottom: 0;'>EV Adoption Forecasting</h1>
    <p style='font-size: 18px; color: #ccc;'>Forecasting Electric Vehicle (EV) growth across Washington State counties over the next few years</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ==== Prepare Selected County Data ====
county_df = df[df['County'] == county].sort_values("Date")
if county_df.empty:
    st.warning(f"No data found for {county}")
    st.stop()

county_code = county_df['county_encoded'].iloc[0]
historical_ev = list(county_df['Electric Vehicle (EV) Total'].values[-6:])
cumulative_ev = list(np.cumsum(historical_ev))
months_since_start = county_df['months_since_start'].max()
latest_date = county_df['Date'].max()

# ==== Forecasting ====
future_rows = []

for i in range(1, forecast_horizon + 1):
    forecast_date = latest_date + pd.DateOffset(months=i)
    months_since_start += 1
    lag1, lag2, lag3 = historical_ev[-1], historical_ev[-2], historical_ev[-3]
    roll_mean = np.mean([lag1, lag2, lag3])
    pct_change_1 = (lag1 - lag2) / lag2 if lag2 != 0 else 0
    pct_change_3 = (lag1 - lag3) / lag3 if lag3 != 0 else 0
    slope = np.polyfit(range(len(cumulative_ev[-6:])), cumulative_ev[-6:], 1)[0]

    new_row = {
        'months_since_start': months_since_start,
        'county_encoded': county_code,
        'ev_total_lag1': lag1,
        'ev_total_lag2': lag2,
        'ev_total_lag3': lag3,
        'ev_total_roll_mean_3': roll_mean,
        'ev_total_pct_change_1': pct_change_1,
        'ev_total_pct_change_3': pct_change_3,
        'ev_growth_slope': slope
    }

    pred = model.predict(pd.DataFrame([new_row]))[0]
    future_rows.append({"Date": forecast_date, "Predicted EV Total": round(pred)})

    historical_ev.append(pred)
    if len(historical_ev) > 6:
        historical_ev.pop(0)
    cumulative_ev.append(cumulative_ev[-1] + pred)
    if len(cumulative_ev) > 6:
        cumulative_ev.pop(0)

# ==== Combine Historical & Forecast ====
historical_cum = county_df[['Date', 'Electric Vehicle (EV) Total']].copy()
historical_cum['Cumulative EV'] = historical_cum['Electric Vehicle (EV) Total'].cumsum()
historical_cum['Source'] = 'Historical'

forecast_df = pd.DataFrame(future_rows)
forecast_df['Cumulative EV'] = forecast_df['Predicted EV Total'].cumsum() + historical_cum['Cumulative EV'].iloc[-1]
forecast_df['Source'] = 'Forecast'

combined = pd.concat([
    historical_cum[['Date', 'Cumulative EV', 'Source']],
    forecast_df[['Date', 'Cumulative EV', 'Source']]
], ignore_index=True)

# ==== Plotting ====
st.subheader(f"📊 Cumulative EV Forecast in {county} County")
fig, ax = plt.subplots(figsize=(12, 6))
for source, group in combined.groupby("Source"):
    ax.plot(group["Date"], group["Cumulative EV"], label=source, marker="o")
ax.set_title(f"Cumulative EV Forecast in {county}", color='white')
ax.set_xlabel("Date", color='white')
ax.set_ylabel("Cumulative EVs", color='white')
ax.grid(alpha=0.3)
ax.set_facecolor("#1e1e1e")
fig.patch.set_facecolor("#121212")
ax.tick_params(colors='white')
ax.legend()
st.pyplot(fig)

# ==== Forecast Summary ====
historical_total = historical_cum['Cumulative EV'].iloc[-1]
forecasted_total = forecast_df['Cumulative EV'].iloc[-1]
if historical_total > 0:
    growth_pct = ((forecasted_total - historical_total) / historical_total) * 100
    trend = "increase 📈" if growth_pct > 0 else "decrease 📉"
    st.success(f"In **{county}**, EV adoption is projected to show a **{trend} of {growth_pct:.2f}%** over the next {forecast_horizon} months.")
else:
    st.warning("⚠️ Insufficient historical EV data for trend analysis.")

# ==== County Comparison ====
st.markdown("---")
st.subheader("📍 Compare Up to 3 Counties")
multi_counties = st.multiselect("Select Counties to Compare", county_list, max_selections=3)

if multi_counties:
    comp_data = []
    for cty in multi_counties:
        df_cty = df[df['County'] == cty].sort_values("Date")
        if df_cty.empty:
            continue

        cty_code = df_cty['county_encoded'].iloc[0]
        hist_ev = list(df_cty['Electric Vehicle (EV) Total'].values[-6:])
        cum_ev = list(np.cumsum(hist_ev))
        months_since = df_cty['months_since_start'].max()
        last_date = df_cty['Date'].max()
        pred_rows = []

        for i in range(1, forecast_horizon + 1):
            months_since += 1
            forecast_date = last_date + pd.DateOffset(months=i)
            lag1, lag2, lag3 = hist_ev[-1], hist_ev[-2], hist_ev[-3]
            roll_mean = np.mean([lag1, lag2, lag3])
            pct1 = (lag1 - lag2) / lag2 if lag2 != 0 else 0
            pct3 = (lag1 - lag3) / lag3 if lag3 != 0 else 0
            slope = np.polyfit(range(len(cum_ev[-6:])), cum_ev[-6:], 1)[0]

            input_row = {
                'months_since_start': months_since,
                'county_encoded': cty_code,
                'ev_total_lag1': lag1,
                'ev_total_lag2': lag2,
                'ev_total_lag3': lag3,
                'ev_total_roll_mean_3': roll_mean,
                'ev_total_pct_change_1': pct1,
                'ev_total_pct_change_3': pct3,
                'ev_growth_slope': slope
            }
            pred = model.predict(pd.DataFrame([input_row]))[0]
            pred_rows.append({"Date": forecast_date, "Predicted EV Total": round(pred)})

            hist_ev.append(pred)
            if len(hist_ev) > 6:
                hist_ev.pop(0)
            cum_ev.append(cum_ev[-1] + pred)
            if len(cum_ev) > 6:
                cum_ev.pop(0)

        hist_df = df_cty[['Date', 'Electric Vehicle (EV) Total']].copy()
        hist_df['Cumulative EV'] = hist_df['Electric Vehicle (EV) Total'].cumsum()

        fc_df = pd.DataFrame(pred_rows)
        fc_df['Cumulative EV'] = fc_df['Predicted EV Total'].cumsum() + hist_df['Cumulative EV'].iloc[-1]
        combined_cty = pd.concat([
            hist_df[['Date', 'Cumulative EV']],
            fc_df[['Date', 'Cumulative EV']]
        ])
        combined_cty['County'] = cty
        comp_data.append(combined_cty)

    if comp_data:
        result_df = pd.concat(comp_data)

        fig, ax = plt.subplots(figsize=(14, 7))
        for cty, group in result_df.groupby("County"):
            ax.plot(group["Date"], group["Cumulative EV"], label=cty, marker="o")
        ax.set_title("County-wise EV Adoption Comparison", color='white')
        ax.set_xlabel("Date", color='white')
        ax.set_ylabel("Cumulative EVs", color='white')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("#1e1e1e")
        fig.patch.set_facecolor("#121212")
        ax.tick_params(colors='white')
        ax.legend()
        st.pyplot(fig)

        growth_lines = []
        for cty in multi_counties:
            cdf = result_df[result_df['County'] == cty].reset_index(drop=True)
            past_val = cdf['Cumulative EV'].iloc[len(cdf) - forecast_horizon - 1]
            future_val = cdf['Cumulative EV'].iloc[-1]
            growth = ((future_val - past_val) / past_val) * 100 if past_val > 0 else 0
            growth_lines.append(f"{cty}: {growth:.2f}%")

        st.success("📈 Forecasted EV growth: " + " | ".join(growth_lines))

# ==== Download Forecast Button ====
st.markdown("---")
csv_bytes = forecast_df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Forecast Data (CSV)", csv_bytes, "ev_forecast.csv", "text/csv")

# ==== Footer ====
st.markdown("""
<div style='text-align: center; color: #999; margin-top: 2em;'>
    Built during <b>AICTE Internship</b> | Developed by <b>Kavya Chopade</b> 🚗🔋
</div>
""", unsafe_allow_html=True)
