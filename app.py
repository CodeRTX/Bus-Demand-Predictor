import streamlit as st
import joblib
import pandas as pd
import numpy as np
import datetime
import holidays
from sklearn.preprocessing import LabelEncoder
import os

st.set_page_config(page_title="Bus Demand Forecast", page_icon="🚌")

# Define paths to saved artifacts
models_dir = './models'
model_path = os.path.join(models_dir, 'lightgbm_model.pkl')
encoder_path = os.path.join(models_dir, 'city_encoder.joblib')

# Load the model and encoder
try:
    best_model = joblib.load(model_path)
    city_encoder = joblib.load(encoder_path)
except FileNotFoundError:
    st.error("Model or encoder files not found. Please ensure 'models' directory exists and contains 'lightgbm_model.joblib' and 'city_encoder.joblib'.")
    st.stop()

# Define 'PI_VAL' here
PI_VAL = np.pi

# Define the feature columns used during training
FEATURE_COLUMNS = ['day', 'month', 'is_weekend', 'is_holiday', 'cumsum_seatcount', 'cumsum_searchcount', 'route_month_avg', 'src_n', 'dest_n', 'dom', 'dom_sin', 'dom_cos',
                   'month_sin', 'month_cos', 'day_sin', 'day_cos']

# Specify country for holiday calendar; adjust as needed
holiday_country = 'IN'   # Example: 'IN' for India
holiday_calendar = holidays.CountryHoliday(holiday_country)

# Function to add date features and holiday flags (replicated from training script)
def add_date_features(df):
    df['day'] = df['doj'].dt.weekday  # 0=Monday
    df['month'] = df['doj'].dt.month  # Not zero-indexed!

    df['is_weekend'] = df['day'].isin([5, 6]).astype(int)
    # Holiday flag using 'python-holidays'
    df['is_holiday'] = df['doj'].dt.date.apply(lambda d: 1 if d in holiday_calendar else 0)
    df['dom'] = df['doj'].dt.day

    # add cyclic(periodic) features
    df['dom_sin'] = np.sin(2 * PI_VAL * df['dom'] / 31)
    df['dom_cos'] = np.cos(2 * PI_VAL * df['dom'] / 31)
    df['month_sin'] = np.sin(2 * PI_VAL * df['month'] / 12)
    df['month_cos'] = np.cos(2 * PI_VAL * df['month'] / 12)
    df['day_sin'] = np.sin(2 * PI_VAL * df['day'] / 7)
    df['day_cos'] = np.cos(2 * PI_VAL * df['day'] / 7)
    return df

# Streamlit App Title
st.title("Bus Demand Forecasting")

# Input widgets
st.header("Enter Route and Date Details")

doj_input = st.date_input("Date of Journey", datetime.date.today())
srcid_input = st.number_input("Source ID", min_value=1, max_value=48, value=1)
destid_input = st.number_input("Destination ID", min_value=1, max_value=48, value=2)

# Note: For simplicity, using placeholder/simplified logic for cumsum and route_month_avg.
# A production app would need a way to look up or compute these based on historical data up to doj-15.
# For this example, we will use static or simplified values.
# In a real scenario, you'd load precomputed summaries or query a database.
# For demonstration, let's use simple number inputs as a placeholder for demonstration
st.header("Enter Transaction-Based Features (Simplified Placeholders)")
cumsum_seatcount_input = st.number_input("Cumulative Seat Count (Placeholder)", min_value=0.0, value=100.0)
cumsum_searchcount_input = st.number_input("Cumulative Search Count (Placeholder)", min_value=0.0, value=1000.0)
route_month_avg_input = st.number_input("Route Month Average (Placeholder)", min_value=0.0, value=2000.0) # This should be looked up based on srcid, destid, and month


# Predict button
if st.button("Predict Demand"):
    # Create a DataFrame row from inputs
    input_data = pd.DataFrame({
        'doj': [pd.to_datetime(doj_input)],
        'srcid': [srcid_input],
        'destid': [destid_input],
        'cumsum_seatcount': [cumsum_seatcount_input],
        'cumsum_searchcount': [cumsum_searchcount_input],
        'route_month_avg': [route_month_avg_input]
    })

    # Apply date features
    input_data = add_date_features(input_data)

    # Apply city encoding
    try:
        input_data['src_n'] = city_encoder.transform(input_data['srcid'])
        input_data['dest_n'] = city_encoder.transform(input_data['destid'])
    except ValueError as e:
        st.error(f"Error encoding city IDs: {e}. Please ensure the entered srcid and destid are within the range of trained cities.")
        st.stop()


    # Ensure feature order matches training data
    X_input = input_data[FEATURE_COLUMNS]

    # Make prediction
    prediction = best_model.predict(X_input)[0]

    # Ensure prediction is non-negative and round
    prediction = max(0, prediction)
    prediction = round(prediction)

    # Display prediction
    st.success(f"Predicted Final Seat Count: {prediction}")
