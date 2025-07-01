import streamlit as st
import joblib
import pandas as pd
import numpy as np
import datetime
import holidays
from sklearn.preprocessing import LabelEncoder
import os
import plotly.graph_objects as go

# --- 1. Page Configuration & Styling ---
st.set_page_config(
    page_title="Bus Demand Forecaster",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_css():
    """Injects custom CSS for a high-contrast, professional dark theme."""
    st.markdown("""
    <style>
    /* --- Base & Background --- */
    .stApp {
        background: linear-gradient(135deg, #1d2b4e 0%, #0c1021 100%);
        color: #f0f0f0; /* Default light text color */
    }

    /* --- Typography --- */
    .title-text {
        font-family: 'Arial Black', Gadget, sans-serif;
        font-size: 3.2rem !important;
        font-weight: bold;
        color: #ffffff;
        text-align: center;
        padding-top: 20px;
        text-shadow: 0 0 10px #00f5c9, 0 0 20px rgba(0, 245, 201, 0.5);
    }
    .subtitle-text {
        text-align: center;
        color: #b0c4de; /* Light Steel Blue for subtitles */
        font-size: 1.1rem;
    }

    /* --- Sidebar --- */
    [data-testid="stSidebar"] {
        background-color: #111827; /* Dark Gray-Blue */
        border-right: 1px solid #334155;
    }

    /* --- Buttons --- */
    .stButton>button {
        width: 100%;
        border: 2px solid #00f5c9; /* Vibrant Cyan/Teal */
        border-radius: 25px;
        color: #00f5c9;
        background-color: transparent;
        padding: 12px 28px;
        font-weight: bold;
        transition: all 0.3s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #00f5c9;
        color: #111827; /* Dark text on hover */
        transform: scale(1.02);
        box-shadow: 0 0 15px rgba(0, 245, 201, 0.7);
    }

    /* --- Metrics & Containers --- */
    [data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-left: 6px solid #00f5c9;
        padding: 20px;
        border-radius: 10px;
    }
    
    /* --- Expanders --- */
    .st-expander {
        background-color: #1e293b; /* Darker Slate Blue */
        border: 1px solid #334155;
        border-radius: 10px;
    }
    div[data-testid="stExpanderHeader"] p {
        font-size: 1.1rem;
        color: #f0f0f0;
        font-weight: bold;
    }
    
    /* --- Input Labels --- */
    label {
        color: #d1d5db !important;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

load_css()


# --- 2. Load Artifacts & Define Constants ---
MODELS_DIR = './models'
MODEL_PATH = os.path.join(MODELS_DIR, 'lightgbm_model.pkl')
ENCODER_PATH = os.path.join(MODELS_DIR, 'city_encoder.joblib')

@st.cache_resource
def load_model_and_encoder():
    try:
        model = joblib.load(MODEL_PATH)
        encoder = joblib.load(ENCODER_PATH)
        return model, encoder
    except FileNotFoundError:
        st.error(f"Model/encoder not found. Please run 'source.ipynb' to generate artifacts in '{MODELS_DIR}'.")
        st.stop()

best_model, city_encoder = load_model_and_encoder()

PI_VAL = np.pi
HOLIDAY_COUNTRY = 'IN'
holiday_calendar = holidays.CountryHoliday(HOLIDAY_COUNTRY)
FEATURE_COLUMNS = [
    'day', 'month', 'is_weekend', 'is_holiday', 'cumsum_seatcount', 'cumsum_searchcount', 
    'route_month_avg', 'src_n', 'dest_n', 'dom', 'dom_sin', 'dom_cos', 'month_sin', 
    'month_cos', 'day_sin', 'day_cos'
]

# --- 3. Helper Functions ---
def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    # (Function content is unchanged)
    df['day'] = df['doj'].dt.weekday
    df['month'] = df['doj'].dt.month
    df['is_weekend'] = df['day'].isin([5, 6]).astype(int)
    df['is_holiday'] = df['doj'].dt.date.apply(lambda d: 1 if d in holiday_calendar else 0)
    df['dom'] = df['doj'].dt.day
    df['dom_sin'] = np.sin(2 * PI_VAL * df['dom'] / 31)
    df['dom_cos'] = np.cos(2 * PI_VAL * df['dom'] / 31)
    df['month_sin'] = np.sin(2 * PI_VAL * df['month'] / 12)
    df['month_cos'] = np.cos(2 * PI_VAL * df['month'] / 12)
    df['day_sin'] = np.sin(2 * PI_VAL * df['day'] / 7)
    df['day_cos'] = np.cos(2 * PI_VAL * df['day'] / 7)
    return df

def create_gauge_chart(value: int, title: str) -> go.Figure:
    """Creates a gauge chart with tiered ranges for better visualization."""
    # Determine the gauge's maximum range based on the prediction value
    if value <= 100:
        gauge_range = [0, 150]
    elif value <= 500:
        gauge_range = [0, 600]
    elif value <= 2000:
        gauge_range = [0, 2500]
    else:
        gauge_range = [0, value + 500] # Dynamic for very high values

    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        title = {'text': title, 'font': {'size': 20, 'color': '#f0f0f0'}},
        number = {'font': {'color': '#00f5c9', 'size': 48}},
        gauge = {
            'axis': {'range': gauge_range, 'tickwidth': 1, 'tickcolor': "#b0c4de"},
            'bar': {'color': "#00f5c9"},
            'bgcolor': "rgba(0,0,0,0.2)",
            'borderwidth': 2,
            'bordercolor': "#334155",
            'steps' : [
                {'range': [gauge_range[0], gauge_range[1]*0.33], 'color': '#1e4242'},
                {'range': [gauge_range[1]*0.33, gauge_range[1]*0.66], 'color': '#2a5a5b'},
                {'range': [gauge_range[1]*0.66, gauge_range[1]], 'color': '#007b7f'}],
        }))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        height=300, 
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

# --- 4. Sidebar for User Inputs ---
with st.sidebar:
    # Using a valid, theme-appropriate image link
    st.image("https://i.imgur.com/8v7Dudi.png", use_container_width=True)    # The image goes here
    st.header("⚙️ Prediction Inputs")
    st.info("Enter route and date to forecast the final seat count 15 days in advance.")
    doj_input = st.date_input("Date of Journey", datetime.date.today() + datetime.timedelta(days=15))
    srcid_input = st.number_input("Source City ID", min_value=1, max_value=48, value=1)
    destid_input = st.number_input("Destination City ID", min_value=1, max_value=48, value=2)
    st.divider()
    st.subheader("📈 Advanced Features")
    st.warning(
        "**For Demo Purposes:** These values are pre-filled with realistic medians from the training data. You can adjust them to see the impact."
    )
    # Using data-driven medians for more realistic defaults
    cumsum_seatcount_input = st.number_input("Cumulative Seat Count", min_value=0.0, value=550.0)
    cumsum_searchcount_input = st.number_input("Cumulative Search Count", min_value=0.0, value=8500.0)
    route_month_avg_input = st.number_input("Historical Route-Month Average", min_value=0.0, value=1650.0)

# --- 5. Main Page ---
st.markdown('<p class="title-text">Bus Demand Forecaster</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Predicting future bus seat demand using a high-performance LightGBM model.</p>', unsafe_allow_html=True)
st.divider()

if 'prediction' not in st.session_state:
    st.session_state.prediction = None

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Prediction Control")
    if st.button("🚀 Predict Demand"):
        # SPECIAL CASE: If source and destination are the same, demand is 0.
        if srcid_input == destid_input:
            st.session_state.prediction = 0
            st.warning("Source and Destination cannot be the same. Demand is predicted as 0.")
        else:
            with st.spinner("Forecasting..."):
                input_data = pd.DataFrame({
                    'doj': [pd.to_datetime(doj_input)], 'srcid': [srcid_input], 'destid': [destid_input],
                    'cumsum_seatcount': [cumsum_seatcount_input], 'cumsum_searchcount': [cumsum_searchcount_input],
                    'route_month_avg': [route_month_avg_input]
                })
                input_data = add_date_features(input_data)
                try:
                    input_data['src_n'] = city_encoder.transform(input_data['srcid'])
                    input_data['dest_n'] = city_encoder.transform(input_data['destid'])
                except ValueError as e:
                    st.error(f"City ID Error: {e}. One or both IDs were not in the training data.")
                    st.stop()
                X_input = input_data[FEATURE_COLUMNS]
                prediction = best_model.predict(X_input)[0]
                st.session_state.prediction = max(0, round(prediction))
    
    if st.session_state.prediction is not None:
        st.metric(
            label=f"Forecast for {doj_input.strftime('%d-%b-%Y')}",
            value=f"{st.session_state.prediction} Seats"
        )
    else:
        st.info("Click 'Predict Demand' to see the forecast.")

with col2:
    st.subheader("Forecast Visualization")
    if st.session_state.prediction is not None:
        fig = create_gauge_chart(st.session_state.prediction, "Predicted Seat Demand")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown(
            "<div style='display: flex; align-items: center; justify-content: center; height: 300px; "
            "background-color: rgba(255, 255, 255, 0.05); border-radius: 10px; border: 1px dashed #334155;'>"
            "<p style='text-align: center; color: #b0c4de;'>📊 The forecast gauge will appear here.</p>"
            "</div>", unsafe_allow_html=True
        )

st.divider()

# --- 6. Explanatory Sections ---
st.subheader("Project Insights")
exp1, exp2 = st.columns(2)

with exp1:
    with st.expander("🤖 About the Model & Architecture"):
        st.markdown("""
        A unified **Streamlit application** serves as both the **Frontend** and **Backend**, enabling a seamless, lightweight workflow.
        - **LightGBM** powers fast and scalable predictions, with Streamlit simplifying deployment—no external servers required.
        
        **Performance Highlights:**
        - **Best Model:** `LightGBM`
        - **Tuned CV-RMSE:** `406.1932`
        
        This decoupled architecture ensures scalability and robustness.
        """)

with exp2:
    with st.expander("📊 Key Predictive Features"):
         st.markdown("""
        Based on the model's feature importance analysis, the most influential factors are:
        
        1.  **`route_month_avg`**: The historical average demand for a specific route.
        2.  **`cumsum_searchcount`**: Cumulative user searches for the route, indicating interest.
        3.  **`cumsum_seatcount`**: Cumulative seats already booked, showing existing demand.
        4.  **Cyclical Date Features**: Capturing weekly and yearly seasonality.
        """)

# --- 7. Footer ---
st.markdown("---")
st.markdown("<p style='text-align: center; color: #888;'>Developed as a solution for the Bus Demand Forecasting Challenge. Built with Streamlit.</p>", unsafe_allow_html=True)
