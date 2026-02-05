import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Sea Cucumber Weight Predictor",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #667eea;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 2rem 0;
    }
    .prediction-value {
        font-size: 3rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #f0f9ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.2rem;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Load or train model
@st.cache_resource
def load_model():
    """Load trained model and scaler, or train new ones"""

    # Feature names
    feature_names = [
        'Area (Ha 1)', 
        'INITIAL WEIGHT/kg', 
        'DENSITY 1 (total stock/Ha)',
        'HARVEST NO. 1', 
        'PRODUCTION 1', 
        'PRODUCTION 1/Ha',
        'DISTANCE/m', 
        'VEGETATION%', 
        'AVERAGE_DEPTH/ft',
        'PENS', 
        'DURATIONMONTHS'
    ]

    target = 'AVG WEIGHT 1'

    try:
        # Try to load existing model
        with open('weight_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('weight_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler, feature_names, "Loaded existing model"

    except FileNotFoundError:
        # Train new model
        try:
            df = pd.read_csv('data/survey data.csv')

            # Clean data
            df['DENSITY 1 (total stock/Ha)'] = df['DENSITY 1 (total stock/Ha)'].replace('#REF!', np.nan)
            df['DENSITY 1 (total stock/Ha)'] = pd.to_numeric(df['DENSITY 1 (total stock/Ha)'])

            # Prepare data
            df_clean = df[feature_names + [target]].dropna()
            X = df_clean[feature_names]
            y = df_clean[target]

            # Scale and train
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            model = LinearRegression()
            model.fit(X_scaled, y)

            # Save model
            with open('weight_model.pkl', 'wb') as f:
                pickle.dump(model, f)
            with open('weight_scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)

            r2_score = model.score(X_scaled, y)
            return model, scaler, feature_names, f"Trained new model (R² = {r2_score:.4f})"

        except Exception as e:
            st.error(f"Error loading/training model: {e}")
            return None, None, feature_names, "Error"

# Load model
model, scaler, feature_names, model_status = load_model()

# Header
st.markdown('<div class="main-header">🐟 Sea Cucumber Weight Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predict final sea cucumber weight using ML-powered analysis</div>', unsafe_allow_html=True)

# Sidebar - Model Info
with st.sidebar:
    st.header("ℹ️ Model Information")
    st.info(model_status)

    st.markdown("---")

    st.header("📊 About This App")
    st.markdown("""
    This application uses a **Linear Regression** model to predict sea cucumber final weight based on:

    - Farm characteristics
    - Stocking parameters
    - Production metrics
    - Environmental factors

    **Accuracy:** ~52% R² score
    """)

    st.markdown("---")

    st.header("📖 How to Use")
    st.markdown("""
    1. Fill in all feature values
    2. Click **Predict Weight**
    3. View prediction results
    4. Download prediction report
    """)

# Check if model loaded
if model is None or scaler is None:
    st.error("❌ Model not available. Please ensure survey-data.csv is in the same directory.")
    st.stop()

# Main content - Two columns layout
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📝 Enter Farm Parameters")

    # Create tabs for organized input
    tab1, tab2, tab3, tab4 = st.tabs(["🏞️ Farm Info", "🐟 Sea Cucumber Info", "📈 Production", "🌍 Environment"])

    with tab1:
        st.subheader("Farm Characteristics")
        col_a, col_b = st.columns(2)

        with col_a:
            area = st.number_input(
                "Farm Area (Hectares)", 
                min_value=0.1, max_value=10.0, value=0.8, step=0.1,
                help="Total farm area in hectares"
            )

            pens = st.number_input(
                "Number of Pens", 
                min_value=1, max_value=10, value=2, step=1,
                help="Number of sea cucumber pens/cages"
            )

        with col_b:
            distance = st.number_input(
                "Distance from Water Source (meters)", 
                min_value=0, max_value=5000, value=300, step=50,
                help="Distance to nearest water source"
            )

            duration = st.number_input(
                "Cycle Duration (Months)", 
                min_value=1.0, max_value=24.0, value=8.0, step=0.5,
                help="Expected farming cycle duration"
            )

    with tab2:
        st.subheader("Sea cucumber Parameters")
        col_a, col_b = st.columns(2)

        with col_a:
            initial_weight = st.number_input(
                "Initial Weight (kg)", 
                min_value=0.01, max_value=1.0, value=0.10, step=0.01, format="%.3f",
                help="Average initial sea cucumber weight"
            )

            density = st.number_input(
                "Stocking Density (sea cucumber/Ha)", 
                min_value=1000, max_value=100000, value=10000, step=1000,
                help="Number of sea cucumber per hectare"
            )

        with col_b:
            harvest_no = st.number_input(
                "Number of sea cucumber to Harvest", 
                min_value=100, max_value=100000, value=8000, step=500,
                help="Expected number of sea cucumber at harvest"
            )

    with tab3:
        st.subheader("Production Metrics")
        col_a, col_b = st.columns(2)

        with col_a:
            production = st.number_input(
                "Total Production (kg)", 
                min_value=100, max_value=50000, value=3600, step=100,
                help="Expected total production in kg"
            )

        with col_b:
            production_ha = st.number_input(
                "Production per Hectare (kg/Ha)", 
                min_value=500, max_value=30000, value=4500, step=100,
                help="Production per hectare"
            )

    with tab4:
        st.subheader("Environmental Factors")
        col_a, col_b = st.columns(2)

        with col_a:
            vegetation = st.slider(
                "Vegetation Coverage (%)", 
                min_value=0, max_value=100, value=60, step=5,
                help="Percentage of vegetation cover"
            )

        with col_b:
            depth = st.slider(
                "Average Water Depth (feet)", 
                min_value=1, max_value=20, value=3, step=1,
                help="Average water depth in feet"
            )

with col2:
    st.header("🎯 Quick Stats")

    # Calculate derived metrics
    total_fish = int(density * area)
    expected_avg_weight = production / harvest_no if harvest_no > 0 else 0

    st.metric("Total sea cucumber Stocked", f"{total_fish:,}")
    st.metric("Expected Harvest", f"{harvest_no:,}")
    st.metric("Survival Rate", f"{(harvest_no/total_fish*100):.1f}%" if total_fish > 0 else "N/A")
    st.metric("Production Target", f"{production:,.0f} kg")

# Prediction section
st.markdown("---")
st.header("🔮 Make Prediction")

col1, col2, col3 = st.columns([1, 1, 1])

with col2:
    predict_button = st.button("🎯 Predict sea cucumber Weight", use_container_width=True)

if predict_button:
    # Prepare input data
    input_data = {
        'Area (Ha 1)': area,
        'INITIAL WEIGHT/kg': initial_weight,
        'DENSITY 1 (total stock/Ha)': density,
        'HARVEST NO. 1': harvest_no,
        'PRODUCTION 1': production,
        'PRODUCTION 1/Ha': production_ha,
        'DISTANCE/m': distance,
        'VEGETATION%': vegetation,
        'AVERAGE_DEPTH/ft': depth,
        'PENS': pens,
        'DURATIONMONTHS': duration
    }

    # Create DataFrame
    input_df = pd.DataFrame([input_data])

    # Scale and predict
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    # Display prediction
    st.markdown(f"""
    <div class="prediction-box">
        <h2>Predicted cucumber Weight</h2>
        <div class="prediction-value">{prediction:.3f} kg</div>
        <p style="font-size: 1.2rem;">{prediction*1000:.0f} grams</p>
    </div>
    """, unsafe_allow_html=True)

    # Additional insights
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Weight Gain", 
            f"{(prediction - initial_weight)*1000:.0f} g",
            delta=f"{((prediction/initial_weight - 1)*100):.1f}%"
        )

    with col2:
        st.metric(
            "Total Harvest Weight", 
            f"{(prediction * harvest_no):.0f} kg",
            delta=f"{((prediction * harvest_no)/production*100 - 100):.1f}% vs target"
        )

    with col3:
        st.metric(
            "Feed Conversion Estimate",
            f"{(production/(harvest_no * initial_weight)):.2f}",
            help="Production / Initial biomass"
        )

    with col4:
        st.metric(
            "Growth Rate",
            f"{((prediction - initial_weight)/duration*1000):.1f} g/month"
        )

    # Detailed breakdown
    st.markdown("---")
    st.subheader("📊 Prediction Details")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("**Input Summary**")
        st.write(f"• Initial Weight: {initial_weight:.3f} kg")
        st.write(f"• Final Weight: {prediction:.3f} kg")
        st.write(f"• Cycle Duration: {duration:.1f} months")
        st.write(f"• Sea Cucumber Count: {harvest_no:,}")
        st.write(f"• Farm Area: {area:.1f} ha")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("**Production Summary**")
        st.write(f"• Total Production: {production:,.0f} kg")
        st.write(f"• Production/Ha: {production_ha:,.0f} kg/ha")
        st.write(f"• Predicted Total: {(prediction * harvest_no):,.0f} kg")
        st.write(f"• Stocking Density: {density:,} SC/ha")
        st.markdown('</div>', unsafe_allow_html=True)

    # Download report
    st.markdown("---")

    # Create detailed report
    report_data = {
        'Parameter': list(input_data.keys()) + ['Predicted Weight (kg)', 'Predicted Weight (g)', 'Weight Gain (g)', 'Growth Rate (g/month)'],
        'Value': list(input_data.values()) + [
            f"{prediction:.3f}",
            f"{prediction*1000:.0f}",
            f"{(prediction - initial_weight)*1000:.0f}",
            f"{((prediction - initial_weight)/duration*1000):.1f}"
        ]
    }
    report_df = pd.DataFrame(report_data)

    csv = report_df.to_csv(index=False)

    st.download_button(
        label="📥 Download Prediction Report (CSV)",
        data=csv,
        file_name=f"weight_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p><strong>🐟 Sea Cucumber Weight Predictor</strong> | Powered by Machine Learning</p>
    <p>Model: Linear Regression | Accuracy: ~52% R² | Features: 11 parameters</p>
</div>
""", unsafe_allow_html=True)
