import streamlit as st
import pandas as pd
import joblib


# Load trained model
model = joblib.load("models/disaster_model.pkl")


# -----------------------------
# Management Index Function
# -----------------------------
def calculate_management_index(
    resource_availability,
    shelter_capacity,
    evacuation_rate,
    response_time
):
    response_performance = max(
        0,
        100 - (response_time / 120 * 100)
    )

    management_index = (
        resource_availability * 0.30
        + shelter_capacity * 0.25
        + evacuation_rate * 0.25
        + response_performance * 0.20
    )

    return management_index, response_performance


# -----------------------------
# Risk Level Function
# -----------------------------
def calculate_risk_level(prediction, management_index):

    severity_score = {
        "Low": 20,
        "Moderate": 50,
        "High": 80,
        "Critical": 100
    }

    disaster_score = severity_score.get(prediction, 50)

    combined_risk = (
        disaster_score * 0.60
        + management_index * 0.40
    )

    if combined_risk >= 75:
        risk_level = "CRITICAL"
    elif combined_risk >= 55:
        risk_level = "HIGH"
    elif combined_risk >= 35:
        risk_level = "MODERATE"
    else:
        risk_level = "LOW"

    return risk_level, combined_risk


# -----------------------------
# Response Recommendation
# -----------------------------
def get_response_recommendation(risk_level):

    if risk_level == "CRITICAL":
        return (
            "Immediate emergency response recommended. "
            "Activate emergency response teams, prioritize evacuation "
            "of vulnerable areas, and allocate available resources immediately."
        )

    elif risk_level == "HIGH":
        return (
            "High-priority response recommended. "
            "Prepare emergency teams, increase resource availability, "
            "and consider evacuation of vulnerable areas."
        )

    elif risk_level == "MODERATE":
        return (
            "Continue active monitoring. "
            "Prepare emergency resources and review evacuation "
            "and shelter arrangements."
        )

    else:
        return (
            "Continue routine monitoring. "
            "Maintain preparedness and keep emergency resources available."
        )


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Edge AI Disaster Response Monitoring System",
    page_icon="🌍",
    layout="wide"
)


# -----------------------------
# Page Title
# -----------------------------
st.title("🌍 Edge AI Disaster Response Monitoring System")

st.write(
    "A Python-based Machine Learning prototype for multi-disaster "
    "severity prediction, risk assessment, and response support."
)

st.divider()


# -----------------------------
# Input Section
# -----------------------------
st.subheader("Enter Disaster Information")

col1, col2 = st.columns(2)


with col1:

    disaster_type = st.selectbox(
        "Disaster Type",
        [
            "Flood",
            "Earthquake",
            "Cyclone",
            "Landslide",
            "Wildfire"
        ]
    )

    rainfall = st.number_input(
        "Rainfall (mm)",
        min_value=0.0,
        max_value=600.0,
        value=350.0
    )

    temperature = st.number_input(
        "Temperature (°C)",
        min_value=-10.0,
        max_value=50.0,
        value=30.0
    )

    humidity = st.number_input(
        "Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=90.0
    )

    wind_speed = st.number_input(
        "Wind Speed (km/h)",
        min_value=0.0,
        max_value=250.0,
        value=40.0
    )

    river_level = st.number_input(
        "River Level (m)",
        min_value=0.0,
        max_value=15.0,
        value=8.0
    )

    earthquake_magnitude = st.number_input(
        "Earthquake Magnitude",
        min_value=0.0,
        max_value=10.0,
        value=0.0
    )


with col2:

    population_density = st.number_input(
        "Population Density",
        min_value=0.0,
        max_value=10000.0,
        value=7000.0
    )

    infrastructure_risk = st.number_input(
        "Infrastructure Risk (%)",
        min_value=0.0,
        max_value=100.0,
        value=70.0
    )

    response_time = st.number_input(
        "Response Time (minutes)",
        min_value=1.0,
        max_value=120.0,
        value=60.0
    )

    resource_availability = st.slider(
        "Resource Availability (%)",
        0,
        100,
        50
    )

    shelter_capacity = st.slider(
        "Shelter Capacity (%)",
        0,
        100,
        50
    )

    evacuation_rate = st.slider(
        "Evacuation Rate (%)",
        0,
        100,
        50
    )


st.divider()


# -----------------------------
# Analyze Button
# -----------------------------
if st.button(
    "🚨 ANALYZE DISASTER",
    use_container_width=True
):

    # -----------------------------
    # Prepare Input Data
    # -----------------------------
    input_data = pd.DataFrame([{

        "Disaster_Type": disaster_type,

        "Rainfall_mm": rainfall,

        "Temperature_C": temperature,

        "Humidity": humidity,

        "Wind_Speed_kmh": wind_speed,

        "River_Level_m": river_level,

        "Earthquake_Magnitude": earthquake_magnitude,

        "Population_Density": population_density,

        "Infrastructure_Risk": infrastructure_risk,

        "Response_Time_min": response_time,

        "Resource_Availability": resource_availability,

        "Shelter_Capacity": shelter_capacity,

        "Evacuation_Rate": evacuation_rate

    }])


    # -----------------------------
    # ML Prediction
    # -----------------------------
    prediction = model.predict(input_data)[0]


    # -----------------------------
    # Management Index
    # -----------------------------
    management_index, response_performance = (
        calculate_management_index(
            resource_availability,
            shelter_capacity,
            evacuation_rate,
            response_time
        )
    )


    # -----------------------------
    # Risk Level
    # -----------------------------
    risk_level, combined_risk = calculate_risk_level(
        prediction,
        management_index
    )


    # -----------------------------
    # Response Recommendation
    # -----------------------------
    recommendation = get_response_recommendation(
        risk_level
    )


    # -----------------------------
    # Results
    # -----------------------------
    st.subheader("📊 Analysis Results")

    result_col1, result_col2, result_col3 = st.columns(3)


    with result_col1:

        st.metric(
            "Predicted Disaster Severity",
            prediction
        )


    with result_col2:

        st.metric(
            "Management Index",
            f"{management_index:.2f}%"
        )


    with result_col3:

        st.metric(
            "Risk Level",
            risk_level
        )


    st.write(
        f"Combined Risk Score: **{combined_risk:.2f}/100**"
    )


    # -----------------------------
    # Automatic Response
    # -----------------------------
    st.subheader("🤖 Automatic Response Recommendation")


    if risk_level == "CRITICAL":

        st.error(
            f"🚨 {recommendation}"
        )

    elif risk_level == "HIGH":

        st.warning(
            f"⚠️ {recommendation}"
        )

    elif risk_level == "MODERATE":

        st.info(
            f"ℹ️ {recommendation}"
        )

    else:

        st.success(
            f"✅ {recommendation}"
        )


    st.divider()


    # -----------------------------
    # Management Indicators
    # -----------------------------
    st.subheader("📈 Management Indicators")


    indicators = pd.DataFrame({

        "Indicator": [

            "Resource Availability",

            "Shelter Capacity",

            "Evacuation Rate",

            "Response Performance"

        ],

        "Value": [

            resource_availability,

            shelter_capacity,

            evacuation_rate,

            response_performance

        ]

    })


    st.bar_chart(
        indicators.set_index("Indicator")
    )


    st.divider()


    # -----------------------------
    # System Information
    # -----------------------------
    st.subheader("ℹ️ System Information")

    st.info(
        "This prototype performs local ML inference using a trained "
        "Random Forest model. The risk level combines predicted disaster "
        "severity with the management index. The response recommendation "
        "is generated using transparent rule-based logic. This is a "
        "prototype decision-support system based on a synthetic dataset "
        "and is not an official emergency management system."
    )

