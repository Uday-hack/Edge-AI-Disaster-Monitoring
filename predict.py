import pandas as pd
import joblib


# ==========================================
# 1. LOAD TRAINED MODEL
# ==========================================

model = joblib.load(
    "models/disaster_model.pkl"
)

print("Model loaded successfully!")


# ==========================================
# 2. CREATE A TEST DISASTER SCENARIO
# ==========================================

test_data = pd.DataFrame([{

    "Disaster_Type": "Flood",

    "Rainfall_mm": 350,

    "Temperature_C": 30,

    "Humidity": 90,

    "Wind_Speed_kmh": 40,

    "River_Level_m": 8,

    "Earthquake_Magnitude": 0,

    "Population_Density": 7000,

    "Infrastructure_Risk": 70,

    "Response_Time_min": 60,

    "Resource_Availability": 50,

    "Shelter_Capacity": 50,

    "Evacuation_Rate": 50

}])


# ==========================================
# 3. MAKE PREDICTION
# ==========================================

prediction = model.predict(
    test_data
)


# ==========================================
# 4. DISPLAY RESULT
# ==========================================

print("\n================================")
print("       DISASTER PREDICTION")
print("================================")

print(
    "Disaster Type:",
    test_data["Disaster_Type"].iloc[0]
)

print(
    "Predicted Severity:",
    prediction[0]
)

print("================================")