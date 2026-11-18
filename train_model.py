import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# 1. LOAD DATASET
# ==========================================

print("Loading dataset...")

df = pd.read_csv("data/disaster_data.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ==========================================
# 2. SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop("Severity", axis=1)

y = df["Severity"]


# ==========================================
# 3. DEFINE FEATURES
# ==========================================

categorical_features = [
    "Disaster_Type"
]


numeric_features = [
    "Rainfall_mm",
    "Temperature_C",
    "Humidity",
    "Wind_Speed_kmh",
    "River_Level_m",
    "Earthquake_Magnitude",
    "Population_Density",
    "Infrastructure_Risk",
    "Response_Time_min",
    "Resource_Availability",
    "Shelter_Capacity",
    "Evacuation_Rate"
]


# ==========================================
# 4. PREPROCESSING
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        ),

        (
            "numeric",
            "passthrough",
            numeric_features
        )
    ]
)


# ==========================================
# 5. CREATE RANDOM FOREST MODEL
# ==========================================

random_forest = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)


# ==========================================
# 6. CREATE COMPLETE PIPELINE
# ==========================================

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            random_forest
        )
    ]
)


# ==========================================
# 7. SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ==========================================
# 8. TRAIN MODEL
# ==========================================

print("\nTraining Random Forest model...")

pipeline.fit(
    X_train,
    y_train
)

print("Training completed!")


# ==========================================
# 9. MAKE PREDICTIONS
# ==========================================

print("\nMaking predictions...")

y_pred = pipeline.predict(X_test)


# ==========================================
# 10. CALCULATE ACCURACY
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n======================================")
print("       MODEL PERFORMANCE")
print("======================================")

print(
    f"\nAccuracy: {accuracy * 100:.2f}%"
)


# ==========================================
# 11. CLASSIFICATION REPORT
# ==========================================

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)


# ==========================================
# 12. CONFUSION MATRIX
# ==========================================

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ==========================================
# 13. SAVE MODEL
# ==========================================

model_path = "models/disaster_model.pkl"

joblib.dump(
    pipeline,
    model_path
)

print("\n======================================")
print("MODEL SAVED SUCCESSFULLY!")
print("======================================")

print(
    f"\nModel location: {model_path}"
)