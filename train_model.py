import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

import joblib


# -----------------------------------
# 1. Load the dataset
# -----------------------------------

file_name = "dataset.csv"

if not os.path.exists(file_name):
    print("ERROR: dataset.csv was not found.")
    print("Make sure dataset.csv is in the same folder as train_model.py")
    exit()

data = pd.read_csv(file_name)

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)


# -----------------------------------
# 2. Check required columns
# -----------------------------------

required_columns = [
    "Area",
    "Floors",
    "Material",
    "Location",
    "LabourCost",
    "Duration",
    "Cost"
]

missing_columns = [
    column for column in required_columns
    if column not in data.columns
]

if missing_columns:
    print("ERROR: These columns are missing:")
    print(missing_columns)
    exit()


# -----------------------------------
# 3. Check missing values
# -----------------------------------

if data[required_columns].isnull().sum().sum() > 0:
    print("ERROR: Missing values found in the dataset.")
    print(data[required_columns].isnull().sum())
    exit()

print("No missing values found.")


# -----------------------------------
# 4. Separate input and output
# -----------------------------------

X = data[
    [
        "Area",
        "Floors",
        "Material",
        "Location",
        "LabourCost",
        "Duration"
    ]
]

y = data["Cost"]


# -----------------------------------
# 5. Define categorical columns
# -----------------------------------

categorical_columns = [
    "Material",
    "Location"
]


# -----------------------------------
# 6. Convert text into numbers
# -----------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ],
    remainder="passthrough"
)


# -----------------------------------
# 7. Create Random Forest model
# -----------------------------------

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)


# -----------------------------------
# 8. Create complete ML pipeline
# -----------------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# -----------------------------------
# 9. Split dataset
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# -----------------------------------
# 10. Train the model
# -----------------------------------

print("\nTraining model...")

pipeline.fit(X_train, y_train)

print("Model trained successfully!")


# -----------------------------------
# 11. Make predictions
# -----------------------------------

predictions = pipeline.predict(X_test)


# -----------------------------------
# 12. Evaluate model
# -----------------------------------

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n----- MODEL RESULTS -----")
print("Mean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 4))


# -----------------------------------
# 13. Save model
# -----------------------------------

model_file = "construction_cost_model.pkl"

joblib.dump(pipeline, model_file)

print("\nModel saved successfully!")
print("Saved as:", model_file)
import os
print("Model locations")
print(os.path.abspath("Construction_cost_model.pkl"))