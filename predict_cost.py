import joblib
import pandas as pd
import os

# Load the trained model
model_file = "construction_cost_model.pkl"

if not os.path.exists(model_file):
    print("ERROR: construction_cost_model.pkl was not found.")
    print("Make sure the model file is in the same folder.")
    exit()

model = joblib.load(model_file)

print("====================================")
print("   CONSTRUCTION COST PREDICTOR")
print("====================================")

# Get user inputs
area = float(input("Enter area in square feet: "))
floors = int(input("Enter number of floors: "))
material = input("Enter material (RCC/Steel): ")
location = input("Enter location (Hyderabad/Warangal): ")
labour_cost = float(input("Enter labour cost: "))
duration = int(input("Enter duration in days: "))

# Create input data
new_data = pd.DataFrame({
    "Area": [area],
    "Floors": [floors],
    "Material": [material],
    "Location": [location],
    "LabourCost": [labour_cost],
    "Duration": [duration]
})

# Predict construction cost
predicted_cost = model.predict(new_data)[0]

print("\n====================================")
print("PREDICTED CONSTRUCTION COST")
print("====================================")
print(f"₹{predicted_cost:,.2f}")
print("====================================")