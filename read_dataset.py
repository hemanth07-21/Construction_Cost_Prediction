import pandas as pd

# Load dataset
df = pd.read_csv("dataset.csv")

# Basic information
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())
print("\nDataset shape:")
print(df.shape)

print("\nMaterial values:")
print(df["Material"].unique())

print("\nLocation values:")
print(df["Location"].unique())
print("\nMaterial values:")
print(df["Material"].unique())

print("\nDataset shape:")
print(df.shape)