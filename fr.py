import joblib
df = joblib.load("reference_data.joblib")
df.to_csv("reference_data.csv", index=False)