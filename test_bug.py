import pandas as pd
import numpy as np

df = pd.DataFrame({
    "Time": ["2023-01", "2023-02", "2023-03", "2023-04"],
    "Variable_A": [12.4, 15.1, 14.2, 18.0],
    "Variable_B": [100, 105, 98, 110],
    "Variable_C": [0.45, 0.52, 0.48, 0.61]
})

data_cols = ["Variable_A", "Variable_B", "Variable_C"]

for col in data_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df_subset = df[data_cols].copy()

# Mock prepare_multivariate_timeseries
agg_dict = {"Variable_A": "sum", "Variable_B": "mean", "Variable_C": "mean"}
weekly_df = df.groupby(["Time"]).agg(agg_dict).dropna()

clustered_df = weekly_df[data_cols]

print(clustered_df.values.dtype)
try:
    X = clustered_df.values.astype(np.float64)
    print("Astype success, dtype:", X.dtype)
except ValueError as e:
    print("ValueError:", e)
