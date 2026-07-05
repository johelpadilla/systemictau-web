import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, "/Users/johelpadilla/grok-safe/Gemini/systemictau_v4.1.5")
sys.path.insert(0, "/Users/johelpadilla/grok-safe/Gemini/systemictau/src")

from systemictau.panel import prepare_multivariate_timeseries
from systemictau.analysis import run_full_analysis

df = pd.DataFrame({
    "Time": ["2023-01", "2023-02", "2023-03", "2023-04"],
    "Variable_A": [12.4, 15.1, 14.2, 18.0],
    "Variable_B": [100, 105, 98, 110],
    "Variable_C": [0.45, 0.52, 0.48, 0.61]
})

data_cols = ["Variable_A", "Variable_B", "Variable_C"]
for col in data_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

agg_dict = {"Variable_A": "sum", "Variable_B": "mean", "Variable_C": "mean"}
X_prepared, weekly_df = prepare_multivariate_timeseries(df, ["Time"], data_cols, agg_dict, True, True)

clustered_df = weekly_df[data_cols]
X = clustered_df.values.astype(np.float64)

try:
    res = run_full_analysis(X, window_size=13, component_names=data_cols)
    print("Success")
except Exception as e:
    import traceback
    traceback.print_exc()
