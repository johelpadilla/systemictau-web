import pandas as pd
import numpy as np
from numba import njit

@njit
def test_numba(X, window_size, stride):
    T, N = X.shape
    return T

df = pd.DataFrame({
    "Time": ["2023-01", "2023-02", "2023-03", "2023-04"],
    "Variable_A": [12.4, 15.1, 14.2, 18.0],
    "Variable_B": [100, 105, 98, 110],
    "Variable_C": [0.45, 0.52, 0.48, 0.61]
})

data_cols = ["Variable_A", "Variable_B", "Variable_C"]

# Mock prepare_multivariate_timeseries
agg_dict = {"Variable_A": "sum", "Variable_B": "mean", "Variable_C": "mean"}
weekly_df = df.groupby(["Time"]).agg(agg_dict).dropna()

clustered_df = weekly_df[data_cols]
X = clustered_df.values

try:
    test_numba(X, 13, 1)
    print("Success without cast")
except Exception as e:
    print("Failed without cast:", e)

try:
    X2 = X.astype(np.float64)
    test_numba(X2, 13, 1)
    print("Success with cast")
except Exception as e:
    print("Failed with cast:", e)
