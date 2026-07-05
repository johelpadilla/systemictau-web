import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from .core import compute_taus
from .recd import compute_recd_increments, accumulate_time

def run_systemic_analysis(df: pd.DataFrame, variables: List[str], window_size: int = 13) -> Dict[str, Any]:
    """
    Pure mathematical orchestrator for the Systemic Tau Paradigm.
    Takes a DataFrame and calculates Tau_s, t*, and RECD.
    """
    if df.empty or not variables:
        return {}

    # 1. Extract raw data matrix X
    X = df[variables].values
    
    # 2. Compute Systemic Tau
    taus_global, taus_per_module = compute_taus(X, window_size=window_size)
    
    # 3. Handle NaNs and calculate baseline stats
    tau_series = taus_global
    
    # 4. Find Critical Reorganization Point (t*)
    # We find the point of maximum absolute derivative of tau_s, or just the maximum point of synchronization
    t_star = None
    try:
        valid_idx = np.where(~np.isnan(tau_series))[0]
        if len(valid_idx) > 0:
            tau_valid = tau_series[valid_idx]
            # Paradigm typically defines t* near max synchronization
            t_star_idx = np.nanargmax(tau_valid)
            t_star = int(valid_idx[t_star_idx])
    except Exception:
        pass

    # 5. Compute RECD increments and accumulation
    recd_dtk = []
    recd_accumulated = []
    
    try:
        dtk_s, g_s, depth_s = compute_recd_increments(tau_series, window_size=window_size)
        acc_t, dtk2, g2, depth2 = accumulate_time(tau_series, window_size=window_size)
        recd_dtk = dtk_s.tolist()
        recd_accumulated = acc_t.tolist()
    except Exception as e:
        print(f"RECD Calculation Error: {e}")

    # Return pure mathematical JSON-serializable dictionary
    return {
        "tau_series": [float(x) if not np.isnan(x) else None for x in tau_series],
        "t_star": t_star,
        "recd_dtk": [float(x) if not np.isnan(x) else None for x in recd_dtk] if len(recd_dtk) > 0 else [],
        "recd_accumulated": [float(x) if not np.isnan(x) else None for x in recd_accumulated] if len(recd_accumulated) > 0 else [],
    }
