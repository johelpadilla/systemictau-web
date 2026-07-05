import numpy as np
import pandas as pd

def compute_ews(time_series: np.ndarray, window_size: int = 13) -> dict:
    """
    Computes Early Warning Signals (EWS) for critical transitions.
    Specifically: Rolling Variance and Rolling Lag-1 Autocorrelation (AC1).
    """
    ts = pd.Series(time_series)
    
    # Rolling Variance
    # As the system approaches a tipping point, resilience decreases, causing variance to spike.
    variance = ts.rolling(window=window_size).var().values
    
    # Rolling AC1 (Autocorrelation)
    # Critical slowing down implies the system's current state is highly dependent on the previous state.
    def ac1(x):
        if len(x) > 1 and np.var(x) > 0:
            return pd.Series(x).autocorr(lag=1)
        return np.nan
        
    ac1_series = ts.rolling(window=window_size).apply(ac1).values
    
    # Rolling Skewness
    # Indicates asymmetry in the fluctuations as the attractor landscape tilts.
    skewness = ts.rolling(window=window_size).skew().values
    
    return {
        "variance": variance,
        "ac1": ac1_series,
        "skewness": skewness
    }
