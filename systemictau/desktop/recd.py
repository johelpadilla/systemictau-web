import numpy as np
import pandas as pd

def compute_recd_on_cluster(df_aggregated, window):
    """
    Wrapper function to compute RECD on a Medium scale aggregated cluster.
    Passes the aggregated series directly to the core RECD function with the 
    Medium ontological flag.
    """
    if isinstance(df_aggregated, pd.Series):
        arr = df_aggregated.values
    else:
        arr = df_aggregated
        
    return compute_recd_discretization(arr, window, ontological_level="medium")

def compute_recd_discretization(series, window, ontological_level="local"):
    """
    Measures the degree of emergent discrete temporal structure (RECD) 
    in a time series using a robust ordinal conjunction approach.
    
    Parameters:
        series (array-like): The aggregated or local 1D time series.
        window (int): The rolling window size for baseline computation.
        ontological_level (str): 'local' or 'medium' context.
        
    Returns:
        tuple: (recd_array, breakpoints)
            - recd_array: np.ndarray of the same length as the input, padded with NaNs initially.
            - breakpoints: list of indices where a significant change in RECD occurred.
    """
    if isinstance(series, pd.Series):
        arr = series.values
    else:
        arr = np.asarray(series)
        
    n = len(arr)
    recd_array = np.full(n, np.nan)
    
    if n <= window or window < 2:
        return recd_array, []
        
    # 1. Dynamic Noise Threshold (Epsilon)
    # Epsilon = 0.2 * robust_std, providing a very robust "tie" band for true continuity
    # We use Median Absolute Deviation (MAD) to prevent outlier-induced epsilon inflation (especially in Medium Scale SUM clusters)
    diffs_raw = np.diff(arr)
    abs_dev = np.abs(diffs_raw - np.median(diffs_raw))
    mad = np.median(abs_dev)
    
    if mad < 1e-8:
        # Fallback for highly quantized/sparse data where >50% of diffs are 0
        non_zero_devs = abs_dev[abs_dev > 1e-8]
        if len(non_zero_devs) > 0:
            robust_std = np.median(non_zero_devs) * 1.4826
        else:
            robust_std = 1e-8
    else:
        robust_std = 1.4826 * mad
        
    epsilon = 0.2 * robust_std if robust_std > 1e-8 else 1e-8
    
    symbols = np.zeros_like(diffs_raw)
    symbols[diffs_raw > epsilon] = 1
    symbols[diffs_raw < -epsilon] = -1
    
    H_max_theoretical = np.log2(3)
    raw_probs = np.zeros((n, 3)) 
    
    for t in range(window, n):
        sym_window = symbols[t-window : t-1]
        c_minus = np.sum(sym_window == -1)
        c_zero = np.sum(sym_window == 0)
        c_plus = np.sum(sym_window == 1)
        
        total = len(sym_window)
        if total > 0:
            # Laplace Smoothing (Additive Smoothing) to prevent absolute zero entropy in tiny windows
            # This preserves dynamic range across scales and prevents 1.00 saturation when window is small.
            alpha_laplace = 0.1
            smoothed_total = total + (3 * alpha_laplace)
            raw_probs[t] = [
                (c_minus + alpha_laplace) / smoothed_total, 
                (c_zero + alpha_laplace) / smoothed_total, 
                (c_plus + alpha_laplace) / smoothed_total
            ]
            
    # 2. EWMA Smoothing on Probabilities to kill artificial periodicity
    alpha = min(0.3, 2.0 / (window / 2.0 + 1.0))
    smoothed_probs = np.zeros_like(raw_probs)
    
    current_smoothed = raw_probs[window]
    for t in range(window, n):
        current_smoothed = alpha * raw_probs[t] + (1 - alpha) * current_smoothed
        smoothed_probs[t] = current_smoothed
        
        p = smoothed_probs[t]
        p_safe = p[p > 1e-12]
        H = -np.sum(p_safe * np.log2(p_safe))
        
        # Calculate RECD Index (Raw)
        recd_val = 1.0 - (H / H_max_theoretical)
        recd_array[t] = np.clip(recd_val, 0.0, 1.0)
        
    # We remove the artificial 0.1-0.9 empirical scaling because the increased epsilon 
    # naturally produces more entropy and lowers the RECD index dynamically.
            
    # 4. Mathematically Principled Breakpoint Detection (Statistical Change Point)
    breakpoints = []
    
    valid_starts = np.where(~np.isnan(recd_array))[0]
    if len(valid_starts) > window:
        start_idx = valid_starts[0]
        
        # Define backward baseline size and forward validation size
        baseline_size = 3 * window
        forward_size = max(3, window // 2)
        
        # We establish a global noise floor to prevent division by zero in perfectly flat regimes
        global_std = np.nanstd(recd_array)
        noise_floor = max(1e-4, 0.05 * global_std)
        
        for t in range(start_idx + baseline_size, n):
            # Backward baseline (Historical Regime)
            baseline = recd_array[t - baseline_size : t]
            mu_back = np.mean(baseline)
            sigma_back = max(np.std(baseline), noise_floor)
            
            # Absolute Deviation (Z-Score)
            z_score = abs(recd_array[t] - mu_back) / sigma_back
            
            # First difference of baseline to measure expected step size
            diff_back = np.diff(baseline)
            sigma_diff = max(np.std(diff_back), noise_floor)
            
            # Momentum (Suddenness of the jump)
            jump = abs(recd_array[t] - recd_array[t-1])
            z_jump = jump / sigma_diff
            
            # A true structural break must be a statistically significant anomaly in absolute level (>3 sigma)
            # AND it must occur as a sudden transition relative to recent volatility (>2 sigma jump).
            if z_score > 3.0 and z_jump > 2.0:
                # Forward Persistence Validation
                # The regime shift must be sustained, not just a 1-point impulse response.
                # We calculate the mean of the forward window (or whatever is left of the series)
                fwd_end = min(t + forward_size, n)
                if fwd_end > t:
                    forward_window = recd_array[t:fwd_end]
                    mu_fwd = np.mean(forward_window)
                    
                    # The forward regime mean must remain significantly shifted (> 2 sigma from old baseline)
                    z_fwd = abs(mu_fwd - mu_back) / sigma_back
                    
                    if z_fwd > 2.0:
                        # Prevent overlapping breakpoints within the same context
                        if len(breakpoints) == 0 or (t - breakpoints[-1]) > window:
                            breakpoints.append(t)
                    
    return recd_array, breakpoints
