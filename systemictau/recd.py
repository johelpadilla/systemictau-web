import numpy as np

# Universal Feigenbaum constants
DELTA = 4.6692016091
THETA_CHAOS = 0.41
THETA_STABLE = 0.50

def gate_function(tau_s, threshold_chaos=THETA_CHAOS, threshold_stable=THETA_STABLE):
    """
    Computes the gate function g(tau_s) according to the RECD law.
    
    Parameters:
    -----------
    tau_s : float or numpy.ndarray
        The Systemic Tau value(s).
    threshold_chaos : float
        The boundary of the chaotic band (default 0.41).
    threshold_stable : float
        The boundary of the stable regime (default +0.50).
        
    Returns:
    --------
    g_val : float or numpy.ndarray
        The evaluated gate function.
    """
    # Evaluate as an array if it's an array, else scalar
    tau_s = np.asarray(tau_s)
    g_val = np.zeros_like(tau_s, dtype=float)
    
    # Regime 1: Stable
    mask_stable = tau_s >= threshold_stable
    g_val[mask_stable] = 1.0
    
    # Regime 2: Chaotic band
    mask_chaos = np.abs(tau_s) < threshold_chaos
    # g(tau) = ((delta - 1) / delta) * (0.41 - |tau_s|) / 0.41
    prefactor = (DELTA - 1.0) / DELTA
    g_val[mask_chaos] = prefactor * (threshold_chaos - np.abs(tau_s[mask_chaos])) / threshold_chaos
    
    # Regime 3: Antisynchronization
    mask_anti = tau_s <= -threshold_chaos
    g_val[mask_anti] = -1.0
    
    # Note: between 0.41 and 0.50, the gate is effectively 0 according to the document's logic, 
    # or tapers off. The text says "Outside this band the gate is effectively closed or inverted".
    # We leave it at 0.0 for values between 0.41 and 0.50.
    
    if g_val.ndim == 0:
        return float(g_val)
    return g_val

def compute_recd_increments(taus_series, window_size=13, dt0=1.0):
    """
    Computes the RECD increments Delta t_k and the associated renormalization depths k.
    
    Parameters:
    -----------
    taus_series : numpy.ndarray
        1D array of Systemic Tau values.
    window_size : int
        The sliding window length n used for normalization.
    dt0 : float
        Base temporal step.
        
    Returns:
    --------
    dtk_series : numpy.ndarray
        The array of increments Delta t_k (0 if gate is closed).
    gate_series : numpy.ndarray
        The gate function values.
    depth_series : numpy.ndarray
        The calculated renormalization depths k.
    """
    T = len(taus_series)
    dtk_series = np.zeros(T)
    gate_series = np.zeros(T)
    depth_series = np.zeros(T, dtype=int)
    
    chaotic_count = 0
    in_run = False
    
    for i in range(T):
        tau = taus_series[i]
        
        if np.isnan(tau):
            continue
            
        g = gate_function(tau)
        gate_series[i] = g
        
        if np.abs(tau) < THETA_CHAOS:
            if not in_run:
                in_run = True
            chaotic_count += 1
            k = chaotic_count
            
            # Delta t_k = delta^{-k} * |tau_s| * dt0 / n
            dtk = (DELTA ** (-k)) * np.abs(tau) * (dt0 / window_size)
            dtk_series[i] = dtk
            depth_series[i] = k
        else:
            in_run = False
            chaotic_count = 0
            dtk_series[i] = 0.0
            depth_series[i] = 0
            
    return dtk_series, gate_series, depth_series

def accumulate_time(taus_series, window_size=13, dt0=1.0):
    """
    Accumulates discrete extramental time T according to the RECD law.
    
    T_n = sum_k g(tau_s(k)) * Delta t_k
    """
    dtk_series, gate_series, depth_series = compute_recd_increments(
        taus_series, window_size=window_size, dt0=dt0
    )
    
    # Cumulative sum
    increments = gate_series * dtk_series
    T_series = np.cumsum(np.nan_to_num(increments))
    
    return T_series, dtk_series, gate_series, depth_series
