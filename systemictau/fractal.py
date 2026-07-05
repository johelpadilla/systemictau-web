import numpy as np

def estimate_higuchi_dimension(T_series, k_max=10):
    """
    Estimates the Higuchi Fractal Dimension of a time series.
    The RECD extramental time T typically yields D ~ 1.98.
    
    Parameters:
    -----------
    T_series : numpy.ndarray
        1D array of the cumulative time series (e.g., T_n).
    k_max : int
        Maximum window size (delay) to use for computing curve length.
        
    Returns:
    --------
    D : float
        The estimated fractal dimension.
    """
    N = len(T_series)
    if N < 2 * k_max:
        raise ValueError("Series is too short for the chosen k_max")
        
    # Array to hold the curve length L(k)
    L = np.zeros(k_max)
    
    for k in range(1, k_max + 1):
        Lk = 0.0
        for m in range(k):
            # Indices for the sub-series
            indices = np.arange(m, N, k)
            
            # Number of points in this sub-series
            n_samples = len(indices)
            
            if n_samples > 1:
                # Sum of absolute differences
                diffs = np.abs(np.diff(T_series[indices]))
                sum_diffs = np.sum(diffs)
                
                # Normalization factor
                norm_factor = (N - 1) / (k * (n_samples - 1))
                
                # Add to total length L(k)
                Lk += (sum_diffs * norm_factor)
                
        # Average over the k sub-series and normalize by k
        L[k-1] = (Lk / k) / k
        
    # Calculate the slope of log(L(k)) vs log(1/k)
    log_L = np.log(L)
    log_k = np.log(1.0 / np.arange(1, k_max + 1))
    
    # Linear fit
    coeffs = np.polyfit(log_k, log_L, 1)
    D = coeffs[0]
    
    return D
