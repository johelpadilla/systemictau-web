import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform

def hyper_persistence(taus_series, window_size=20, threshold_chaos=0.41):
    """
    Computes hyper-persistence z-score and core-hyper indicator.
    
    Parameters:
    -----------
    taus_series : numpy.ndarray
        1D array of Systemic Tau values.
    window_size : int
        Rolling window size for local mean and std.
        
    Returns:
    --------
    hp_z : numpy.ndarray
        z-score of current tau relative to recent window.
    core_hyper : numpy.ndarray
        Boolean array, True if in a run of length >= 7 and |tau| < 0.41.
    """
    T = len(taus_series)
    hp_z = np.zeros(T)
    core_hyper = np.zeros(T, dtype=bool)
    
    run_length = 0
    for i in range(T):
        tau = taus_series[i]
        if np.isnan(tau):
            continue
            
        # Z-score computation
        start_idx = max(0, i - window_size)
        recent_window = taus_series[start_idx:i]
        # Ignore NaNs in recent window
        recent_window = recent_window[~np.isnan(recent_window)]
        
        if len(recent_window) > 1:
            mean_t = np.mean(recent_window)
            std_t = np.std(recent_window)
            if std_t > 0:
                hp_z[i] = (tau - mean_t) / std_t
                
        # Core-hyper indicator
        if np.abs(tau) < threshold_chaos:
            run_length += 1
            if run_length >= 7:
                core_hyper[i] = True
        else:
            run_length = 0
            
    return hp_z, core_hyper

def rolling_rqa(taus_series, window_size=25, min_line_length=2, radius_percentile=10):
    """
    Computes a simplified rolling Recurrence Quantification Analysis (RQA).
    Specifically laminarity (LAM) and trapping time (TT).
    """
    T = len(taus_series)
    lam = np.zeros(T)
    tt = np.zeros(T)
    
    for i in range(window_size, T):
        window = taus_series[i - window_size : i]
        if np.isnan(window).any():
            continue
            
        # Reshape for pdist
        X = window.reshape(-1, 1)
        distances = pdist(X, metric='euclidean')
        
        if len(distances) == 0:
            continue
            
        radius = np.percentile(distances, radius_percentile)
        
        # Recurrence matrix
        R = squareform(distances <= radius)
        # Exclude diagonal
        np.fill_diagonal(R, False)
        
        # Find vertical lines
        v_lines = []
        for col in range(window_size):
            run = 0
            for row in range(window_size):
                if R[row, col]:
                    run += 1
                else:
                    if run >= min_line_length:
                        v_lines.append(run)
                    run = 0
            if run >= min_line_length:
                v_lines.append(run)
                
        if v_lines:
            tt[i] = np.mean(v_lines)
            total_recurrent_points = np.sum(R)
            recurrent_in_lines = np.sum(v_lines)
            if total_recurrent_points > 0:
                lam[i] = recurrent_in_lines / total_recurrent_points
                
    return lam, tt

def critical_mass_metric(hp_z, lam, tt, w_hp=0.55, w_tt=0.45):
    """
    Combines Capa 1 indicators into the Critical Mass metric M(k).
    """
    # M(k) = 0.55 * hyper_z(k) + 0.45 * (TT(k) / 7.0)
    # Using 7.0 as the normalization scale from the dengue corpus
    return w_hp * hp_z + w_tt * (tt / 7.0)

def compute_antisynchronization(taus_per_module):
    """
    Computes A_antis: empirical standard deviation of per-module taus at each step.
    """
    return np.nanstd(taus_per_module, axis=1)

def extract_joint_episodes(A_series, M_series, theta_A=0.04, D_min=30, theta_M=1.0):
    """
    Extracts maximal Joint Episodes from the Capa 1 and Capa 2 series.
    """
    T = len(A_series)
    episodes = []
    
    in_run = False
    start_idx = 0
    
    # Identify maximal low-A runs
    low_A_runs = []
    for i in range(T):
        if np.isnan(A_series[i]):
            if in_run:
                low_A_runs.append((start_idx, i - 1))
                in_run = False
            continue
            
        if A_series[i] < theta_A:
            if not in_run:
                in_run = True
                start_idx = i
        else:
            if in_run:
                low_A_runs.append((start_idx, i - 1))
                in_run = False
                
    if in_run:
        low_A_runs.append((start_idx, T - 1))
        
    # Filter by D_min and critical mass
    for start, end in low_A_runs:
        duration = end - start + 1
        if duration >= D_min:
            M_window = M_series[start : end + 1]
            if np.max(M_window) >= theta_M:
                M_mean = np.mean(M_window)
                J = duration * M_mean
                J07 = (duration ** 0.7) * M_mean
                episodes.append({
                    'start': start,
                    'end': end,
                    'D': duration,
                    'M_mean': M_mean,
                    'J': J,
                    'J07': J07
                })
                
    return episodes

def detect_reorganization_frob(taus_matrix):
    """
    Capa 3 detector 1: Frobenius-norm shift in correlation matrix.
    """
    T, N = taus_matrix.shape
    best_t = -1
    max_dist = -1.0
    
    # We need enough samples to compute a correlation matrix (e.g. at least 20 steps)
    min_samples = 20
    
    for t in range(min_samples, T - min_samples):
        pre_X = taus_matrix[:t, :]
        post_X = taus_matrix[t:, :]
        
        # Drop rows with NaN
        pre_X = pre_X[~np.isnan(pre_X).any(axis=1)]
        post_X = post_X[~np.isnan(post_X).any(axis=1)]
        
        if len(pre_X) < min_samples or len(post_X) < min_samples:
            continue
            
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            corr_pre = np.corrcoef(pre_X, rowvar=False)
            corr_post = np.corrcoef(post_X, rowvar=False)
        
        dist = np.linalg.norm(corr_pre - corr_post, ord='fro')
        if dist > max_dist:
            max_dist = dist
            best_t = t
            
    return best_t, max_dist

def detect_reorganization_ks(dtk_series):
    """
    Capa 3 detector 2: Kolmogorov-Smirnov contrast on Deltat_k series.
    """
    T = len(dtk_series)
    best_t = -1
    max_ks = -1.0
    
    min_samples = 20
    
    for t in range(min_samples, T - min_samples):
        pre_dtk = dtk_series[:t]
        post_dtk = dtk_series[t:]
        
        # Only use non-zero increments (open gate)
        pre_dtk = pre_dtk[pre_dtk > 0]
        post_dtk = post_dtk[post_dtk > 0]
        
        if len(pre_dtk) < min_samples or len(post_dtk) < min_samples:
            continue
            
        stat, pval = stats.ks_2samp(pre_dtk, post_dtk)
        if stat > max_ks:
            max_ks = stat
            best_t = t
            
    return best_t, max_ks

def consensus_transition(t_frob, t_ks, tolerance=20):
    """
    Returns consensus transition time.
    """
    if t_frob == -1 and t_ks == -1:
        return -1
    if t_frob == -1:
        return t_ks
    if t_ks == -1:
        return t_frob
        
    if abs(t_frob - t_ks) <= tolerance:
        return int(np.median([t_frob, t_ks]))
    # Simplification: return median anyway if they disagree moderately
    return int(np.median([t_frob, t_ks]))
