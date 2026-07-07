import numpy as np
import warnings

try:
    from ripser import ripser
    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False

from .core import _kendall_tau_matrix_numba, _kendall_tau_numba, HAS_NUMBA

def compute_tda_features(
    X: np.ndarray,
    window_size: int,
    stride: int = 5,
    mode: str = "fast",
    distance_metric: str = "abs_kendall",
    persistence_threshold: float = 0.1
) -> dict:
    """
    Computes persistence summaries of H0 and H1 using persistent homology over sliding windows.
    
    Args:
        X (np.ndarray): Shape (T, N). The multivariate time series data.
        window_size (int): Size of the sliding window used for evaluation.
        stride (int, optional): Number of windows to skip between computations. Higher stride vastly improves performance. Defaults to 5.
        mode (str, optional): Computation mode. "fast" uses `stride`. "full" ignores stride and computes every window (T-window_size). Defaults to "fast".
        distance_metric (str, optional): How Kendall Tau is mapped to distance. "abs_kendall" uses 1 - |tau|. "signed_kendall" uses 0.5 * (1 - tau). Defaults to "abs_kendall".
        persistence_threshold (float, optional): Minimum lifespan (death - birth) for a topological feature to be considered significant and not noise. Defaults to 0.1.
        
    Returns:
        dict: A dictionary containing arrays of shape (T,) padded with NaNs where computation was skipped, plus 'computation_windows' indicating valid indices.
    """
    if not HAS_RIPSER:
        raise ImportError("The 'ripser' library is required for Geospatial Topology (TDA). Please run: pip install ripser")

    T, N = X.shape
    if N < 3:
        raise ValueError("TDA requires at least 3 components/nodes to form meaningful simplicial complexes.")

    tda_total_persistence_h1 = np.full(T, np.nan)
    tda_max_persistence_h1 = np.full(T, np.nan)
    tda_n_significant_holes = np.full(T, np.nan)
    tda_persistence_entropy_h1 = np.full(T, np.nan)
    
    tda_n_connected_components_h0 = np.full(T, np.nan)
    tda_max_persistence_h0 = np.full(T, np.nan)
    # Generate indices for computation based on stride
    if mode == "full":
        stride = 1
        
    calc_indices = list(range(window_size - 1, T, stride))
    
    for t in calc_indices:
        window = X[t - window_size + 1 : t + 1, :]
        
        # 1. Compute Kendall Tau Matrix
        if HAS_NUMBA:
            tau_matrix = _kendall_tau_matrix_numba(window)
        else:
            warnings.warn("Numba is not available. TDA computation will be extremely slow. Consider disabling TDA or installing numba.", RuntimeWarning)
            import scipy.stats as stats
            tau_matrix = np.zeros((N, N))
            for i in range(N):
                for j in range(i + 1, N):
                    tau, _ = stats.kendalltau(window[:, i], window[:, j])
                    tau = tau if not np.isnan(tau) else 0.0
                    tau_matrix[i, j] = tau
                    tau_matrix[j, i] = tau
            np.fill_diagonal(tau_matrix, 1.0)
        
        # 2. Convert to Distance Metric
        if distance_metric == "abs_kendall":
            D = 1.0 - np.abs(tau_matrix)
        else: # "signed_kendall"
            D = 0.5 * (1.0 - tau_matrix)
            
        np.fill_diagonal(D, 0.0) # Ensure diagonal is strictly 0
        
        # Validate Distance Matrix (Symmetric, non-negative)
        D = np.maximum(D, 0.0)  # Avoid floating point negative zeros
        if not np.allclose(D, D.T):
            D = (D + D.T) / 2.0 # Force exact symmetry
        
        # 3. Persistent Homology
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # maxdim=1 computes H0 and H1
            res = ripser(D, distance_matrix=True, maxdim=1)
            
        diagrams = res['dgms']
        
        if len(diagrams) > 1:
            H1 = diagrams[1]
            
            # Filter by threshold and remove infinite lifespans
            valid_holes = []
            for birth, death in H1:
                if death != np.inf and (death - birth) > persistence_threshold:
                    valid_holes.append(death - birth)
            
            valid_holes = np.array(valid_holes)
            
            if len(valid_holes) > 0:
                total_pers = np.sum(valid_holes)
                max_pers = np.max(valid_holes)
                n_holes = len(valid_holes)
                
                # Persistence Entropy
                if len(valid_holes) > 1:
                    p = valid_holes / total_pers
                    entropy = -np.sum(p * np.log(p))
                else:
                    entropy = 0.0
            else:
                total_pers = 0.0
                max_pers = 0.0
                n_holes = 0
                entropy = 0.0
                
            tda_total_persistence_h1[t] = total_pers
            tda_max_persistence_h1[t] = max_pers
            tda_n_significant_holes[t] = n_holes
            tda_persistence_entropy_h1[t] = entropy
        
        if len(diagrams) > 0:
            H0 = diagrams[0]
            # For H0, death == np.inf represents the single connected component that spans the entire space
            valid_components = []
            for birth, death in H0:
                if death != np.inf and (death - birth) > persistence_threshold:
                    valid_components.append(death - birth)
            
            n_components = len(valid_components) + 1 # +1 for the infinite component
            max_pers_h0 = np.max(valid_components) if len(valid_components) > 0 else 0.0
            
            tda_n_connected_components_h0[t] = n_components
            tda_max_persistence_h0[t] = max_pers_h0

    return {
        "total_persistence_h1": tda_total_persistence_h1,
        "max_persistence_h1": tda_max_persistence_h1,
        "n_significant_holes": tda_n_significant_holes,
        "persistence_entropy_h1": tda_persistence_entropy_h1,
        "n_connected_components_h0": tda_n_connected_components_h0,
        "max_persistence_h0": tda_max_persistence_h0,
        "computation_windows": calc_indices
    }
