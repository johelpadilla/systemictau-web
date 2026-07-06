import numpy as np
import scipy.stats as stats
import itertools
from dataclasses import dataclass, field
from typing import Dict, Any
import time
import logging

logger = logging.getLogger(__name__)

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    
    def njit(*args, **kwargs):
        def wrapper(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return wrapper

def _kendall_tau_fast(x, y):
    """
    Computes Kendall's tau between two 1D arrays of ranks.
    Uses scipy.stats.kendalltau.
    """
    tau, _ = stats.kendalltau(x, y)
    return tau if not np.isnan(tau) else 0.0

@njit
def _kendall_tau_numba(x, y):
    n = len(x)
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            sign_x = 1 if x[i] < x[j] else (-1 if x[i] > x[j] else 0)
            sign_y = 1 if y[i] < y[j] else (-1 if y[i] > y[j] else 0)
            if sign_x * sign_y > 0:
                concordant += 1
            elif sign_x * sign_y < 0:
                discordant += 1
    total = n * (n - 1) / 2
    if total == 0:
        return 0.0
    return (concordant - discordant) / total

@njit
def _compute_taus_numba(X, window_size, stride):
    T, N = X.shape
    taus_global = np.full(T, np.nan)
    taus_per_module = np.full((T, N), np.nan)
    
    for t in range(window_size - 1, T, stride):
        window = X[t - window_size + 1 : t + 1, :]
        
        tau_matrix = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                tau = _kendall_tau_numba(window[:, i], window[:, j])
                tau_matrix[i, j] = tau
                tau_matrix[j, i] = tau
                
        sum_tau = 0.0
        count = 0
        for i in range(N):
            for j in range(i + 1, N):
                sum_tau += tau_matrix[i, j]
                count += 1
        taus_global[t] = sum_tau / count if count > 0 else np.nan
        
        for i in range(N):
            sum_mod = 0.0
            for j in range(N):
                if i != j:
                    sum_mod += tau_matrix[i, j]
            taus_per_module[t, i] = sum_mod / (N - 1) if N > 1 else np.nan
            
    return taus_global, taus_per_module

@njit
def _compute_taus_adaptive_numba(X, base_window_size, stride):
    T, N = X.shape
    taus_global = np.full(T, np.nan)
    taus_per_module = np.full((T, N), np.nan)
    
    # 1. Global Baseline Volatility (MASD)
    sum_diffs = 0.0
    count_diffs = 0
    for t in range(1, T):
        for i in range(N):
            val = abs(X[t, i] - X[t - 1, i])
            if not np.isnan(val):
                sum_diffs += val
                count_diffs += 1
                
    V_base = (sum_diffs / count_diffs) if count_diffs > 0 else 1e-9
    if V_base == 0:
        V_base = 1e-9
        
    w_min = max(7, int(base_window_size * 0.5))
    w_max = min(T, int(base_window_size * 2.0))
    
    for t in range(base_window_size - 1, T, stride):
        # 2. Local Volatility
        local_sum = 0.0
        local_count = 0
        for k in range(t - base_window_size + 2, t + 1):
            if k < 1: continue
            for i in range(N):
                val = abs(X[k, i] - X[k - 1, i])
                if not np.isnan(val):
                    local_sum += val
                    local_count += 1
                    
        V_t = (local_sum / local_count) if local_count > 0 else V_base
        
        # 3. Regime Ratio
        R_t = V_t / V_base
        
        # 4. Adaptive Window W_t
        if R_t <= 0:
            W_t = w_max
        else:
            W_t = int(base_window_size / R_t)
            
        if W_t < w_min: W_t = w_min
        if W_t > w_max: W_t = w_max
        
        if t - W_t + 1 < 0:
            W_t = t + 1
            
        window = X[t - W_t + 1 : t + 1, :]
        
        tau_matrix = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                tau = _kendall_tau_numba(window[:, i], window[:, j])
                tau_matrix[i, j] = tau
                tau_matrix[j, i] = tau
                
        sum_tau = 0.0
        count = 0
        for i in range(N):
            for j in range(i + 1, N):
                sum_tau += tau_matrix[i, j]
                count += 1
        taus_global[t] = sum_tau / count if count > 0 else np.nan
        
        for i in range(N):
            sum_mod = 0.0
            for j in range(N):
                if i != j:
                    sum_mod += tau_matrix[i, j]
            taus_per_module[t, i] = sum_mod / (N - 1) if N > 1 else np.nan
            
    return taus_global, taus_per_module

def compute_taus(X, window_size=13, stride=1, adaptive=False):
    """
    Computes Systemic Tau and per-module tau over a sliding window.
    Uses Numba if available for massive speedup.
    """
    T, N = X.shape
    if N < 2:
        raise ValueError("At least 2 components are required.")
        
    if HAS_NUMBA:
        if adaptive:
            return _compute_taus_adaptive_numba(X, window_size, stride)
        else:
            return _compute_taus_numba(X, window_size, stride)
        
    # Only warn once per session (called 3x per full analysis: Local/Medium/Global)
    if not getattr(compute_taus, "_numba_warned", False):
        logger.warning("Numba not found. Using scipy fallback (slower). "
                       "Install with: pip install numba for big speedups.")
        compute_taus._numba_warned = True
    taus_global = np.full(T, np.nan)
    taus_per_module = np.full((T, N), np.nan)
    
    for t in range(window_size - 1, T, stride):
        window = X[t - window_size + 1 : t + 1, :]
        ranks = stats.rankdata(window, axis=0)
        
        tau_matrix = np.zeros((N, N))
        for i, j in itertools.combinations(range(N), 2):
            tau = _kendall_tau_fast(ranks[:, i], ranks[:, j])
            tau_matrix[i, j] = tau
            tau_matrix[j, i] = tau
            
        upper_tri_indices = np.triu_indices(N, k=1)
        taus_global[t] = np.mean(tau_matrix[upper_tri_indices])
        
        for i in range(N):
            mask = np.ones(N, dtype=bool)
            mask[i] = False
            taus_per_module[t, i] = np.mean(tau_matrix[i, mask])
            
    return taus_global, taus_per_module


@dataclass
class SystemicTauResult:
    """
    Data class representing the result of a Systemic Tau computation.
    """
    taus_global: np.ndarray
    taus_per_module: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


def compute_taus_dask(X_dask, window_size=13, stride=1):
    """
    Computes Systemic Tau on a Dask Array for out-of-core streaming computations.
    """
    try:
        import dask.array as da
    except ImportError:
        raise ImportError("Dask is required. Run 'pip install systemictau[performance]'")
        
    T, N = X_dask.shape
    
    # We will use map_blocks to apply the numba function to chunks of data.
    # Note: This is a simplified rolling window approach for demonstration.
    # Proper rolling windows on dask chunks require overlapping chunks.
    
    # For now, we will compute on overlap blocks
    depth = {0: window_size, 1: 0}
    
    def block_compute(chunk):
        # chunk shape will be (T_chunk + window_size, N)
        # However, due to boundary conditions, some chunks may be smaller
        if chunk.shape[0] < window_size:
            # Not enough data for a single window
            return np.full((chunk.shape[0], N+1), np.nan)
        
        tg, tm = compute_taus(chunk, window_size=window_size, stride=stride)
        # We concatenate tg and tm to return a single array per chunk
        return np.hstack((tg[:, None], tm))
        
    res = da.map_overlap(
        block_compute,
        X_dask,
        depth=depth,
        boundary='none',
        trim=False,
        dtype=float,
        new_axis=None # since input is 2D and output is 2D
    )
    
    # Actually, map_overlap changes shape. The output shape should be (T, N+1).
    # This is a conceptual implementation of Dask streaming.
    return res

def _compute_taus_jax(X, window_size=13, stride=1):
    """
    Experimental JAX backend for Systemic Tau (Hardware accelerated GPU/TPU).
    """
    try:
        import jax.numpy as jnp
        from jax import jit, vmap  # noqa: F401
    except ImportError:
        raise ImportError("JAX is required. Run 'pip install systemictau[performance]'")
        
    T, N = X.shape
    X_jnp = jnp.array(X)
    
    # Pre-allocate output arrays (JAX arrays are immutable, so we build them via vmap)
    # This is a placeholder for a true vectorized JAX implementation of Kendall Tau.
    # A full JAX implementation requires custom XLA primitives for rank sorting.
    # For now, we simulate the structure.
    
    @jit
    def jax_window_tau(window):
        # Simplified placeholder for JAX compilation
        return jnp.mean(jnp.var(window, axis=0))
        
    # Example logic (not fully equivalent to Numba Kendall Tau yet due to XLA constraints)
    taus_global = np.full(T, np.nan)
    taus_per_module = np.full((T, N), np.nan)
    
    for t in range(window_size - 1, T, stride):
        window = X_jnp[t - window_size + 1 : t + 1, :]
        taus_global[t] = jax_window_tau(window)
        taus_per_module[t, :] = jnp.zeros(N)
        
    return taus_global, taus_per_module

def systemic_tau(X: np.ndarray, window_size: int = 13, stride: int = 1, method: str = 'kendall', n_jobs: int = -1, engine: str = "numba") -> SystemicTauResult:
    """
    Unified entry point for computing Systemic Tau.
    """
    start_time = time.time()
    
    if method != 'kendall':
        raise NotImplementedError(f"Method '{method}' is not implemented yet. Only 'kendall' is supported.")
        
    if engine == "dask":
        res = compute_taus_dask(X, window_size=window_size, stride=stride)
        res_computed = res.compute()
        taus_global = res_computed[:, 0]
        taus_per_module = res_computed[:, 1:]
    elif engine == "jax":
        taus_global, taus_per_module = _compute_taus_jax(X, window_size=window_size, stride=stride)
    else:
        taus_global, taus_per_module = compute_taus(X, window_size=window_size, stride=stride)
    
    computation_time = time.time() - start_time
    metadata = {
        'window_size': window_size,
        'stride': stride,
        'method': method,
        'n_components': X.shape[1],
        'time_steps': X.shape[0],
        'computation_time_seconds': computation_time
    }
    
    return SystemicTauResult(
        taus_global=taus_global,
        taus_per_module=taus_per_module,
        metadata=metadata
    )
