import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
import warnings

from .core import compute_taus
from .recd import accumulate_time
from .layers import (
    hyper_persistence,
    rolling_rqa,
    critical_mass_metric,
    compute_antisynchronization,
    extract_joint_episodes,
    detect_reorganization_frob,
    detect_reorganization_ks,
    consensus_transition
)
from .fractal import estimate_higuchi_dimension


def prepare_multivariate_timeseries(
    df: pd.DataFrame, 
    time_cols: Union[List[str], str], 
    value_cols: Union[List[str], str], 
    agg: Union[Dict[str, str], str] = "mean",
    sort: bool = True,
    dropna: bool = True
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Converts a DataFrame (potentially in long panel format) into a multivariate 
    matrix (T, N) ready for Systemic Tau analysis.
    
    Parameters:
    - df: The raw dataframe.
    - time_cols: Columns defining chronological order.
    - value_cols: Numeric columns to use as components.
    - agg: Aggregation method when multiple rows exist per time combination.
    - sort: Whether to sort chronologically by time_cols.
    - dropna: Whether to drop resulting rows with NaNs.
    
    Returns:
    - X (np.ndarray): Shape (T, N) multivariate matrix ready for the math engine.
    - weekly (pd.DataFrame): The aggregated dataframe for reference.
    """
    if isinstance(time_cols, str):
        time_cols = [time_cols]
    if isinstance(value_cols, str):
        value_cols = [value_cols]
        
    if not all(col in df.columns for col in time_cols):
        raise ValueError(f"Time columns {time_cols} not found in the dataset.")
    if not all(col in df.columns for col in value_cols):
        raise ValueError(f"Value columns {value_cols} not found in the dataset.")
        
    # Build aggregation dictionary
    if isinstance(agg, str):
        agg_dict = {col: agg for col in value_cols}
    else:
        # User provided explicit dict, ensure all value_cols have an agg function, fallback to mean
        agg_dict = {col: agg.get(col, "mean") for col in value_cols}

    weekly = df.groupby(time_cols).agg(agg_dict).reset_index()
    
    if sort:
        weekly = weekly.sort_values(time_cols).reset_index(drop=True)
        
    if dropna:
        weekly = weekly.dropna(subset=value_cols).reset_index(drop=True)
        
    # Check length and NaNs (just warnings/exceptions to protect math engine)
    if len(weekly) < 50:
        import warnings
        warnings.warn(f"Aggregated time series is very short (T={len(weekly)}). Minimum 50 points recommended for robust systemic analysis.")
        
    nan_count = weekly[value_cols].isna().sum().sum()
    if nan_count > 0:
        raise ValueError(f"Aggregated matrix contains {nan_count} NaNs. Numba math engine will fail. Consider imputation before this step or dropna=True.")
        
    X = weekly[value_cols].values
    
    return X, weekly


def run_full_tau_analysis(X: np.ndarray, window_size: int = 13, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Executes the entire mathematically rigorous Systemic Tau pipeline (Capas 1 to 3 + fractal validation)
    on a prepared multivariate time series matrix X.
    
    Returns:
    - dict: A complete, typed dictionary ready for academic report generation.
    """
    if metadata is None:
        metadata = {
            "window_size": window_size,
            "dataset": "Aggregated Panel",
            "date": pd.Timestamp.now().strftime("%Y-%m-%d"),
            "package_version": "3.0.0"
        }
    metadata["N_components"] = X.shape[1] if X.ndim > 1 else 1
    metadata["T_points"] = X.shape[0]
    
    # 1. Tau Sistémico
    taus_global, taus_per_module = compute_taus(X, window_size=window_size)

    # 2. RECD
    T_series, dtk_series, gate_series, depths = accumulate_time(taus_global)

    # 3. Capa 1
    hp_z, core_hyper = hyper_persistence(taus_global)
    lam, tt = rolling_rqa(taus_global)
    M_series = critical_mass_metric(hp_z, lam, tt)

    # 4. Capa 2 - Teorema 24v: Mapeo de Árbol Diádico Unimodal
    # E[|tau|]_k = 2^(k-1) / (2^k - 1) => k = log2(|tau| / (2*|tau| - 1))
    dyadic_k_series = np.zeros_like(taus_global)
    for i, t_val in enumerate(np.abs(taus_global)):
        if pd.isna(t_val):
            dyadic_k_series[i] = np.nan
        elif t_val <= 0.500001:
            dyadic_k_series[i] = np.inf
        elif t_val >= 0.999:
            dyadic_k_series[i] = 1.0
        else:
            dyadic_k_series[i] = np.log2(t_val / (2 * t_val - 1))
            
    estimated_period_series = 2 ** dyadic_k_series

    # 5. Capa 2 - Joint Episodes
    if taus_per_module is not None and taus_per_module.ndim > 1:
        A_series = compute_antisynchronization(taus_per_module)
        episodes = extract_joint_episodes(A_series, M_series, theta_A=0.05, D_min=10)
    else:
        episodes = []

    # 5. Capa 3 - Reorganización
    t_frob = None
    max_dist = 0.0
    if taus_per_module is not None and taus_per_module.ndim > 1:
        t_frob, max_dist = detect_reorganization_frob(taus_per_module)
        
    t_ks, max_ks = detect_reorganization_ks(dtk_series)
    t_star = consensus_transition(t_frob, t_ks)

    # 6. Validación fractal
    try:
        D = estimate_higuchi_dimension(dtk_series, k_max=20)
    except Exception:
        D = 1.0 # fallback if it fails on very short series

    # 7. Descriptive Stats Compilation
    max_M = float(np.max(M_series))
    mean_M = float(np.mean(M_series))
    lam_mean = float(np.mean(lam)) if isinstance(lam, np.ndarray) else float(lam)
    tt_mean = float(np.mean(tt)) if isinstance(tt, np.ndarray) else float(tt)
    mean_dtk_open = float(np.nanmean(dtk_series[dtk_series > 0])) if np.any(dtk_series > 0) else 0.0

    nonlinear_stats = {
        "hp_z": float(np.nanmean(np.abs(hp_z))) if len(hp_z) > 0 else 0.0,
        "laminarity": lam_mean,
        "trapping_time": tt_mean,
        "max_M": max_M,
        "mean_M": mean_M,
        "mean_dtk_open": mean_dtk_open
    }

    # Package the final dictionary
    results: Dict[str, Any] = {
        "X": X,
        "taus_global": taus_global,
        "taus_per_module": taus_per_module if taus_per_module is not None else np.array([]),
        "T_series": T_series,
        "dtk_series": dtk_series,
        "gate_series": gate_series,
        "depths": depths,
        "episodes": episodes,
        "dyadic_k_series": dyadic_k_series,
        "estimated_period_series": estimated_period_series,
        "t_frob": t_frob,
        "t_ks": t_ks,
        "t_star": t_star,
        "max_dist": float(max_dist),
        "max_ks": float(max_ks),
        "fractal_D": float(D),
        "metadata": metadata,
        "figures": None,
        "nonlinear_stats": nonlinear_stats,
        "episodes_table": episodes  # Passed explicitly for the report
    }
    
    return results
