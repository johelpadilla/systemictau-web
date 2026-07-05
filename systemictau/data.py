import numpy as np
from typing import List, Optional, Any
from .core import systemic_tau, SystemicTauResult

def preprocess(X: np.ndarray, method: str = 'linear') -> np.ndarray:
    """
    Preprocesses the time series array by handling missing values (NaNs).
    
    Parameters:
    -----------
    X : np.ndarray
        Multivariate time series array of shape (T, N).
    method : str, optional
        Imputation method. Options: 
        - 'linear': linear interpolation along the time axis.
        - 'ffill': forward fill, then backward fill.
        - 'drop': drop any row with at least one NaN.
        
    Returns:
    --------
    np.ndarray
        Preprocessed array.
    """
    X_out = X.copy()
    if method == 'drop':
        # Drop rows with any NaN
        valid_rows = ~np.isnan(X_out).any(axis=1)
        return X_out[valid_rows]
        
    for j in range(X_out.shape[1]):
        col = X_out[:, j]
        nans = np.isnan(col)
        if not nans.any():
            continue
            
        if method == 'linear':
            # Linear interpolation
            not_nans = ~nans
            col[nans] = np.interp(nans.nonzero()[0], not_nans.nonzero()[0], col[not_nans])
        elif method == 'ffill':
            # Forward fill
            idx = np.arange(len(col))
            # Get the index of the last valid value for each position
            valid_idx = np.where(~nans)[0]
            if len(valid_idx) == 0:
                continue # all NaNs
            
            # Create an array of previous valid indices
            # For NaNs before the first valid value, use the first valid value (bfill)
            prev_valid = np.maximum.accumulate(np.where(~nans, idx, 0))
            
            # Handle leading NaNs by setting their prev_valid to the first valid index
            first_valid = valid_idx[0]
            prev_valid[:first_valid] = first_valid
            
            col[:] = col[prev_valid]
        else:
            raise ValueError(f"Unknown imputation method: {method}")
            
    return X_out

def from_dataframe(df: Any, time_col: Optional[str] = None, value_cols: Optional[List[str]] = None, 
                   imputation: str = 'linear', window_size: int = 13, **kwargs) -> SystemicTauResult:
    """
    Computes Systemic Tau directly from a pandas DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe.
    time_col : str, optional
        The column to use as time (useful for sorting). If None, relies on index.
    value_cols : list of str, optional
        The columns to compute tau on. If None, uses all numeric columns except time_col.
    imputation : str, optional
        The missing data imputation method ('linear', 'ffill', 'drop').
    window_size : int, optional
        The sliding window size.
    **kwargs
        Passed to `systemic_tau`.
        
    Returns:
    --------
    SystemicTauResult
    """
    # Guard import
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for from_dataframe. Install via 'pip install pandas'")
        
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
        
    df_work = df.copy()
    
    if time_col is not None:
        df_work = df_work.sort_values(time_col)
    
    if value_cols is None:
        # Select numeric columns
        cols = df_work.select_dtypes(include=[np.number]).columns.tolist()
        if time_col in cols:
            cols.remove(time_col)
        value_cols = cols
        
    X = df_work[value_cols].values
    
    # Preprocess
    X_clean = preprocess(X, method=imputation)
    
    return systemic_tau(X_clean, window_size=window_size, **kwargs)

def from_xarray(da: Any, time_dim: str = 'time', feature_dim: str = 'feature', 
                imputation: str = 'linear', window_size: int = 13, **kwargs) -> SystemicTauResult:
    """
    Computes Systemic Tau directly from an xarray DataArray.
    
    Parameters:
    -----------
    da : xarray.DataArray
        The input data array.
    time_dim : str
        The dimension representing time.
    feature_dim : str
        The dimension representing the multivariate components.
    imputation : str
        The missing data imputation method.
    window_size : int
        The sliding window size.
    **kwargs
        Passed to `systemic_tau`.
        
    Returns:
    --------
    SystemicTauResult
    """
    # Guard import
    try:
        import xarray as xr
    except ImportError:
        raise ImportError("xarray is required for from_xarray. Install via 'pip install xarray'")
        
    if not isinstance(da, xr.DataArray):
        raise TypeError("da must be an xarray DataArray")
        
    # Ensure correct dimension order (time, feature)
    da_work = da.transpose(time_dim, feature_dim, ...)
    
    X = da_work.values
    if X.ndim > 2:
        # Flatten extra dimensions or raise an error. For now, assume 2D.
        raise ValueError(f"DataArray must be 2D along {time_dim} and {feature_dim}")
        
    X_clean = preprocess(X, method=imputation)
    return systemic_tau(X_clean, window_size=window_size, **kwargs)
