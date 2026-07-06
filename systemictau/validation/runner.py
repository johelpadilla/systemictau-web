import numpy as np
from typing import Optional, Dict, Any
from .surrogates import generate_iaaft_surrogates
from .results import SurrogateValidationResult
from ..analysis import run_full_analysis
import time

def run_surrogate_validation(
    X_real: np.ndarray,
    n_surrogates: int = 500,
    mode: str = "fast",
    window_size: int = 13,
    theta_A: float = 0.05,
    seed: Optional[int] = None,
    **kwargs
) -> SurrogateValidationResult:
    """
    Executes the Surrogate Validation pipeline using IAAFT.
    
    Args:
        X_real (np.ndarray): The empirical multivariate time series.
        n_surrogates (int): Number of surrogates to generate. Default 500.
        mode (str): "fast" (extracts only t*) or "full" (runs complete engine).
        window_size (int): Systemic Tau sliding window size.
        theta_A (float): Threshold for episodes.
        seed (int): Random seed.
        **kwargs: Passed to run_full_analysis.
        
    Returns:
        SurrogateValidationResult: The final validation result.
    """
    start_time = time.time()
    
    # 1. Run empirical analysis
    real_result = run_full_analysis(
        X_real, 
        window_size=window_size, 
        theta_A=theta_A, 
        **kwargs
    )
    
    real_t_star = real_result.t_star
    
    # 2. Generate surrogates
    surrogates = generate_iaaft_surrogates(
        X_real, 
        n_surrogates=n_surrogates, 
        seed=seed
    )
    
    # 3. Process surrogates
    surrogate_t_stars = []
    
    for i in range(n_surrogates):
        X_surr = surrogates[i]
        
        # We always use run_full_analysis but in 'fast' mode we might bypass fractal
        # For now, we just run the standard engine. The C-extensions / numba make it fast enough.
        validate_fractal = False if mode == "fast" else True
        
        surr_res = run_full_analysis(
            X_surr,
            window_size=window_size,
            theta_A=theta_A,
            validate_fractal=validate_fractal,
            **kwargs
        )
        surrogate_t_stars.append(surr_res.t_star)
        
    # 4. Statistical Analysis
    if real_t_star is None:
        percentile_rank = 0.0
        is_significant = False
    else:
        # We want to see how unusual the empirical t_star is.
        # A valid surrogate t_star means the linear properties ALONE were enough to 
        # trigger a phase transition detection. 
        # The number of times surrogates generated ANY t* is our primary metric.
        valid_surr_t = [t for t in surrogate_t_stars if t is not None]
        
        # P-value is roughly the fraction of surrogates that yielded a valid t*
        # If t* is rarely found in surrogates, the real t* is highly significant.
        false_positive_rate = len(valid_surr_t) / n_surrogates
        
        # If we have a specific hypothesis about WHEN t* occurs, we could compare the timing.
        # But generally, just the existence of t* in independent random-phase data is the null.
        percentile_rank = (1.0 - false_positive_rate) * 100.0
        is_significant = false_positive_rate < 0.05
        
    execution_time = time.time() - start_time
    
    metadata = {
        "execution_time_seconds": round(execution_time, 2),
        "false_positive_rate": false_positive_rate if real_t_star is not None else 1.0
    }
    
    return SurrogateValidationResult(
        real_result=real_result,
        surrogate_t_stars=surrogate_t_stars,
        percentile_rank=percentile_rank,
        is_significant=is_significant,
        n_surrogates=n_surrogates,
        mode=mode,
        metadata=metadata
    )
