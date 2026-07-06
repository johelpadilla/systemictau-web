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
        # We track the maximum Systemic Tau strength as our test statistic
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            max_tau = float(np.nanmax(surr_res.taus_global)) if len(surr_res.taus_global) > 0 else 0.0
        surrogate_t_stars.append(max_tau)
        
    # 4. Statistical Analysis
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        real_max_tau = float(np.nanmax(real_result.taus_global)) if len(real_result.taus_global) > 0 else 0.0
    
    # Calculate the percentile of the real_max_tau within the surrogate distribution.
    # We want to know what percentage of surrogates have a WEAKER maximum tau than the real data.
    # A high percentile (e.g. 99%) means the real coupling strength is stronger than 99% of random surrogates.
    if len(surrogate_t_stars) > 0:
        weaker_surrogates = sum(1 for tau in surrogate_t_stars if tau < real_max_tau)
        percentile_rank = (weaker_surrogates / n_surrogates) * 100.0
    else:
        percentile_rank = 0.0
        
    # Significant if the real max_tau is greater than 95% of the surrogates (p < 0.05 equivalent for a right-tailed test)
    is_significant = percentile_rank >= 95.0
        
    execution_time = time.time() - start_time
    
    metadata = {
        "execution_time_seconds": round(execution_time, 2),
        "real_max_tau": real_max_tau,
        "surrogate_max_tau_mean": np.mean(surrogate_t_stars) if surrogate_t_stars else 0.0,
        "surrogate_max_tau_std": np.std(surrogate_t_stars) if surrogate_t_stars else 0.0
    }
    
    return SurrogateValidationResult(
        real_result=real_result,
        surrogate_t_stars=surrogate_t_stars, # this now stores max_tau values
        percentile_rank=percentile_rank,
        is_significant=is_significant,
        n_surrogates=n_surrogates,
        mode=mode,
        metadata=metadata
    )
