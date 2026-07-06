__version__ = "4.6.0"
from .core import compute_taus, systemic_tau, SystemicTauResult
from .recd import compute_recd_increments, accumulate_time, gate_function
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
from .reporting import generate_academic_report
from .panel import prepare_multivariate_timeseries, run_full_tau_analysis
from .analysis import run_full_analysis
from .results import OntologicalAscentResult
from .robustness import run_sensitivity_analysis
from .generators import ChaosGenerator
from .visualization import plot_tau_evolution, plot_joint_episodes, plot_ontological_layers
from .data import preprocess, from_dataframe, from_xarray
from .validation import evaluate_early_warning, run_surrogate_validation
from .spatial import spatial_tau
from .dengue import compute_dengue_outbreak_risk
from .climate import detect_climate_tipping_points
from .finance import compute_market_crash_risk

# Studio (web) is heavy; import on demand to keep base light
try:
    from . import studio  # noqa: F401
    HAS_STUDIO = True
except Exception:
    HAS_STUDIO = False

__all__ = [
    "systemic_tau",
    
    # Report Engine

    "generate_academic_report",
    
    "SystemicTauResult",
    "compute_taus",
    "run_full_tau_analysis",
    "prepare_multivariate_timeseries",
    
    # Orquestador del pipeline completo
    "run_full_analysis",
    "OntologicalAscentResult",
    
    # RECD & Time Accumulation
    "compute_recd_increments",
    "accumulate_time",
    "gate_function",
    "hyper_persistence",
    "rolling_rqa",
    "critical_mass_metric",
    "compute_antisynchronization",
    "extract_joint_episodes",
    "detect_reorganization_frob",
    "detect_reorganization_ks",
    "consensus_transition",
    "estimate_higuchi_dimension",
    
    "ChaosGenerator",
    "plot_tau_evolution",
    "plot_joint_episodes",
    "plot_ontological_layers",
    "preprocess",
    "from_dataframe",
    "from_xarray",
    "evaluate_early_warning",
    "spatial_tau",
    "compute_dengue_outbreak_risk",
    "detect_climate_tipping_points",
    "compute_market_crash_risk",
    "run_surrogate_validation",
    "studio",
]
