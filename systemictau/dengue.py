"""
Systemic Tau - Epidemiology (Dengue) Extension
Specialized tools for epidemiological surveillance using Kairological Dynamics.
"""
from systemictau.spatial import spatial_tau

def compute_dengue_outbreak_risk(cases_df, window_size=13, spatial_k=4):
    """
    Computes an outbreak risk score by combining temporal Systemic Tau
    with spatial synchrony hotspots across neighboring regions.
    """
    # In a full implementation, this uses evaluate_early_warning on the Critical Mass
    res = spatial_tau(cases_df, value_cols=['cases'], k_neighbors=spatial_k, window_size=window_size)
    return res
