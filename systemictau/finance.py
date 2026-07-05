"""
Systemic Tau - Finance Extension
Specialized tools for analyzing market crashes and portfolio systemic risk using Kairological Dynamics.
"""
from systemictau import systemic_tau

def compute_market_crash_risk(prices_matrix):
    """
    Analyzes asset correlation collapse in financial markets.
    """
    res = systemic_tau(prices_matrix, window_size=20)
    return res
