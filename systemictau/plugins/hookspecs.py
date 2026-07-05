import pluggy
import numpy as np

hookspec = pluggy.HookspecMarker("systemictau")

@hookspec
def compute_custom_correlation(X: np.ndarray) -> np.ndarray:
    """
    Computes a custom correlation or similarity metric across features in X.
    Allows 3rd party researchers to swap out Kendall Tau for mutual information, 
    Pearson, or custom topological metrics without forking the repo.
    """
    pass

@hookspec
def on_critical_mass_detected(t_star: int, episodes: list):
    """
    Triggered when a critical mass/ontological ascent is detected.
    """
    pass
