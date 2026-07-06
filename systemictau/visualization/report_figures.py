import base64
import io
from typing import Dict, Optional

from systemictau.results import OntologicalAscentResult

import pandas as pd

def generate_report_figures(results: OntologicalAscentResult, ews_results: Optional[pd.DataFrame] = None) -> Dict[str, str]:
    """
    Generates all core topological figures for the academic report in base64 format.
    Wraps each figure in a try-except block to ensure robustness.
    """
    figures_dict = {}
    
    # Bridge to dict for visualization functions
    mean_dtk_open = 0.0
    if hasattr(results, "dtk_series") and len(results.dtk_series) > 0:
        import numpy as np
        valid = results.dtk_series[results.dtk_series > 0]
        if len(valid) > 0:
            mean_dtk_open = float(np.nanmean(valid))

    res_dict = {
        "X": results.X,
        "taus_global": results.taus_global,
        "taus_per_module": getattr(results, "taus_per_module", None),
        "T_series": results.T_series,
        "dtk_series": getattr(results, "dtk_series", None),
        "episodes": results.episodes,
        "t_frob": results.t_frob,
        "t_ks": results.t_ks,
        "t_star": results.t_star,
        "max_dist": results.frob_max if results.frob_max is not None else 0.0,
        "max_ks": results.ks_max if results.ks_max is not None else 0.0,
        "fractal_D": results.fractal_D,
        "metadata": results.metadata,
        "nonlinear_stats": {
            "hp_z": results.hp_z,
            "laminarity": results.lam,
            "trapping_time": results.tt,
            "max_M": results.M_max,
            "mean_M": results.M_mean,
            "mean_dtk_open": mean_dtk_open
        }
    }
    
    # Tier 1: Ontological Overview
    try:
        from systemictau.visualization.tier_viz import plot_ontological_overview
        import matplotlib.pyplot as plt
        fig_t1 = plot_ontological_overview(results)
        buf = io.BytesIO()
        fig_t1.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        figures_dict["Tier 1: Ontological Overview"] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig_t1)
    except Exception as e:
        print(f"Failed to generate Tier 1: {e}")
        
    # Removed Tier 2: Layer Details from report (per v4.6.0 plan)

    # Removed Classical Dual-Plot and Systemic Time Manifold from report
        
    # Plotly Topological Heatmap
    if res_dict.get("taus_per_module") is not None and len(res_dict["taus_per_module"]) > 0:
        try:
            from systemictau.visualization.monumental_viz import plot_topological_heatmap
            fig_heatmap = plot_topological_heatmap(res_dict["taus_per_module"], res_dict["t_star"])
            figures_dict["Topological Heatmap"] = base64.b64encode(fig_heatmap.to_image(format="png", scale=2)).decode('utf-8')
        except Exception as e:
            print(f"Failed to generate Topological Heatmap: {e}")
            
    # Removed 3D Phase Space Attractor from report
        
    # Plotly Unimodal Return Map (Cobweb)
    try:
        from systemictau.visualization.monumental_viz import plot_unimodal_return_map
        fig_unimodal = plot_unimodal_return_map(res_dict)
        figures_dict["Unimodal Return Map (Feigenbaum Universality)"] = base64.b64encode(fig_unimodal.to_image(format="png", scale=2)).decode('utf-8')
    except Exception as e:
        print(f"Failed to generate Unimodal Return Map: {e}")

    # Plotly EWS Dashboard
    if ews_results is not None:
        try:
            from systemictau.visualization.monumental_viz import plot_ews_dashboard
            fig_ews = plot_ews_dashboard(res_dict, ews_results)
            figures_dict["Early Warning Signals (Critical Slowing Down)"] = base64.b64encode(fig_ews.to_image(format="png", scale=2)).decode('utf-8')
        except Exception as e:
            print(f"Failed to generate EWS Dashboard: {e}")
        
    return figures_dict
