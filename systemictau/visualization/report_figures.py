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
            
    # Plotly TDA Dashboard
    if hasattr(results, "tda_results") and results.tda_results is not None:
        try:
            import plotly.graph_objects as go
            tda = results.tda_results
            t_star = res_dict["t_star"]
            
            # Plot 1: Persistence Curves
            fig1 = go.Figure()
            tau = res_dict["taus_global"]
            fig1.add_trace(go.Scatter(y=tau, mode='lines', name='Systemic Tau', line=dict(color='rgba(200, 200, 200, 0.4)', width=3, dash='dot')))
            
            x_idx = tda['computation_windows']
            fig1.add_trace(go.Scatter(x=x_idx, y=[tda['total_persistence_h1'][i] for i in x_idx], mode='lines+markers', name='Total Persistence (H1)', line=dict(color='#ff4b4b', width=2), marker=dict(size=4)))
            fig1.add_trace(go.Scatter(x=x_idx, y=[tda['max_persistence_h1'][i] for i in x_idx], mode='lines+markers', name='Max Persistence (H1)', line=dict(color='#4b4bff', width=2), marker=dict(size=4)))
            
            if t_star is not None:
                fig1.add_vline(x=t_star, line_width=2, line_dash="dash", line_color="red", annotation_text="t*")
                
            fig1.update_layout(title="Systemic Transition vs Topological Holes", xaxis_title="Time Step (t)", yaxis_title="Persistence / Tau", height=400, template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
            
            figures_dict["TDA: Systemic Transition vs Topological Holes"] = base64.b64encode(fig1.to_image(format="png", scale=2)).decode('utf-8')
            
            # Plot 2: Entropy and Fragmentation
            fig2 = go.Figure()
            # H0 Components
            fig2.add_trace(go.Scatter(x=x_idx, y=[tda.get('n_connected_components_h0', [np.nan]*len(tau))[i] for i in x_idx], mode='lines+markers', name='Connected Components (H0)', line=dict(color='#00d2d3', width=2), yaxis='y1'))
            # Entropy
            fig2.add_trace(go.Scatter(x=x_idx, y=[tda['persistence_entropy_h1'][i] for i in x_idx], mode='lines+markers', name='Persistence Entropy', line=dict(color='#ff9f43', width=2), yaxis='y2'))
            
            if t_star is not None:
                fig2.add_vline(x=t_star, line_width=2, line_dash="dash", line_color="red", annotation_text="t*")
                
            fig2.update_layout(
                title="Topological Fragmentation vs Chaos", 
                xaxis_title="Time Step (t)", 
                yaxis=dict(title="H0 Components", side="left", showgrid=False),
                yaxis2=dict(title="Entropy", side="right", overlaying="y", showgrid=False),
                height=400, 
                template="plotly_white", 
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(x=0.01, y=0.99)
            )
            
            figures_dict["TDA: Topological Chaos"] = base64.b64encode(fig2.to_image(format="png", scale=2)).decode('utf-8')
        except Exception as e:
            print(f"Failed to generate TDA Dashboard: {e}")
            
    if hasattr(results, 'ordinal_results') and results.ordinal_results is not None:
        try:
            b64_ordinal = generate_ordinal_chart_base64(results)
            if b64_ordinal:
                figures_dict["Ordinal Memory Dynamics"] = b64_ordinal
        except Exception as e:
            print(f"Failed to generate Ordinal Memory chart: {e}")
            
    if hasattr(results, 'nested_recd_results') and results.nested_recd_results is not None:
        try:
            b64_recd = generate_nested_recd_chart_base64(results)
            if b64_recd:
                figures_dict["Nested RECD Extramental Clock"] = b64_recd
        except Exception as e:
            print(f"Failed to generate Nested RECD chart: {e}")
        
    return figures_dict
def generate_ordinal_chart_base64(res) -> str:
    """
    Generates a base64 encoded PNG of the Ordinal Memory dynamics.
    """
    import plotly.graph_objects as go
    
    ordinal = res.ordinal_results
    if not ordinal:
        return ""
        
    x_idx = ordinal['computation_windows']
    y_flow = ordinal['total_flow'][x_idx]
    y_asym = ordinal['net_asymmetry'][x_idx]
    mode = ordinal['mode']
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x_idx,
        y=y_flow,
        mode='lines+markers',
        name='Total Information Flow',
        line=dict(color='#00d2d3', width=2),
        marker=dict(size=4)
    ))
    
    if mode == 'full':
        fig.add_trace(go.Scatter(
            x=x_idx,
            y=y_asym,
            mode='lines+markers',
            name='Net Flow Asymmetry',
            line=dict(color='#ff9f43', width=2),
            marker=dict(size=4)
        ))
        
    if res.t_star is not None:
        fig.add_vline(x=res.t_star, line_width=2, line_dash="dash", line_color="red", annotation_text="t*")
        
    fig.update_layout(
        title="Ordinal Memory Dynamics (Rank MI / STE)",
        xaxis_title="Time Step (t)",
        yaxis_title="Information / Coupling",
        height=400,
        width=800,
        template="simple_white",
        legend=dict(x=0.01, y=0.99)
    )
    
    img_bytes = fig.to_image(format="png", engine="kaleido")
    return base64.b64encode(img_bytes).decode("utf-8")

def generate_nested_recd_chart_base64(res) -> str:
    """
    Generates a base64 encoded PNG of the Nested RECD Extramental Clock dashboard.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
    
    if not hasattr(res, "nested_recd_results") or not res.nested_recd_results:
        return ""
        
    recd = res.nested_recd_results
    t_recd = np.arange(len(recd["phi1"]))
    
    phi1 = np.asarray(recd["phi1"])
    phi2 = np.asarray(recd["phi2"])
    phi3 = np.asarray(recd["phi3"])
    excess3 = np.asarray(recd["excess3"])
    T_series = np.asarray(recd["T_recd"])
    
    contrib1 = np.asarray(recd.get("contrib1", phi1))
    contrib2 = np.asarray(recd.get("contrib2", phi2))
    contrib3 = np.asarray(recd.get("contrib3", phi3))
    
    total_contrib = contrib1 + contrib2 + contrib3
    frac3 = np.zeros_like(contrib3)
    valid = total_contrib > 0
    frac3[valid] = contrib3[valid] / total_contrib[valid]
    
    tstar = getattr(res, "t_star", None)
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Extramental Clock Accumulation (T_recd) & Excess3",
            "Nested Ordinal Contributions (αk * Φk)",
            "Fractional Contribution of Level 3 (Emergence)"
        )
    )
    
    # 1. T_recd and Excess3
    fig.add_trace(go.Scatter(
        x=t_recd, y=T_series, mode='lines', name='T_recd Clock',
        line=dict(color='#8e44ad', width=2)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=t_recd, y=excess3, mode='lines', name='Excess3 (Level 3 proxy)',
        line=dict(color='#f43f5e', width=1, dash='dot')
    ), row=1, col=1)
    
    # 2. Contributions
    fig.add_trace(go.Scatter(
        x=t_recd, y=contrib1, mode='lines', name='Contrib 1 (Φ1)',
        line=dict(color='#3498db', width=1.5)
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=t_recd, y=contrib2, mode='lines', name='Contrib 2 (Φ2)',
        line=dict(color='#2ecc71', width=1.5)
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=t_recd, y=contrib3, mode='lines', name='Contrib 3 (Φ3)',
        line=dict(color='#e74c3c', width=2)
    ), row=2, col=1)
    
    # 3. Fractional Contribution
    fig.add_trace(go.Scatter(
        x=t_recd, y=frac3, mode='lines', name='Frac Level 3',
        line=dict(color='#9b59b6', width=2),
        fill='tozeroy', fillcolor='rgba(155, 89, 182, 0.2)'
    ), row=3, col=1)
    fig.add_hline(y=1.0/3.0, line_dash="dot", line_color="#7f8c8d", annotation_text="1/3 uniform", row=3, col=1)
    
    if tstar is not None and tstar < len(t_recd):
        for i in range(1, 4):
            fig.add_vline(x=tstar, line_width=2, line_dash="dash", line_color="red", row=i, col=1)
    
    fig.update_layout(
        height=800,
        width=800,
        margin=dict(l=20, r=20, t=40, b=20),
        template='simple_white',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    img_bytes = fig.to_image(format="png", engine="kaleido", scale=2)
    return base64.b64encode(img_bytes).decode("utf-8")
