import plotly.graph_objects as go
import numpy as np
from typing import Dict, Any

# Monumental Theme Palette
COLORS = {
    'bg': '#0f172a',           # Deep slate
    'tau': '#2dd4bf',          # Teal
    'recd': '#f43f5e',         # Rose
    'tstar': '#f97316',        # Orange
    'dtk': '#38bdf8',          # Sky
    'grid': 'rgba(255,255,255,0.05)',
    'text': '#f8fafc'
}

def plot_3d_strange_attractor(results: Dict[str, Any]) -> go.Figure:
    """
    Renders the Systemic Tau trajectory as a 3D Strange Attractor in Phase Space.
    Axes: 
    X: Systemic Tau (τ_s)
    Y: Velocity (dτ_s/dt)
    Z: Discretization Depth (k)
    """
    taus = np.asarray(results["taus_global"])
    dtk = np.asarray(results["dtk_series"])
    tstar = results["t_star"]
    
    # Calculate Velocity (simple derivative)
    velocity = np.zeros_like(taus)
    velocity[1:] = np.diff(taus)
    
    # Estimate depth k from dtk
    # We know dtk = delta^-k * |tau| * dt0/n, so k is proportional to the spikes
    # For visualization, we'll map large dtk spikes to deep k values.
    k_depth = np.zeros_like(taus)
    chaotic_mask = np.abs(taus) < 0.41
    k_depth[chaotic_mask] = np.log1p(dtk[chaotic_mask] * 100) # Pseudo-depth for visual scaling

    valid = ~np.isnan(taus)
    t_idx = np.arange(len(taus))
    
    fig = go.Figure()
    
    # Main Trajectory
    fig.add_trace(go.Scatter3d(
        x=taus[valid],
        y=velocity[valid],
        z=k_depth[valid],
        mode='lines+markers',
        line=dict(
            color=t_idx[valid],
            colorscale='Viridis',
            width=4
        ),
        marker=dict(
            size=2,
            color=t_idx[valid],
            colorscale='Viridis',
            opacity=0.8
        ),
        name='System Trajectory'
    ))
    
    # Highlight T*
    if tstar is not None and not np.isnan(tstar):
        tstar = int(tstar)
        fig.add_trace(go.Scatter3d(
            x=[taus[tstar]],
            y=[velocity[tstar]],
            z=[k_depth[tstar]],
            mode='markers',
            marker=dict(size=10, color=COLORS['tstar'], symbol='diamond'),
            name='t* (Critical Point)'
        ))
        
    # The Chaotic Band (Feigenbaum Gate) Wall at ±0.41
    fig.add_trace(go.Surface(
        x=np.array([[0.41, 0.41], [0.41, 0.41]]),
        y=np.array([[np.nanmin(velocity), np.nanmax(velocity)], [np.nanmin(velocity), np.nanmax(velocity)]]),
        z=np.array([[0, 0], [np.nanmax(k_depth), np.nanmax(k_depth)]]),
        colorscale=[[0, 'rgba(244, 63, 94, 0.2)'], [1, 'rgba(244, 63, 94, 0.2)']],
        showscale=False,
        name='Chaotic Gate (0.41)'
    ))
    
    fig.update_layout(
        title="3D Phase Space Attractor (Tau vs Velocity vs Fractal Depth)",
        scene=dict(
            xaxis_title="Systemic Tau (τ_s)",
            yaxis_title="Velocity (dτ_s/dt)",
            zaxis_title="Fractal Depth (k)",
            bgcolor=COLORS['bg'],
            xaxis=dict(gridcolor=COLORS['grid'], zerolinecolor=COLORS['grid']),
            yaxis=dict(gridcolor=COLORS['grid'], zerolinecolor=COLORS['grid']),
            zaxis=dict(gridcolor=COLORS['grid'], zerolinecolor=COLORS['grid'])
        ),
        paper_bgcolor=COLORS['bg'],
        font=dict(color=COLORS['text']),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig

def plot_discretization_manifold(results: Dict[str, Any]) -> go.Figure:
    """
    Renders the true geometric meaning of RECD: The Discretization Manifold.
    Plots Chronological Time (t) vs Systemic Time (T_RECD).
    Shows how time literally stops/plateaus at the critical point.
    """
    accum_t = np.asarray(results["T_series"])
    t_star = results.get("t_star")
    t_chrono = np.arange(len(accum_t))
    valid = ~np.isnan(accum_t)
    
    fig = go.Figure()
    
    # The Manifold Curve
    fig.add_trace(go.Scatter(
        x=t_chrono[valid],
        y=accum_t[valid],
        mode='lines',
        line=dict(color=COLORS['dtk'], width=3, shape='vh'), # 'vh' for step-like discrete jumps
        fill='tozeroy',
        fillcolor='rgba(56, 189, 248, 0.1)',
        name='Systemic Time T_RECD'
    ))
    
    # Critical point marker
    if t_star is not None and not np.isnan(t_star):
        t_star = int(t_star)
        fig.add_vline(x=t_star, line_dash="dash", line_color=COLORS['tstar'], annotation_text="t* Collapse")
        fig.add_trace(go.Scatter(
            x=[t_star],
            y=[accum_t[t_star]],
            mode='markers',
            marker=dict(color=COLORS['tstar'], size=12, symbol='star'),
            name='Singularity'
        ))

    fig.update_layout(
        title="Discretization Manifold: Chronological vs Systemic Time",
        xaxis_title="Chronological Time (t)",
        yaxis_title="Accumulated Systemic Time (T_RECD)",
        plot_bgcolor=COLORS['bg'],
        paper_bgcolor=COLORS['bg'],
        font=dict(color=COLORS['text']),
        xaxis=dict(gridcolor=COLORS['grid']),
        yaxis=dict(gridcolor=COLORS['grid']),
        hovermode="x unified"
    )
    return fig

def plot_topological_heatmap(taus_per_module: np.ndarray, t_star: int) -> go.Figure:
    """
    Renders the exact topological synchronization matrix (T x N).
    Visualizes how the spatial modules lock together before the collapse.
    """
    if taus_per_module is None or len(taus_per_module.shape) < 2:
        return go.Figure()
        
    T, N = taus_per_module.shape
    
    fig = go.Figure(data=go.Heatmap(
        z=taus_per_module.T,
        colorscale='RdBu',
        zmid=0,
        colorbar=dict(title="Kendall τ")
    ))
    
    if t_star is not None and not np.isnan(t_star):
        t_star = int(t_star)
        fig.add_vline(x=t_star, line_color=COLORS['tstar'], line_width=3, line_dash="solid", annotation_text="t*")

    fig.update_layout(
        title="Topological Synchronization Heatmap (Space vs Time)",
        xaxis_title="Time Step (t)",
        yaxis_title="Spatial Component (Module index)",
        plot_bgcolor=COLORS['bg'],
        paper_bgcolor=COLORS['bg'],
        font=dict(color=COLORS['text'])
    )
    return fig

def plot_ews_dashboard(results: Dict[str, Any], ews: Dict[str, Any]) -> go.Figure:
    """
    Renders Early Warning Signals (Variance, AC1, Skewness) alongside Systemic Tau.
    Provides visual proof of Critical Slowing Down before t*.
    """
    taus = np.asarray(results["taus_global"])
    t_star = results.get("t_star")
    variance = ews["variance"]
    ac1 = ews["ac1"]
    
    t_chrono = np.arange(len(taus))
    
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        subplot_titles=("Systemic Tau (τ_s)", "Rolling Variance (Resilience Loss)", "Lag-1 Autocorrelation (Critical Slowing Down)"),
                        vertical_spacing=0.08)
    
    # 1. Tau
    fig.add_trace(go.Scatter(x=t_chrono, y=taus, mode='lines', line=dict(color=COLORS['tau'], width=2), name="τ_s"), row=1, col=1)
    
    # 2. Variance
    fig.add_trace(go.Scatter(x=t_chrono, y=variance, mode='lines', line=dict(color=COLORS['dtk'], width=2), name="Variance", fill='tozeroy'), row=2, col=1)
    
    # 3. AC1
    fig.add_trace(go.Scatter(x=t_chrono, y=ac1, mode='lines', line=dict(color=COLORS['tstar'], width=2), name="AC1"), row=3, col=1)
    
    # Add t* marker to all subplots
    if t_star is not None and not np.isnan(t_star):
        t_star = int(t_star)
        for i in range(1, 4):
            fig.add_vline(x=t_star, line_dash="dash", line_color=COLORS['recd'], row=i, col=1)
            
    fig.update_layout(
        height=700,
        plot_bgcolor=COLORS['bg'],
        paper_bgcolor=COLORS['bg'],
        font=dict(color=COLORS['text']),
        hovermode="x unified",
        showlegend=False
    )
    
    for i in range(1, 4):
        fig.update_xaxes(gridcolor=COLORS['grid'], row=i, col=1)
        fig.update_yaxes(gridcolor=COLORS['grid'], row=i, col=1)
        
    return fig
