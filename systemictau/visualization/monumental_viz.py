import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

def plot_dyadic_cascade(results: Dict[str, Any]) -> go.Figure:
    """
    Renders the Dyadic Cascade Tree Mapping (Teorema 24v)
    Visualizes the mapped k-level across time.
    """
    if "dyadic_k_series" not in results:
        return go.Figure()
        
    k_series = results["dyadic_k_series"]
    tstar = results.get("t_star", 0)
    t_idx = np.arange(len(k_series))
    
    # Replace infinity with a high number (e.g. 20) for plotting purposes
    # representing Chaos / Feigenbaum Accumulation
    plot_k = np.copy(k_series)
    plot_k[np.isinf(plot_k)] = 20
    
    fig = go.Figure()
    
    # Chaos region background
    fig.add_hrect(y0=18, y1=22, line_width=0, fillcolor="red", opacity=0.1, annotation_text="Feigenbaum Chaos Accumulation", annotation_position="top left")
    
    fig.add_trace(go.Scatter(
        x=t_idx,
        y=plot_k,
        mode='lines',
        name='Cascade Level (k)',
        line=dict(color=COLORS['dtk'], width=2)
    ))
    
    if tstar > 0:
        fig.add_vline(x=tstar, line_dash="dash", line_color=COLORS['tstar'],
                      annotation_text=f"t*={tstar}", annotation_position="top right")

    fig.update_layout(
        title="Dyadic Tree Topology (Teorema 24v)",
        xaxis_title="Time",
        yaxis_title="Cascade Depth Level (k)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(
            range=[22, -1],  # Fixed range so chaos (20) is at bottom and top is 0
            gridcolor='rgba(128, 128, 128, 0.2)',
            zeroline=False
        ),
        xaxis=dict(
            gridcolor='rgba(128, 128, 128, 0.2)',
            zeroline=False
        ),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def plot_unimodal_return_map(results: Dict[str, Any]) -> go.Figure:
    """
    Renders the Reconstructed Unimodal Return Map (Cobweb Map)
    Plots tau_s(t) vs tau_s(t+1) with true cobweb geometry (staircasing to y=x) 
    and temporal color gradients to reveal the descent into chaos.
    """
    if "taus_global" not in results:
        return go.Figure()
        
    taus = np.asarray(results["taus_global"])
    tstar = results.get("t_star", 0)
    
    # We need t and t+1
    x_val = taus[:-1]
    y_val = taus[1:]
    time_indices = np.arange(len(x_val))
    
    # Filter NaNs
    valid = ~(np.isnan(x_val) | np.isnan(y_val))
    x_val = x_val[valid]
    y_val = y_val[valid]
    time_indices = time_indices[valid]
    
    fig = go.Figure()
    
    # 1. Reference line y = x (Fixed points line)
    fig.add_trace(go.Scatter(
        x=[-1.1, 1.1],
        y=[-1.1, 1.1],
        mode='lines',
        name='Attractor Diagonal (y = x)',
        line=dict(color='rgba(255, 255, 255, 0.4)', width=2, dash='dash'),
        hoverinfo='skip'
    ))
    
    # 2. True Cobweb Trajectory (Staircase)
    # To draw a cobweb: (x_t, y_t) -> (y_t, y_t) -> (y_t, y_{t+1})
    cobweb_x = []
    cobweb_y = []
    for i in range(len(x_val)-1):
        cobweb_x.extend([x_val[i], y_val[i], y_val[i]])
        cobweb_y.extend([y_val[i], y_val[i], y_val[i+1]])
        
    fig.add_trace(go.Scatter(
        x=cobweb_x,
        y=cobweb_y,
        mode='lines',
        name='Cobweb Trajectory',
        line=dict(color='rgba(56, 189, 248, 0.3)', width=1), # Faint sky blue
        hoverinfo='skip'
    ))
    
    # 3. The actual state points colored by time (Evolution into chaos)
    fig.add_trace(go.Scatter(
        x=x_val,
        y=y_val,
        mode='markers',
        name='System State τ(t)',
        marker=dict(
            size=7,
            color=time_indices,
            colorscale='Plasma',  # Plasma goes from deep blue (early) to bright yellow (late/chaos)
            showscale=True,
            colorbar=dict(title=dict(text="Time (t)", font=dict(color='white')), thickness=15, outlinewidth=0, tickfont=dict(color='white')),
            line=dict(color='black', width=0.5)
        ),
        text=[f"t={t}<br>τ(t)={x:.3f}<br>τ(t+1)={y:.3f}" for t, x, y in zip(time_indices, x_val, y_val)],
        hoverinfo='text'
    ))
    
    # 4. Highlight t* (Topological Reorganization Point)
    if tstar > 0 and tstar < len(x_val):
        fig.add_trace(go.Scatter(
            x=[x_val[tstar-1]], 
            y=[y_val[tstar-1]],
            mode='markers',
            name='Transition Point (t*)',
            marker=dict(
                symbol='star',
                size=18,
                color='#f97316', # Orange
                line=dict(color='white', width=1.5)
            ),
            text=[f"CRITICAL TRANSITION t*={tstar}"],
            hoverinfo='text'
        ))
    
    # 5. Highlight the absolute Chaos region (|tau| <= 0.5)
    fig.add_vrect(x0=-0.5, x1=0.5, fillcolor="#ef4444", opacity=0.08, line_width=0, annotation_text="Feigenbaum Accumulation Zone", annotation_position="top left", annotation_font=dict(color="#ef4444", size=10))
    fig.add_hrect(y0=-0.5, y1=0.5, fillcolor="#ef4444", opacity=0.08, line_width=0)
    
    fig.update_layout(
        title=dict(text="<b>Unimodal Cobweb Map</b><br><sup>Empirical proof of Feigenbaum Universality in Systemic Tau</sup>", font=dict(size=18)),
        xaxis_title="Systemic Tau τ_s(t)",
        yaxis_title="Systemic Tau τ_s(t+1)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(
            range=[-1.05, 1.05],
            gridcolor='rgba(255, 255, 255, 0.1)',
            zerolinecolor='rgba(255, 255, 255, 0.2)',
            tickfont=dict(color='rgba(255,255,255,0.7)'),
            title=dict(font=dict(color='white'))
        ),
        xaxis=dict(
            range=[-1.05, 1.05],
            gridcolor='rgba(255, 255, 255, 0.1)',
            zerolinecolor='rgba(255, 255, 255, 0.2)',
            tickfont=dict(color='rgba(255,255,255,0.7)'),
            title=dict(font=dict(color='white'))
        ),
        margin=dict(l=20, r=20, t=60, b=20),
        height=600,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(0,0,0,0.5)",
            font=dict(color="white")
        )
    )
    return fig

def plot_nested_recd_dashboard(results: Dict[str, Any]) -> go.Figure:
    """
    Renders the Nested RECD 3-panel dashboard (T_recd clock, Conjunctions, Fractions)
    """
    if "nested_recd_results" not in results or not results["nested_recd_results"]:
        return go.Figure()
        
    recd = results["nested_recd_results"]
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
    
    tstar = results.get("t_star", None)
    
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
            fig.add_vline(x=tstar, line_width=2, line_dash="dash", line_color=COLORS['tstar'], row=i, col=1)
    
    fig.update_layout(
        height=800,
        margin=dict(l=20, r=20, t=40, b=20),
        template='plotly_white',
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_yaxes(title_text="Cumulative", row=1, col=1)
    fig.update_yaxes(title_text="α_k * Φ_k", row=2, col=1)
    fig.update_yaxes(title_text="Fraction", row=3, col=1, range=[0, 1.05])
    fig.update_xaxes(title_text="Time Step", row=3, col=1)
    
    return fig
