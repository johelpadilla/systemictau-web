import numpy as np

# =============================================================================
# Shared professional palette (kept in sync with studio/viz.py for consistency)
# =============================================================================
LEGACY_SCALE_PALETTE = {
    "Local": "#2563EB",      # Strong saturated blue
    "Medium": "#059669",     # Teal / Emerald
    "Global": "#7C3AED",     # Deep violet
}
ACCENT_RECD_LEGACY = "#DC2626"
ACCENT_TSTAR_LEGACY = "#EA580C"
THRESHOLD_CHAOS = "#94A3B8"
THRESHOLD_STABLE = "#64748B"


def _check_matplotlib():
    """Helper to check if matplotlib is installed since it's an optional dependency."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        return plt, mpl
    except ImportError:
        raise ImportError(
            "The visualization module requires matplotlib. "
            "Install it via 'pip install systemictau[visualization]' or 'pip install matplotlib'."
        )

def plot_tau_evolution(taus_global, T_series=None, episodes=None, ax=None):
    """
    Plots the evolution of Systemic Tau and optionally the RECD T_series and Joint Episodes.
    Modernized with professional styling for consistency with studio reports.
    """
    plt, mpl = _check_matplotlib()
    
    # Apply lightweight professional rc
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "lines.linewidth": 2.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 4.8))
        
    t = np.arange(len(taus_global))
    color_tau = LEGACY_SCALE_PALETTE.get("Local", "#2563EB")
    ax.plot(t, taus_global, color=color_tau, label='Systemic Tau τ_s', linewidth=2.8, alpha=0.92)
    
    # Plot T_series on secondary y-axis if provided (RECD dashed warm)
    if T_series is not None:
        ax2 = ax.twinx()
        ax2.plot(t, T_series, color=ACCENT_RECD_LEGACY, label='RECD T', linewidth=2.6, linestyle='--')
        ax2.set_ylabel("RECD T (accumulated)")
        ax2.tick_params(axis="y", labelcolor=ACCENT_RECD_LEGACY)
        
    # Highlight joint episodes if provided
    if episodes is not None:
        for ep in episodes:
            ax.axvspan(ep['start'], ep['end'], color='#FEF3C7', alpha=0.35, 
                       label='Joint Episode' if ep == episodes[0] else "")
            
    # Subtle reference thresholds
    ax.axhline(y=0.41, color=THRESHOLD_CHAOS, linestyle="--", linewidth=1.0, alpha=0.85, label="Chaos (0.41)")
    ax.axhline(y=0.50, color=THRESHOLD_STABLE, linestyle="--", linewidth=1.0, alpha=0.85, label="Stable (0.50)")
    
    ax.set_xlabel("Time step (windowed)")
    ax.set_ylabel("Systemic Tau τ_s")
    ax.set_title("Systemic Tau Evolution")
    ax.legend(loc="upper left", frameon=True, framealpha=0.95)
    ax.set_facecolor("#F8FAFC")
    
    return ax

def plot_joint_episodes(episodes, M_series, A_series, ax=None):
    """
    Plots the Critical Mass (M) and Anti-synchronization (A) series and highlights joint episodes.
    Updated styling for visual consistency.
    """
    plt, _ = _check_matplotlib()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 4.5))
        
    t = np.arange(len(M_series))
    ax.plot(t, M_series, color="#7C3AED", label='Critical Mass (M)', linewidth=2.6, alpha=0.9)
    
    ax2 = ax.twinx()
    ax2.plot(t, A_series, color="#D97706", label='Anti-sync (A)', linewidth=2.6, alpha=0.85)
    
    # Highlight joint episodes if provided
    if episodes is not None:
        for ep in episodes:
            ax.axvspan(ep['start'], ep['end'], color='#FEF3C7', alpha=0.3)
            
    ax.set_xlabel("Time step")
    ax.set_ylabel("Critical Mass $M$")
    ax2.set_ylabel("Anti-sync $A$")
    ax.set_title("Layer 2: Relational Coherence & Joint Episodes")
    ax.set_facecolor("#F8FAFC")
    
    # Combine legends from both axes
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper right", frameon=True, framealpha=0.95)
    
    return ax

def plot_ontological_layers(hp_z, lam, tt, M_series, A_series, t_star=None):
    """
    Plots a multi-panel figure showing the progression of the three ontological layers.
    Updated with consistent professional styling and better t* visibility.
    """
    plt, mpl = _check_matplotlib()
    mpl.rcParams.update({"lines.linewidth": 2.5, "font.family": "DejaVu Sans"})
    
    fig, axs = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    t = np.arange(len(hp_z))
    
    # Layer 1: Local Intensification
    axs[0].plot(t, hp_z, label='Hyper-persistence (Z)', color='#2563EB', linewidth=2.5)
    axs[0].plot(t, lam, label='Laminarity (LAM)', color='#0EA5E9', linewidth=2.2)
    axs[0].plot(t, tt, label='Trapping Time (TT)', color='#059669', linewidth=2.2)
    axs[0].set_ylabel("Layer 1 Metrics")
    axs[0].legend(loc="upper right", frameon=True, framealpha=0.95)
    axs[0].set_title("Layer 1: Local Intensification")
    axs[0].set_facecolor("#F8FAFC")
    
    # Layer 2: Relational Coherence
    axs[1].plot(t, M_series, color='#7C3AED', label='Critical Mass (M)', linewidth=2.5)
    axs[1].plot(t, A_series, color='#D97706', label='Anti-sync (A)', linewidth=2.5)
    axs[1].set_ylabel("Layer 2 Metrics")
    axs[1].legend(loc="upper right", frameon=True, framealpha=0.95)
    axs[1].set_title("Layer 2: Relational Coherence")
    axs[1].set_facecolor("#F8FAFC")
    
    # Layer 3: Ontological Ascent
    axs[2].text(0.5, 0.55, 'Ontological Ascent Detection', 
                horizontalalignment='center', verticalalignment='center', transform=axs[2].transAxes, fontsize=11)
    if t_star is not None:
        for ax in axs:
            ax.axvline(x=t_star, color=ACCENT_TSTAR_LEGACY, linestyle='-', linewidth=2.2, label='Transition $t^*$')
        axs[2].text(0.5, 0.35, f'Transition confirmed at $t^* = {t_star}$', 
                    horizontalalignment='center', verticalalignment='center', transform=axs[2].transAxes, 
                    color=ACCENT_TSTAR_LEGACY, fontsize=11, fontweight='bold')
        axs[0].legend()
    
    axs[2].set_xlabel("Time step (windowed)")
    axs[2].set_yticks([])
    axs[2].set_title("Layer 3: Ontological Ascent")
    axs[2].set_facecolor("#F8FAFC")
    
    for ax in axs:
        ax.tick_params(colors='#1E293B')
    plt.tight_layout()
    return fig
