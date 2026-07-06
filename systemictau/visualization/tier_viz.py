"""
Two-Tier Visualization Module for Systemic Tau v4.6.0.
Tier 1: Ontological Overview
Tier 2: Layer Details
"""

import matplotlib.pyplot as plt
import numpy as np
from ..results import OntologicalAscentResult

def plot_ontological_overview(result: OntologicalAscentResult, save_path: str = None) -> plt.Figure:
    """
    Tier 1: High-level overview combining RECD, Joint Episodes, and Critical Transition t*.
    Includes embedded English narrative.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 1. Plot RECD (Systemic Time)
    T_series = result.T_series
    if len(T_series) > 0:
        ax.plot(T_series, color='navy', linewidth=2, label="Systemic Time (RECD)")
        
    # 2. Highlight Joint Episodes
    for ep in result.episodes:
        start = ep['start']
        end = ep['end']
        ax.axvspan(start, end, color='orange', alpha=0.3, label="Joint Episode (Lock-in)" if ep == result.episodes[0] else "")
        
    # 3. Mark t*
    if result.t_star is not None:
        ax.axvline(result.t_star, color='red', linestyle='--', linewidth=2, label=f"Critical Transition t*={result.t_star}")
        
    # 4. Narrative text box
    summary_text = result.summary(level="short", lang="en")
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
            
    ax.set_title("Tier 1: Ontological Overview", fontsize=14, fontweight='bold')
    ax.set_xlabel("Physical Time (t)")
    ax.set_ylabel("Systemic State")
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_layer_details(result: OntologicalAscentResult, save_path: str = None) -> plt.Figure:
    """
    Tier 2: Deep-dive plotting all 3 layers separately.
    Layer 1: Tau Series & Fractal Dimension
    Layer 2: Mass / Entropy & Episodes
    Layer 3: RECD Acceleration & Reorganization
    """
    fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    # Layer 1
    if result.taus_global is not None and len(result.taus_global) > 0:
        axs[0].plot(result.taus_global, color='purple', alpha=0.6, label="Global Systemic Tau")
    axs[0].set_title(f"Layer 1: Microscopic Entanglement (Fractal D = {result.fractal_D:.2f})")
    axs[0].set_ylabel("Tau Value")
    axs[0].legend(loc='upper right')
    axs[0].grid(True, alpha=0.3)
    
    # Layer 2
    for ep in result.episodes:
        start = ep['start']
        end = ep['end']
        axs[1].axvspan(start, end, color='orange', alpha=0.3)
    axs[1].set_title(f"Layer 2: Mesoscopic Crystallization ({len(result.episodes)} Joint Episodes)")
    # Normally we'd plot M_series here, but we don't store it by default to save memory. 
    # We can plot tau absolute mean or something proxy if M_series isn't saved.
    axs[1].plot(np.abs(result.taus_global), color='green', alpha=0.5, label="|Tau| Magnitude")
    axs[1].set_ylabel("Magnitude")
    axs[1].legend(loc='upper right')
    axs[1].grid(True, alpha=0.3)
    
    # Layer 3
    if len(result.T_series) > 0:
        axs[2].plot(result.T_series, color='navy', label="RECD")
    if result.t_star is not None:
        axs[2].axvline(result.t_star, color='red', linestyle='--', label=f"t* = {result.t_star}")
    axs[2].set_title("Layer 3: Macroscopic Transition")
    axs[2].set_xlabel("Physical Time (t)")
    axs[2].set_ylabel("RECD")
    axs[2].legend(loc='upper right')
    axs[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig
