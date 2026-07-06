"""
Systemic Tau Studio — Publication-Grade Visualization Layer

This module produces elegant, minimalist, high-clarity scientific figures
suitable for top-tier journals. Emphasis on:
- Consistent visual language across ontological scales
- Precise annotation (t*, regimes, RECD increments)
- Balanced typography, spacing, and color
- Direct export to PNG/SVG/PDF at high resolution
"""

from __future__ import annotations
import io
from typing import Dict, Any, Optional, Tuple, List

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.figure import Figure
    from matplotlib.gridspec import GridSpec
except ImportError as e:
    raise ImportError(
        "Systemic Tau Studio visualizations require matplotlib. "
        "Install with: pip install 'systemictau[visualization]' or pip install matplotlib"
    ) from e

# Silence common noisy matplotlib font messages (harmless, especially on macOS
# with many system fonts + bundled DejaVu in the .app).
import warnings
warnings.filterwarnings(
    "ignore",
    message="findfont: Failed to find font weight",
    category=UserWarning,
    module="matplotlib"
)
warnings.filterwarnings(
    "ignore",
    message="Failed to extract font properties",
    category=UserWarning,
)
# Also ignore the generic "findfont" INFO/DEBUG noise if it bubbles up as warnings
warnings.filterwarnings("ignore", message="findfont", category=UserWarning)
warnings.filterwarnings("ignore", message="findfont", category=DeprecationWarning)

# =============================================================================
# Professional Color Palette (elegant, print-safe, perceptually reasonable)
# =============================================================================
SCALE_PALETTE = {
    "Local": "#2563EB",      # Strong saturated blue
    "Medium": "#059669",     # Teal / Emerald
    "Global": "#7C3AED",     # Deep violet
    "Hypothetical-Global": "#8B5CF6",
    "Hypothetical-Medium": "#10B981",
}

ACCENT_TAU = "#1E88E5"
ACCENT_RECD = "#DC2626"
ACCENT_TSTAR = "#EA580C"   # Bright orange-red for visibility
GRID_COLOR = "#E2E8F0"
TEXT_COLOR = "#1E293B"
LIGHT_BG = "#F8FAFC"
LIGHT_BORDER = "#E2E8F0"

# =============================================================================
# Centralized line widths, alphas, and common styling (maintainability)
# =============================================================================
LINEWIDTH_MAIN = 2.8
LINEWIDTH_THICK = 2.9
LINEWIDTH_ACCENT = 2.2
LINEWIDTH_THRESHOLD = 1.0
LINEWIDTH_TSTAR = 2.3
ALPHA_MAIN = 0.92
ALPHA_THRESHOLD = 0.85
MARKERSIZE_STANDARD = 9
DEFAULT_FIGSIZE = (11.0, 5.0)
WIDE_FIGSIZE = (12.0, 5.0)
DETAIL_FIGSIZE = (12.4, 9.2)
DETAIL_HEIGHT_RATIOS = (1.0, 1.42)   # deliberate extra height on bottom to elevate Δt_k to first-class status
LINEWIDTH_DTK = 2.35                 # strong, confident spike weight for the increments panel
DTK_COLOR = "#0369A1"
DTK_ACCENT = "#0284C8"
DTK_ALPHA = 0.96
DTK_MARKERSIZE = 36
TSTAR_BAND_COLOR = "#FEF3C7"
TSTAR_BAND_ALPHA = 0.26              # prominent but elegant unifying band
HSPACE_DUAL = 0.018                  # extremely tight — the two panels must read as one object
DUAL_LEFT_MARGIN = 0.065
DUAL_RIGHT_MARGIN = 0.96
DUAL_TOP_MARGIN = 0.915
DUAL_BOTTOM_MARGIN = 0.055


def add_tstar_marker(ax, tstar: int, y_pos: float = 0.82, label_y: float = 0.86, offset: int = 3):
    """Centralized helper for prominent, elegant t* (line + downward triangle + label).
    Used consistently in dual plots and main overview figures.
    """
    ax.axvline(x=tstar, color=ACCENT_TSTAR, linestyle="-", linewidth=LINEWIDTH_TSTAR, alpha=0.94, zorder=4)
    ax.scatter([tstar], [y_pos], marker="v", s=68, color=ACCENT_TSTAR, zorder=7,
               edgecolor="white", linewidths=0.6)
    ax.text(tstar + offset, label_y, f"t* = {tstar}", color=ACCENT_TSTAR, fontsize=9.5,
            fontweight="bold", va="bottom")


def get_scale_palette() -> Dict[str, str]:
    """Return the canonical scale color map."""
    return SCALE_PALETTE.copy()


def apply_publication_style(fig: Figure, axes: Optional[List[plt.Axes]] = None) -> None:
    """
    Apply a clean, journal-grade style to a figure.
    Call after plotting content, before tight_layout / save.
    """
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "font.weight": "normal",
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": GRID_COLOR,
        "grid.linewidth": 0.4,
        "lines.linewidth": LINEWIDTH_MAIN,
        "patch.linewidth": 0.8,
    })

    fig.patch.set_facecolor("white")
    if axes is None:
        axes = fig.axes
    for ax in axes:
        ax.set_facecolor(LIGHT_BG)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color(TEXT_COLOR)
            ax.spines[spine].set_linewidth(0.9)
        ax.tick_params(colors=TEXT_COLOR, length=3.5, width=0.7)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color(TEXT_COLOR)
        ax.grid(False)


def _ensure_axes(ax: Optional[plt.Axes], figsize: Tuple[float, float] = None) -> Tuple[Figure, plt.Axes]:
    if figsize is None:
        figsize = DEFAULT_FIGSIZE
    if ax is not None:
        return ax.figure, ax
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def plot_multi_scale_tau_trajectories(
    scale_results: Dict[str, Dict[str, Any]],
    *,
    ax: Optional[plt.Axes] = None,
    show_tstar: bool = True,
    title: str = "Systemic Tau τ_s Evolution Across Ontological Levels",
    max_points: int = 800,
) -> Figure:
    """
    Redesigned main overview plot per professional recommendations:
    - Wider aspect ratio
    - Thicker lines + strong professional palette
    - Prominent, well-labeled t* verticals
    - Clean summary box instead of cluttered bottom text
    - Light background + better typography/spacing
    - Subtitle explaining ontological ascent
    """
    fig, ax = _ensure_axes(ax, figsize=WIDE_FIGSIZE)

    palette = get_scale_palette()
    has_any = False
    tstar_data = {}

    for scale in ("Local", "Medium", "Global"):
        if scale not in scale_results:
            continue
        res = scale_results[scale]
        taus = np.asarray(res.get("taus_global", []))
        if taus.size == 0 or np.all(np.isnan(taus)):
            continue

        t = np.arange(len(taus))
        color = palette.get(scale, "#333333")

        valid = ~np.isnan(taus)
        t_plot = t[valid]
        taus_plot = taus[valid]

        n = len(taus_plot)
        if n > max_points:
            stride = max(1, n // max_points)
            t_plot = t_plot[::stride]
            taus_plot = taus_plot[::stride]

        ax.plot(
            t_plot, taus_plot,
            color=color,
            linewidth=LINEWIDTH_THICK,
            label=f"{scale}",
            alpha=ALPHA_MAIN,
            rasterized=(n > 1500),
        )

        if show_tstar and "t_star" in res and res["t_star"] is not None:
            try:
                tstar = int(res["t_star"])
                if 0 <= tstar < len(taus):
                    tstar_data[scale] = tstar
                    add_tstar_marker(ax, tstar)
            except Exception:
                pass

        has_any = True

    if not has_any:
        ax.text(0.5, 0.5, "No valid τ_s series", ha="center", va="center", transform=ax.transAxes, color="#666")
    else:
        # Lighter threshold lines
        ax.axhline(y=0.41, color="#94A3B8", linestyle="--", linewidth=LINEWIDTH_THRESHOLD, alpha=ALPHA_THRESHOLD, label="Chaos (0.41)")
        ax.axhline(y=0.50, color="#64748B", linestyle="--", linewidth=LINEWIDTH_THRESHOLD, alpha=ALPHA_THRESHOLD, label="Stable (0.50)")

    ax.set_xlabel("Time step (windowed)", fontsize=11)
    ax.set_ylabel("Systemic Tau  τ_s", fontsize=11)
    ax.set_title(title, pad=6, fontsize=13, fontweight="bold")

    # Subtitle
    ax.text(0.5, 0.98, "Local → Medium → Global aggregation and critical transitions",
            transform=ax.transAxes, ha="center", fontsize=9, style="italic", color="#475569")

    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor=LIGHT_BORDER, fontsize=9)
    ax.set_ylim(-0.68, 1.08)
    ax.set_facecolor(LIGHT_BG)

    # Clean summary box for transitions (replaces cluttered bottom text)
    if tstar_data:
        box_lines = ["Critical Transitions (t*)", "──────────────────────"]
        for sc in ("Local", "Medium", "Global"):
            if sc in tstar_data:
                box_lines.append(f"{sc:8}: {tstar_data[sc]}")
        box_text = "\n".join(box_lines)
        ax.text(0.98, 0.18, box_text, transform=ax.transAxes, fontsize=9,
                family="monospace", va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="#CBD5E1", alpha=0.97))

    apply_publication_style(fig)
    fig.tight_layout(pad=0.55)
    return fig


def plot_ontological_decay(
    scale_metrics: Dict[str, Dict[str, float]],
    *,
    ax: Optional[plt.Axes] = None,
    metrics: Optional[List[str]] = None,
    title: str = "Systemic Metrics Across Ontological Levels",
    perspective: Optional[str] = None,
) -> Figure:
    """
    Redesigned Ontological Decay / Emergence plot (follows detailed feedback):
    - Thicker cleaner lines (2.8–3.0 pt)
    - Professional palette + light background
    - Elegant percentage change cards / box instead of cluttered inline labels
    - Better typography, spacing, markers
    """
    fig, ax = _ensure_axes(ax, figsize=WIDE_FIGSIZE)

    scales = ["Local", "Medium", "Global"]
    present_scales = [s for s in scales if s in scale_metrics]

    if not present_scales:
        ax.text(0.5, 0.5, "No scale metrics", ha="center", transform=ax.transAxes)
        apply_publication_style(fig)
        return fig

    if metrics is None:
        metrics = ["mean_tau", "recd_T_final", "coherence"]

    x = np.arange(len(present_scales))

    # Stronger recommended palette
    metric_style = {
        "mean_tau":    {"color": "#2563EB", "ls": "-",  "marker": "o", "lw": LINEWIDTH_THICK},
        "recd_T_final":{"color": "#0F766E", "ls": "-",  "marker": "s", "lw": LINEWIDTH_THICK},
        "coherence":   {"color": "#D97706", "ls": "--", "marker": "^", "lw": LINEWIDTH_MAIN},
    }

    pct_changes = {}

    for metric in metrics:
        vals = [scale_metrics[s].get(metric, np.nan) for s in present_scales]
        vals = np.asarray(vals, dtype=float)

        style = metric_style.get(metric, {"color": "#475569", "ls": "-", "marker": "o", "lw": 2.5})

        ax.plot(x, vals, color=style["color"], linestyle=style["ls"],
                marker=style["marker"], markersize=MARKERSIZE_STANDARD, linewidth=style["lw"],
                label=metric.replace("_", " ").title(), zorder=4, alpha=ALPHA_MAIN)

        if len(present_scales) >= 2 and present_scales[0] == "Local" and present_scales[-1] == "Global":
            v0, vg = vals[0], vals[-1]
            if np.isfinite(v0) and np.isfinite(vg) and abs(v0) > 1e-9:
                pct = (vg - v0) / (abs(v0) + 1e-9) * 100
                pct_changes[metric] = pct

    ax.set_xticks(x)
    ax.set_xticklabels(present_scales, fontsize=11)
    ax.set_xlabel("Ontological Level", fontsize=11)
    ax.set_ylabel("Metric Value", fontsize=11)

    full_title = title
    if perspective:
        full_title = f"{title} — {perspective.title()} Perspective"
    ax.set_title(full_title, pad=6, fontsize=13, fontweight="bold")

    # Subtitle
    ax.text(0.5, 0.98, "Evolution of systemic stability, temporal discretization, and coherence under ontological ascent",
            transform=ax.transAxes, ha="center", fontsize=9, style="italic", color="#475569")

    ax.legend(loc="upper left", frameon=True, fontsize=9, framealpha=0.96, edgecolor=LIGHT_BORDER)
    ax.grid(True, axis="y", linestyle=":", alpha=0.35, zorder=0)
    ax.set_facecolor(LIGHT_BG)

    # Elegant summary box for % changes (instead of cluttered annotations)
    if pct_changes:
        box_lines = ["% Change Local → Global"]
        for m, p in pct_changes.items():
            label = m.replace("_", " ").title()
            sign = "+" if p >= 0 else ""
            box_lines.append(f"{label:12}: {sign}{p:.1f}%")
        ax.text(0.98, 0.95, "\n".join(box_lines), transform=ax.transAxes,
                fontsize=9.5, va="top", ha="right", family="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#CBD5E1", alpha=0.97))

    apply_publication_style(fig)
    fig.tight_layout(pad=0.55)
    return fig


def plot_recd_discretization(
    scale_results: Dict[str, Dict[str, Any]],
    *,
    ax: Optional[plt.Axes] = None,
    title: str = "Accumulated Discrete Time (RECD) Across Ontological Levels",
) -> Figure:
    """
    Redesigned RECD accumulation plot:
    - High precision final values (4 decimals)
    - Thicker lines + final point markers with labels
    - Better legend with actual values
    - Light background + improved hierarchy
    """
    fig, ax = _ensure_axes(ax)  # uses DEFAULT_FIGSIZE

    palette = get_scale_palette()
    plotted = False
    all_finals = []
    series_data = {}  # scale -> (t, T, final)

    for scale in ("Local", "Medium", "Global"):
        if scale not in scale_results:
            continue
        res = scale_results[scale]
        T = res.get("accum_T")
        if T is None:
            T = res.get("T_series")
        if T is None:
            continue
        T = np.asarray(T)
        if T.size == 0 or np.all(np.isnan(T)):
            continue

        t = np.arange(len(T))
        color = palette.get(scale, "#333")
        final_val = float(T[-1])
        series_data[scale] = (t, T, final_val)
        all_finals.append(final_val)
        plotted = True

    scale_factor = 1.0
    y_label = "Accumulated Discrete Time  T  (RECD)"
    if plotted:
        max_final = max(all_finals) if all_finals else 0.0
        if max_final > 0 and max_final < 0.05:
            scale_factor = 1000.0
            y_label = "Accumulated Discrete Time  T  (RECD × 10³)"

    for scale, (t, T, final_val) in series_data.items():
        color = palette.get(scale, "#333")
        Ty = T * scale_factor if scale_factor != 1.0 else T
        fy = final_val * scale_factor
        ax.plot(t, Ty, color=color, linewidth=LINEWIDTH_MAIN, label=f"{scale}  (final = {final_val:.4f})", alpha=ALPHA_MAIN)
        ax.scatter([len(T)-1], [fy], s=55, color=color, zorder=5, edgecolor="white", linewidths=0.7)

    if not plotted:
        ax.text(0.5, 0.5, "No RECD accumulation data", ha="center", transform=ax.transAxes, color="#666")
    else:
        ax.set_ylabel(y_label)
        ax.set_xlabel("Time step (windowed)")

    ax.set_title(title, pad=6, fontsize=12, fontweight="bold")
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor=LIGHT_BORDER, fontsize=9)
    ax.set_facecolor(LIGHT_BG)

    apply_publication_style(fig)
    fig.tight_layout(pad=0.55)
    return fig


def _draw_tstar_band_and_markers(ax_tau, ax_dtk, ts_val: int, t: np.ndarray, dtk: Optional[np.ndarray]):
    """
    The primary visual glue that makes the dual plot feel like ONE figure.
    Draws a clean, elegant vertical band + strong anchor line that crosses both panels
    without interruption. Markers and labels are placed with high precision and
    consistent styling on both panels.
    """
    if ts_val is None or len(t) == 0:
        return

    n = len(t)
    # Slightly wider elegant band — the central unifying element
    band_w = max(3, n // 65)
    band_col = TSTAR_BAND_COLOR

    # Layered band for soft professional edge (outer soft + core)
    for ax in (ax_tau, ax_dtk):
        ax.axvspan(ts_val - band_w * 1.15, ts_val + band_w * 1.15,
                   color=band_col, alpha=TSTAR_BAND_ALPHA * 0.55, zorder=0)
        ax.axvspan(ts_val - band_w, ts_val + band_w,
                   color=band_col, alpha=TSTAR_BAND_ALPHA, zorder=0)

    # Very strong continuous vertical anchor line (the "spine" of the figure at t*)
    for ax in (ax_tau, ax_dtk):
        ax.axvline(x=ts_val, color=ACCENT_TSTAR, linestyle="-",
                   linewidth=LINEWIDTH_TSTAR * 1.05, alpha=0.98, zorder=6)

    # --- Top panel (τ_s) marker ---
    # Place high but safely inside the data range
    try:
        y0, y1 = ax_tau.get_ylim()
        tri_y = y0 + (y1 - y0) * 0.87
        txt_y = y0 + (y1 - y0) * 0.945
    except Exception:
        tri_y, txt_y = 0.88, 0.96

    ax_tau.scatter([ts_val], [tri_y], marker="v", s=58, color=ACCENT_TSTAR,
                   zorder=9, edgecolor="white", linewidths=0.6)
    ax_tau.annotate(f"t* = {ts_val}",
                    xy=(ts_val, tri_y),
                    xytext=(ts_val + max(3, n // 48), txt_y),
                    fontsize=9.7, fontweight="bold", color=ACCENT_TSTAR,
                    va="bottom", ha="left",
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                              edgecolor="none", alpha=0.0))

    # --- Bottom panel (Δt_k) marker ---
    if dtk is not None and len(dtk) > ts_val:
        spike = float(dtk[ts_val])
        dmax = float(np.nanmax(dtk)) if len(dtk) > 0 else 1.0
        label_y = spike + max(0.10 * dmax, 0.018)

        ax_dtk.scatter([ts_val], [spike], marker="v", s=58, color=ACCENT_TSTAR,
                       zorder=10, edgecolor="white", linewidths=0.6)
        ax_dtk.annotate(f"t* = {ts_val}",
                        xy=(ts_val, spike),
                        xytext=(ts_val + max(3, n // 48), label_y),
                        fontsize=9.7, fontweight="bold", color=ACCENT_TSTAR,
                        va="bottom", ha="left",
                        bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                                  edgecolor="none", alpha=0.0))


def plot_scale_detail(
    scale: str,
    result: Dict[str, Any],
    *,
    fig: Optional[Figure] = None,
    title_prefix: str = "",
) -> Figure:
    """
    Completely rewritten dual-plot generator.

    Philosophy:
      - The figure is a SINGLE integrated object, not two separate charts.
      - GridSpec + aggressive minimal hspace + removed internal spines.
      - A prominent, elegant vertical t* band + anchor is the visual "bridge"
        that makes top (τ_s + RECD) and bottom (Δt_k) belong together.
      - Bottom panel (Δt_k) is deliberately given strong visual mass and polish
        so it feels first-class.

    Structure is fresh, explicit, and easy to maintain.
    All previous functionality preserved (annotations, scales, colors, RECD final value, etc.).
    """
    if fig is None:
        fig = plt.figure(figsize=DETAIL_FIGSIZE)

    # Scale color (same logic as before for compatibility)
    palette = get_scale_palette()
    if scale == "Hypothetical-Global":
        display = "Hypothetical Global"
        color = palette.get("Hypothetical-Global", "#8B5CF6")
    elif scale == "Hypothetical-Medium":
        display = "Hypothetical Medium"
        color = palette.get("Hypothetical-Medium", "#10B981")
    else:
        display = scale
        color = palette.get(scale, "#2563EB")

    # ========== FRESH GRIDSpec LAYOUT ==========
    gs = GridSpec(
        2, 1,
        figure=fig,
        height_ratios=DETAIL_HEIGHT_RATIOS,
        hspace=HSPACE_DUAL
    )
    ax_tau = fig.add_subplot(gs[0])          # top: τ_s + RECD twin
    ax_dtk = fig.add_subplot(gs[1], sharex=ax_tau)  # bottom: Δt_k

    # Make it read as one object: kill the artificial boundary between panels
    ax_tau.spines["bottom"].set_visible(False)
    ax_dtk.spines["top"].set_visible(False)
    ax_tau.tick_params(labelbottom=False, bottom=False)
    ax_dtk.spines["right"].set_visible(False)   # keep right clean on bottom

    # ========== DATA ==========
    taus = np.asarray(result.get("taus_global", []))
    Tser = result.get("accum_T")
    if Tser is None:
        Tser = result.get("T_series")
    tstar = result.get("t_star")
    dtk_raw = result.get("dtk")

    t = np.arange(len(taus)) if len(taus) > 0 else np.array([])
    ts_val = int(tstar) if tstar is not None and not (isinstance(tstar, float) and np.isnan(tstar)) else None

    # ========== TOP: τ_s + RECD twin ==========
    valid = ~np.isnan(taus) if len(taus) > 0 else np.zeros(0, dtype=bool)

    if valid.any():
        ax_tau.plot(t[valid], taus[valid], color=color,
                    linewidth=LINEWIDTH_MAIN, zorder=5, label="τ_s")

    # Subtle, professional thresholds (match overview style)
    ax_tau.axhline(0.41, color="#94A3B8", ls="--", lw=LINEWIDTH_THRESHOLD, alpha=0.78, zorder=1)
    ax_tau.axhline(0.50, color="#64748B", ls="--", lw=LINEWIDTH_THRESHOLD, alpha=0.78, zorder=1)

    final_recd = None
    if Tser is not None and len(Tser) > 0:
        Tser = np.asarray(Tser)
        ax_recd = ax_tau.twinx()
        ax_recd.plot(t[:len(Tser)], Tser,
                     color=ACCENT_RECD, linewidth=LINEWIDTH_MAIN,
                     linestyle="--", zorder=4, alpha=0.97, label="RECD")

        last_idx = min(len(t) - 1, len(Tser) - 1)
        last_t = t[last_idx]
        final_recd = float(Tser[last_idx])

        # Strong, clean end marker
        ax_recd.scatter([last_t], [final_recd], s=56, color=ACCENT_RECD,
                        zorder=8, edgecolor="white", linewidths=0.75, marker="o")

        ax_recd.set_ylabel("RECD  T  (accumulated)", color=ACCENT_RECD, fontsize=10)
        ax_recd.tick_params(axis="y", labelcolor=ACCENT_RECD, labelsize=9)

        # Elegant final value box (right side, high contrast, matches report boxes)
        ax_recd.text(0.982, 0.94, f"final = {final_recd:.4f}",
                     transform=ax_recd.transAxes,
                     fontsize=9.5, ha="right", va="top",
                     color=ACCENT_RECD, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                               edgecolor=ACCENT_RECD, alpha=0.97, linewidth=1.25))

    ax_tau.set_ylabel("Systemic Tau  τ_s", fontsize=11)
    ax_tau.legend(loc="upper left", fontsize=9, frameon=True,
                  framealpha=0.96, edgecolor=LIGHT_BORDER)
    ax_tau.set_facecolor(LIGHT_BG)

    # Generous headroom on top for t* triangle
    if valid.any():
        tmin = float(np.nanmin(taus[valid]))
        tmax = float(np.nanmax(taus[valid]))
        headroom = max(0.10, (tmax - tmin) * 0.18)
        ax_tau.set_ylim(tmin - 0.08, max(tmax + headroom * 0.5, 0.96))

    # ========== BOTTOM: Δt_k — now a first-class citizen ==========
    dtk = None
    if dtk_raw is not None and len(dtk_raw) > 0:
        dtk = np.asarray(dtk_raw)
        t_dtk = t[:len(dtk)]

        # Powerful, clean spikes
        ax_dtk.vlines(t_dtk, 0, dtk, colors=DTK_COLOR,
                      linewidth=LINEWIDTH_DTK, alpha=DTK_ALPHA, zorder=3)

        # Prominent peak markers (double layer for polish)
        ax_dtk.scatter(t_dtk, dtk, s=DTK_MARKERSIZE, color=DTK_ACCENT,
                       zorder=7, edgecolors="white", linewidths=0.5, marker="o")
        ax_dtk.scatter(t_dtk, dtk, s=DTK_MARKERSIZE * 0.55, color="white",
                       zorder=6, alpha=0.9)   # subtle halo

        # Rich area fill for visual weight without clutter
        ax_dtk.fill_between(t_dtk, 0, dtk, color=DTK_ACCENT, alpha=0.18, zorder=1)

        # Strong zero baseline
        ax_dtk.axhline(0.0, color="#9CA3AF", linewidth=0.9, linestyle="-",
                       zorder=0, alpha=0.85)

        # Good breathing room
        dmax = float(np.nanmax(dtk)) if len(dtk) > 0 else 1.0
        ax_dtk.set_ylim(0, dmax * 1.32 if dmax > 0 else 1.0)
    else:
        ax_dtk.text(0.5, 0.5, "No RECD increment data",
                    ha="center", transform=ax_dtk.transAxes, color="#666")

    ax_dtk.set_xlabel("Time step (windowed)", fontsize=11)
    ax_dtk.set_ylabel("RECD Increments  Δt_k", fontsize=11)
    ax_dtk.set_facecolor(LIGHT_BG)
    ax_dtk.tick_params(labelsize=9)

    # ========== THE UNIFYING t* ELEMENT (called after both panels exist) ==========
    if ts_val is not None:
        _draw_tstar_band_and_markers(ax_tau, ax_dtk, ts_val, t, dtk)

    # ========== TITLE — single header for the whole integrated figure ==========
    fig.suptitle(f"{title_prefix}{display}  —  τ_s  +  Δt_k Increments",
                 fontsize=13, fontweight="bold", y=0.975, color=TEXT_COLOR)

    # ========== FINAL STYLING (match the highest quality overview plots) ==========
    apply_publication_style(fig, axes=[ax_tau, ax_dtk])

    # Extremely tight, professional framing
    fig.subplots_adjust(
        left=DUAL_LEFT_MARGIN,
        right=DUAL_RIGHT_MARGIN,
        top=DUAL_TOP_MARGIN,
        bottom=DUAL_BOTTOM_MARGIN,
        hspace=HSPACE_DUAL
    )

    # Ensure identical x range (defensive)
    if len(t) > 1:
        ax_tau.set_xlim(-0.5, len(t) - 0.5)

    return fig


def export_figure(
    fig: Figure,
    dest: Optional[str] = None,
    fmt: str = "pdf",
    dpi: int = 300,
    **save_kwargs,
) -> bytes | str:
    """
    Export a figure to high-quality publication format.
    If dest is None, returns bytes (useful for Streamlit download).
    fmt in {'pdf', 'svg', 'png'}.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, facecolor="white", edgecolor="none",
                bbox_inches="tight", pad_inches=0.1, **save_kwargs)
    buf.seek(0)
    data = buf.getvalue()

    if dest:
        with open(dest, "wb") as f:
            f.write(data)
        return dest
    return data


def create_report_pdf(
    arg1: Any,
    arg2: Any = None,
    *,
    raw_shape: Optional[tuple] = None,
    window_size: int = 13,
    filename: Optional[str] = None,
    title: str = "Systemic Tau Studio — Multi-Perspective Ontological Report",
    insight_text: Optional[str] = None,
    sensitivity_report: Any = None,   # NEW: optional sensitivity results
    surrogate_result: Any = None,     # NEW: optional IAAFT surrogate results
    **kwargs,
) -> bytes | str:
    """
    Supports both:
      - New: create_report_pdf(analyses_dict, raw_shape=..., window_size=...)
      - Old/compat: create_report_pdf(scale_results, scale_metrics, insight_text=...)
    Professional multi-perspective report.
    - Proper cover with metadata
    - Comparative summary table
    - Per-perspective sections with improved plots + clean text
    - Highlights differences between perspectives
    """
    # --- Compatibility dispatcher ---
    if isinstance(arg1, dict) and arg1 and "Local" in arg1 and isinstance(list(arg1.values())[0], dict) and "taus_global" in str(list(arg1.values())[0]):
        # Old call style: first arg is scale_results for a single perspective
        analyses = {"Report": {"scale_results": arg1, "scale_metrics": (arg2 or {}), "insight_text": insight_text or kwargs.get("insight_text")}}
    elif isinstance(arg1, dict) and len(arg1) > 0 and all(isinstance(v, dict) for v in arg1.values()):
        analyses = arg1
    else:
        analyses = {"Report": {"scale_results": arg1 or {}, "scale_metrics": arg2 or {}, "insight_text": insight_text}}

    buf = io.BytesIO()
    perspectives = list(analyses.keys())

    with PdfPages(buf) as pdf:
        # ========== COVER / SUMMARY PAGE ==========
        cover = plt.figure(figsize=(8.5, 11))
        ax = cover.add_subplot(111)
        ax.axis("off")

        ax.text(0.5, 0.93, "SYSTEMIC TAU STUDIO", ha="center", fontsize=19, fontweight="bold")
        ax.text(0.5, 0.87, "Multi-Perspective Ontological Analysis Report", ha="center", fontsize=14, fontweight="bold")

        # Metadata box
        meta_lines = []
        if raw_shape:
            meta_lines.append(f"Dataset: {raw_shape[0]} rows × {raw_shape[1]} columns (raw)")
        meta_lines.append(f"Window size (w): {window_size}")
        meta_lines.append(f"Perspectives analyzed: {', '.join(perspectives)}")

        for i, (p, an) in enumerate(analyses.items()):
            sm = an.get("scale_metrics", {})
            cfg = an.get("config", {})
            n_local = sm.get("Local", {}).get("n_modules", "?")
            n_global = sm.get("Global", {}).get("n_modules", "?")
            meta_lines.append(f"{p}: Local modules={n_local}, Global macro-clusters={n_global}")

            # Clustering documentation (Priority from last session)
            cl = cfg.get("clustering_method", {})
            gg = cfg.get("global_groups", {})
            mg = cfg.get("medium_groups", {})
            if cl or gg:
                med_c = cl.get("medium", "auto")
                glo_c = cl.get("global", "auto")
                meta_lines.append(f"  Clustering → Medium: {med_c} | Global: {glo_c}")
                if gg:
                    meta_lines.append(f"  Global groups: {len(gg)} macro-clusters")
                if mg:
                    meta_lines.append(f"  Medium groups: {len(mg)} clusters")

        ax.text(0.08, 0.76, "\n".join(meta_lines), fontsize=9.5, va="top", family="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5", edgecolor="#E0E0E0"))

        # Key findings (simple)
        ax.text(0.08, 0.52, "KEY CONTRASTS BETWEEN PERSPECTIVES", fontsize=11, fontweight="bold")
        findings = []
        if len(perspectives) >= 2:
            findings.append("• Mean Systemic Tau and RECD behavior can differ dramatically")
            findings.append("  depending on whether modules are defined as locations (spatial)")
            findings.append("  or as measured variables (multivariate).")
            findings.append("• Check the comparative table and per-perspective sections below.")
        else:
            findings.append("Single perspective analysis.")

        ax.text(0.08, 0.46, "\n".join(findings), fontsize=9, va="top")

        ax.text(0.5, 0.08, "Principle of Ontological Ascent — Local → Medium → Global",
                ha="center", fontsize=9, style="italic", color="#555")

        pdf.savefig(cover, bbox_inches="tight")
        plt.close(cover)

        # ========== COMPARATIVE TABLE PAGE ==========
        comp_fig = plt.figure(figsize=(9.8, 5.2))
        ax = comp_fig.add_subplot(111)
        ax.axis("off")
        ax.set_title("Comparative Metrics Across Perspectives", fontsize=14, fontweight="bold", pad=6)

        # Build data for proper table
        headers = ["Perspective", "Mean τ (Local)", "RECD Final (L→G)", "Coherence (L)", "t* (L / M / G)"]
        cell_data = []
        for p, an in analyses.items():
            sm = an.get("scale_metrics", {})
            sr = an.get("scale_results", {})
            l = sm.get("Local", {})
            g = sm.get("Global", {})
            tau_l = l.get("mean_tau", float("nan"))
            recd_l = l.get("recd_T_final", float("nan"))
            recd_g = g.get("recd_T_final", float("nan"))
            coh = l.get("coherence", float("nan"))
            ts = []
            for sc in ("Local", "Medium", "Global"):
                t = sr.get(sc, {}).get("t_star")
                ts.append(str(int(t)) if t is not None and not (isinstance(t, float) and np.isnan(t)) else "—")
            cell_data.append([
                str(p),
                f"{tau_l:.3f}",
                f"{recd_l:.4f} → {recd_g:.4f}",
                f"{coh:.3f}",
                " / ".join(ts)
            ])

        if cell_data:
            tbl = ax.table(cellText=cell_data, colLabels=headers, loc="center",
                           cellLoc="center", colColours=["#E2E8F0"]*len(headers))
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(9)
            tbl.scale(1.1, 1.65)
            for (row, col), cell in tbl.get_celld().items():
                if row == 0:
                    cell.set_text_props(fontweight="bold")
                cell.set_edgecolor("#CBD5E1")
        else:
            ax.text(0.5, 0.5, "No data for comparison table", ha="center", transform=ax.transAxes)

        # Note under table
        ax.text(0.5, 0.12, "RECD values use 4-decimal precision to show actual magnitude of discrete time accumulation.",
                ha="center", fontsize=8, style="italic", color="#475569", transform=ax.transAxes)
        pdf.savefig(comp_fig, bbox_inches="tight")
        plt.close(comp_fig)

        # ========== CROSS-PERSPECTIVE INSIGHTS (new) ==========
        if len(perspectives) >= 2:
            cross = plt.figure(figsize=(8.5, 7))
            cax = cross.add_subplot(111)
            cax.axis("off")
            cax.text(0.5, 0.96, "Cross-Perspective Insights", ha="center", fontsize=14, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.25", facecolor="#F0F4F8", edgecolor="#CBD5E1", alpha=0.9))

            insights = []
            try:
                p0, p1 = perspectives[0], perspectives[1]
                sm0 = analyses[p0].get("scale_metrics", {})
                sm1 = analyses[p1].get("scale_metrics", {})
                l0, g0 = sm0.get("Local", {}), sm0.get("Global", {})
                l1, g1 = sm1.get("Local", {}), sm1.get("Global", {})

                tau_diff = abs(l0.get("mean_tau", 0) - l1.get("mean_tau", 0))
                recd0 = (l0.get("recd_T_final", 0), g0.get("recd_T_final", 0))
                recd1 = (l1.get("recd_T_final", 0), g1.get("recd_T_final", 0))

                insights.append(f"• Strong contrast detected between '{p0}' and '{p1}' perspectives.")
                insights.append(f"  Mean τ difference at Local scale: {tau_diff:.4f}")
                insights.append(f"  RECD accumulation: {p0} {recd0[0]:.4f}→{recd0[1]:.4f} vs {p1} {recd1[0]:.4f}→{recd1[1]:.4f}")
                insights.append("")
                insights.append("• Spatial (locations as modules) vs Multivariate (variables as modules)")
                insights.append("  often produce different emergence patterns and t* timings.")
                insights.append("  Use both views together rather than choosing one 'correct' scale.")
            except Exception:
                insights.append("Multiple perspectives analyzed. Compare the per-perspective sections below for differences in τ, RECD growth, and t*.")

            cax.text(0.08, 0.88, "\n".join(insights), fontsize=9.5, va="top", family="DejaVu Sans",
                     transform=cax.transAxes)
            pdf.savefig(cross, bbox_inches="tight")
            plt.close(cross)

        # ========== PER-PERSPECTIVE SECTIONS ==========
        for p, an in analyses.items():
            sr = an.get("scale_results", {})
            sm = an.get("scale_metrics", {})
            cfg = an.get("config", {})
            insight = an.get("insight_text", None)

            # Perspective header page - clean professional header
            hdr = plt.figure(figsize=(8.5, 2.2))
            hax = hdr.add_subplot(111)
            hax.axis("off")
            hax.text(0.5, 0.72, f"Perspective: {p}", ha="center", fontsize=15, fontweight="bold")
            nmod = sm.get("Local", {}).get("n_modules", "?")
            hax.text(0.5, 0.42, f"Local modules: {nmod}   •   Window size: {window_size}", ha="center", fontsize=10)
            pdf.savefig(hdr, bbox_inches="tight")
            plt.close(hdr)

            # === Clustering configuration (from last session task) ===
            cfg = an.get("config", {})
            gg = cfg.get("global_groups", {})
            mg = cfg.get("medium_groups", {})
            cl = cfg.get("clustering_method", {})
            if gg or mg or cl:
                try:
                    from systemictau.desktop.scale_manager import OntologicalScaleManager
                    cluster_text = []
                    if "global" in cl or gg:
                        gmeth = cl.get("global", "manual" if gg else "auto")
                        cluster_text.append(OntologicalScaleManager.describe_clustering(gmeth, gg or {}))
                    if mg:
                        cluster_text.append("Medium clusters (manual):")
                        for name, vars_ in mg.items():
                            cluster_text.append(f"  • {name} ← [{', '.join(vars_)}]")
                    if cluster_text:
                        cl_fig = plt.figure(figsize=(8.5, 3.5))
                        clax = cl_fig.add_subplot(111)
                        clax.axis("off")
                        clax.text(0.5, 0.92, "Clustering Configuration", ha="center", fontsize=12, fontweight="bold")
                        clax.text(0.08, 0.82, "\n".join(cluster_text), fontsize=9, va="top", family="monospace",
                                  transform=clax.transAxes)
                        pdf.savefig(cl_fig, bbox_inches="tight")
                        plt.close(cl_fig)
                except Exception:
                    pass

            # Core plots
            for maker in [
                lambda: plot_multi_scale_tau_trajectories(sr),
                lambda: plot_ontological_decay(sm, perspective=p),
                lambda: plot_recd_discretization(sr),
            ]:
                f = maker()
                pdf.savefig(f, bbox_inches="tight")
                plt.close(f)

            # Detail plots (more compact)
            for scale in ("Local", "Medium", "Global"):
                if scale in sr:
                    fd = plot_scale_detail(scale, sr[scale])
                    pdf.savefig(fd, bbox_inches="tight")
                    plt.close(fd)

            # Clean interpretation
            if insight:
                txt_fig = plt.figure(figsize=(8.5, 7))
                tax = txt_fig.add_subplot(111)
                tax.axis("off")
                tax.text(0.05, 0.95, f"Interpretation — {p}", fontsize=12, fontweight="bold")

                # Strip HTML properly
                clean = (insight
                         .replace("<b>", "").replace("</b>", "")
                         .replace("<i>", "").replace("</i>", "")
                         .replace("<br>", "\n").replace("<br><br>", "\n\n")
                         .replace("<div>", "").replace("</div>", ""))
                tax.text(0.05, 0.88, clean, fontsize=8.2, va="top", family="DejaVu Sans",
                         wrap=True, transform=tax.transAxes)
                pdf.savefig(txt_fig, bbox_inches="tight")
                plt.close(txt_fig)

    # ========== SENSITIVITY ANALYSIS SECTION (optional) ==========
    if sensitivity_report is not None:
        try:
            sens_fig = plt.figure(figsize=(8.5, 11))
            sax = sens_fig.add_subplot(111)
            sax.axis("off")

            sax.text(0.5, 0.96, "Parameter Sensitivity & Robustness", ha="center", fontsize=14, fontweight="bold")
            sax.text(0.5, 0.92, "Stability of t*, τ_s and RECD across parameter variations", ha="center", fontsize=9)

            # Summary
            if getattr(sensitivity_report, "summary", None):
                sax.text(0.08, 0.86, sensitivity_report.summary, fontsize=8, va="top", family="monospace", wrap=True)

            # Stability numbers
            stab = getattr(sensitivity_report, "stability", {}) or {}
            y = 0.72
            sax.text(0.08, y, "Stability Summary", fontsize=11, fontweight="bold")
            y -= 0.04

            for scale in ["Local", "Medium", "Global"]:
                ts = stab.get(f"{scale}_t_star")
                if ts:
                    sax.text(0.08, y, f"{scale}: t* median={ts.get('median')}, std={ts.get('std')}, "
                                      f"stable(±3)={ts.get('stability_score_within_3',0)*100:.0f}%", fontsize=8, family="monospace")
                    y -= 0.03

            overall = stab.get("overall_t_star_stability")
            if overall is not None:
                sax.text(0.08, y-0.02, f"Overall t* stability: {overall:.0%}", fontsize=9, fontweight="bold")

            # Sample table of first few runs
            y = 0.45
            sax.text(0.08, y, "Sample runs (first 8):", fontsize=9, fontweight="bold")
            y -= 0.03
            lines = []
            for r in (getattr(sensitivity_report, "results", []) or [])[:8]:
                if r.success:
                    p = r.params
                    m = r.metrics.get("Global", {})
                    lines.append(f"w={p.get('window_size')} k={p.get('global_num_clusters')}: t*={m.get('t_star')}, τ={m.get('mean_tau'):.3f}")

            if lines:
                sax.text(0.08, y, "\n".join(lines), fontsize=7.5, family="monospace", va="top")

            pdf.savefig(sens_fig, bbox_inches="tight")
            plt.close(sens_fig)
        except Exception:
            pass  # never break the whole report because of sensitivity

    # ========== SURROGATE VALIDATION (IAAFT) ==========
    if surrogate_result is not None:
        try:
            surr_fig = plt.figure(figsize=(8.5, 11))
            sax = surr_fig.add_subplot(111)
            sax.axis("off")

            sax.text(0.5, 0.96, "IAAFT Surrogate Validation", ha="center", fontsize=16, fontweight="bold")
            sax.text(0.5, 0.92, "Testing the significance of the systemic topological collapse", ha="center", fontsize=10)

            # Use the text-based summary we added
            summary_lines = surrogate_result.summary().split('\n')
            
            y_pos = 0.85
            for line in summary_lines:
                if line.startswith("## ") or line.startswith("### "):
                    sax.text(0.08, y_pos, line.replace("#", "").strip(), fontsize=12, fontweight="bold")
                    y_pos -= 0.04
                elif line.startswith("- "):
                    sax.text(0.12, y_pos, line, fontsize=10)
                    y_pos -= 0.03
                else:
                    sax.text(0.08, y_pos, line, fontsize=10)
                    y_pos -= 0.03
                    
                if y_pos < 0.1: # new page if needed (rare for summary)
                    break
            
            pdf.savefig(surr_fig, bbox_inches="tight")
            plt.close(surr_fig)
        except Exception as e:
            print(f"Error generating surrogate PDF page: {e}")

    data = buf.getvalue()
    if filename:
        with open(filename, "wb") as f:
            f.write(data)
        return filename
    return data
