"""
Systemic Tau — Parameter Sensitivity & Robustness Analysis

Core reusable module for evaluating how stable key outputs
(t*, mean τ_s, RECD) are under small changes to analysis parameters.

Designed to be called from Studio (and potentially Desktop/CLI in the future).

Key features:
- Parameter sweeps over window size, global cluster count, etc.
- Stability metrics (std of t*, fraction of configs with stable t*)
- Structured results easy to visualize and include in reports.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from itertools import product
from typing import Dict, Any, List, Optional, Callable, Tuple

import numpy as np
import pandas as pd

# Import the main high-level runner from Studio (preferred) or fallback
try:
    from systemictau.studio.analysis import run_multi_ontological_analysis
    HAS_STUDIO_RUNNER = True
except Exception:
    HAS_STUDIO_RUNNER = False
    run_multi_ontological_analysis = None  # type: ignore


DEFAULT_WINDOW_SIZES = [9, 11, 13, 15, 17]
DEFAULT_GLOBAL_CLUSTERS = [2, 3, 4]
DEFAULT_AGG_METHODS = ["sum"]
DEFAULT_RECD_OPTIONS = [True]


@dataclass
class SensitivityResult:
    """Single configuration result."""
    params: Dict[str, Any]
    metrics: Dict[str, Dict[str, float]]  # scale -> {"mean_tau": , "recd_T_final": , "t_star": }
    config: Dict[str, Any]
    success: bool = True
    error: Optional[str] = None


@dataclass
class SensitivityReport:
    """Full sensitivity analysis output."""
    results: List[SensitivityResult] = field(default_factory=list)
    base_params: Dict[str, Any] = field(default_factory=dict)
    stability: Dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    parameters_tested: List[str] = field(default_factory=list)


def _extract_key_metrics(analysis: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Pull the important numbers we care about for stability."""
    scale_metrics = analysis.get("scale_metrics", {})
    scale_results = analysis.get("scale_results", {})

    out: Dict[str, Dict[str, float]] = {}
    for scale in ("Local", "Medium", "Global"):
        m = scale_metrics.get(scale, {})
        sr = scale_results.get(scale, {})
        out[scale] = {
            "mean_tau": float(m.get("mean_tau", np.nan)),
            "recd_T_final": float(m.get("recd_T_final", np.nan)),
            "t_star": float(sr.get("t_star")) if sr.get("t_star") is not None else np.nan,
            "tau_std": float(m.get("tau_std", np.nan)),
        }
    return out


def run_parameter_sensitivity(
    df: pd.DataFrame,
    value_cols: List[str],
    *,
    # Base configuration (passed through to the runner)
    base_window_size: int = 13,
    base_medium_groups: Optional[Dict[str, List[str]]] = None,
    base_global_groups: Optional[Dict[str, List[str]]] = None,
    base_auto_global: bool = True,
    # Sensitivity parameters
    window_sizes: Optional[List[int]] = None,
    global_cluster_counts: Optional[List[int]] = None,
    agg_methods: Optional[List[str]] = None,
    recd_options: Optional[List[bool]] = None,
    max_combinations: int = 16,
    # Progress
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> SensitivityReport:
    """
    Run the full multi-ontological analysis across combinations of parameters.

    Returns a structured SensitivityReport with per-run metrics and stability scores.
    """
    if not HAS_STUDIO_RUNNER or run_multi_ontological_analysis is None:
        raise RuntimeError("Studio runner (run_multi_ontological_analysis) is required for sensitivity analysis.")

    window_sizes = window_sizes or DEFAULT_WINDOW_SIZES
    global_cluster_counts = global_cluster_counts or DEFAULT_GLOBAL_CLUSTERS
    agg_methods = agg_methods or DEFAULT_AGG_METHODS
    recd_options = recd_options or DEFAULT_RECD_OPTIONS

    # Build the parameter grid (limit to avoid explosion)
    param_grid = list(product(window_sizes, global_cluster_counts, agg_methods, recd_options))

    if len(param_grid) > max_combinations:
        # Prefer keeping the base window and base cluster count in the set
        # Simple strategy: take a balanced sample around the base
        param_grid = param_grid[:max_combinations]

    results: List[SensitivityResult] = []
    tested_params: List[str] = []

    total = len(param_grid)
    for i, (w, g_clusters, agg_m, use_recd) in enumerate(param_grid):
        params = {
            "window_size": int(w),
            "global_num_clusters": int(g_clusters),
            "agg_method": agg_m,
            "recd_enabled": bool(use_recd),
        }

        if progress_callback:
            progress_callback(i + 1, total, f"window={w}, global_clusters={g_clusters}")

        try:
            # Call the main runner with this parameter set.
            # Note: for now we ignore recd_enabled and agg_method in the runner
            # (they are future extension points). We still record them.
            analysis = run_multi_ontological_analysis(
                df,
                value_cols=value_cols,
                window_size=w,
                medium_groups=base_medium_groups,
                global_groups=base_global_groups,
                global_num_clusters=g_clusters,
                auto_global=not bool(base_global_groups),  # respect manual if provided
            )

            metrics = _extract_key_metrics(analysis)
            results.append(SensitivityResult(
                params=params,
                metrics=metrics,
                config=analysis.get("config", {}),
                success=True,
            ))
        except Exception as e:
            results.append(SensitivityResult(
                params=params,
                metrics={},
                config={},
                success=False,
                error=str(e),
            ))

    # Compute stability
    stability = _compute_stability(results)

    # Build human summary
    summary = _build_summary(stability, results)

    report = SensitivityReport(
        results=results,
        base_params={
            "window_size": base_window_size,
            "medium_groups": base_medium_groups or {},
            "global_groups": base_global_groups or {},
            "auto_global": base_auto_global,
        },
        stability=stability,
        summary=summary,
        parameters_tested=["window_size", "global_num_clusters"],
    )
    return report


def _compute_stability(results: List[SensitivityResult]) -> Dict[str, Any]:
    """Calculate stability statistics across successful runs."""
    successful = [r for r in results if r.success and r.metrics]

    if not successful:
        return {"n_runs": 0, "n_successful": 0}

    scales = ["Local", "Medium", "Global"]
    stability: Dict[str, Any] = {"n_runs": len(results), "n_successful": len(successful)}

    for scale in scales:
        t_stars = []
        mean_taus = []
        recd_finals = []

        for r in successful:
            m = r.metrics.get(scale, {})
            if m.get("t_star") is not None and not np.isnan(m.get("t_star", np.nan)):
                t_stars.append(float(m["t_star"]))
            if not np.isnan(m.get("mean_tau", np.nan)):
                mean_taus.append(float(m["mean_tau"]))
            if not np.isnan(m.get("recd_T_final", np.nan)):
                recd_finals.append(float(m["recd_T_final"]))

        if t_stars:
            t_arr = np.array(t_stars)
            t_mean = float(np.nanmean(t_arr))
            t_std = float(np.nanstd(t_arr))
            # Stability score: % of runs where t* is within ±3 of the median
            median_t = float(np.nanmedian(t_arr))
            within_band = np.sum(np.abs(t_arr - median_t) <= 3) / len(t_arr)
            stability[f"{scale}_t_star"] = {
                "mean": round(t_mean, 2),
                "std": round(t_std, 2),
                "median": round(median_t, 2),
                "stability_score_within_3": round(float(within_band), 3),
                "min": int(np.min(t_arr)),
                "max": int(np.max(t_arr)),
                "n": len(t_stars),
            }
        else:
            stability[f"{scale}_t_star"] = None

        if mean_taus:
            stability[f"{scale}_mean_tau"] = {
                "mean": round(float(np.mean(mean_taus)), 4),
                "std": round(float(np.std(mean_taus)), 4),
            }
        if recd_finals:
            stability[f"{scale}_recd_final"] = {
                "mean": round(float(np.mean(recd_finals)), 4),
                "std": round(float(np.std(recd_finals)), 4),
            }

    # Overall stability verdict
    overall_scores = []
    for s in scales:
        sc = stability.get(f"{s}_t_star")
        if sc and "stability_score_within_3" in sc:
            overall_scores.append(sc["stability_score_within_3"])

    if overall_scores:
        stability["overall_t_star_stability"] = round(float(np.mean(overall_scores)), 3)
    else:
        stability["overall_t_star_stability"] = None

    return stability


def _build_summary(stability: Dict[str, Any], results: List[SensitivityResult]) -> str:
    n = stability.get("n_successful", 0)
    if n == 0:
        return "No successful runs."

    lines = [f"Sensitivity Analysis over {n} successful parameter combinations."]

    for scale in ["Local", "Medium", "Global"]:
        sc = stability.get(f"{scale}_t_star")
        if sc:
            lines.append(
                f"{scale}: t* median={sc.get('median')}, std={sc.get('std')}, "
                f"stable within ±3 steps in {sc.get('stability_score_within_3', 0)*100:.0f}% of configs."
            )

    overall = stability.get("overall_t_star_stability")
    if overall is not None:
        verdict = "HIGHLY STABLE" if overall > 0.8 else ("MODERATELY STABLE" if overall > 0.5 else "SENSITIVE TO PARAMETERS")
        lines.append(f"Overall verdict: {verdict} (avg stability {overall:.0%}).")

    return "\n".join(lines)


# Convenience: simple window-size only sweep (very fast to call)
def quick_window_sensitivity(
    df: pd.DataFrame,
    value_cols: List[str],
    windows: List[int] = None,
    **kwargs
) -> SensitivityReport:
    windows = windows or [9, 11, 13, 15, 17]
    return run_parameter_sensitivity(
        df,
        value_cols,
        window_sizes=windows,
        global_cluster_counts=[kwargs.get("base_global", 3)],
        max_combinations=len(windows),
        **{k: v for k, v in kwargs.items() if k not in ("window_sizes", "global_cluster_counts")},
    )
