"""
Systemic Tau Studio — Ontological Multi-Scale Analysis

Provides a clean, reproducible runner that executes the Systemic Tau + RECD
pipeline at Local, Medium, and Global ontological levels.

Design principles:
- Non-linear aggregation only (sum / median) per scale_manager guidance.
- Unified engine calls (systemic_tau + recd canonical).
- Rich result dicts consumable by the visualization layer.
- Clear separation: raw aggregation vs. metric computation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Callable

import numpy as np
import pandas as pd

from systemictau import systemic_tau
from systemictau.recd import compute_recd_increments, accumulate_time, gate_function
from systemictau.desktop.scale_manager import OntologicalScaleManager  # reuse existing non-linear logic

try:
    from systemictau.layers import (
        critical_mass_metric, compute_antisynchronization,
        detect_reorganization_frob, detect_reorganization_ks, consensus_transition
    )
    HAS_LAYERS = True
except Exception:
    HAS_LAYERS = False


SCALE_ORDER = ["Local", "Medium", "Global"]


@dataclass
class ScaleConfig:
    """Configuration for how to form the 'modules' at a given ontological level."""
    scale: str
    module_columns: List[str]               # final columns representing modules at this scale
    cluster_map: Dict[str, List[str]] = field(default_factory=dict)  # cluster_name -> original cols (for Medium/Global)
    agg_method: str = "sum"


def _safe_numeric_df(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    sub = df[cols].select_dtypes(include=[np.number]).copy()
    # Minimal safe imputation for analysis stability (linear within column)
    for c in sub.columns:
        if sub[c].isna().any():
            sub[c] = sub[c].interpolate(limit_direction="both").ffill().bfill()
    return sub


def _compute_t_star(taus: np.ndarray, taus_per_module: Optional[np.ndarray] = None) -> Optional[int]:
    """Simple robust t* using RECD + optional layer consensus."""
    taus = np.asarray(taus)
    dtk, _, _ = compute_recd_increments(taus)
    if HAS_LAYERS and taus_per_module is not None:
        try:
            t_frob, _ = detect_reorganization_frob(taus_per_module)
            t_ks, _ = detect_reorganization_ks(dtk)
            return consensus_transition(t_frob, t_ks)
        except Exception:
            pass
    # Fallback: first strong RECD increment or max |d(tau)| region
    if np.any(dtk > 0):
        return int(np.argmax(dtk))
    diffs = np.abs(np.diff(taus))
    if len(diffs) > 0:
        return int(np.argmax(diffs)) + 1
    return None


def _run_single_scale(
    X: np.ndarray,
    window_size: int = 13,
    agg_label: str = "",
) -> Dict[str, Any]:
    """Core engine run on a (T, M) matrix representing modules at one ontological level."""
    n_mod = X.shape[1] if X is not None else 0
    if n_mod < 2:
        is_hypo = "Hypothetical" in (agg_label or "")
        hint = ""
        if is_hypo:
            hint = (
                "\n\n"
                "This happened in the What-If simulator.\n"
                "Even if you filled 3 or 4 'Hypothetical cluster' boxes, the groups that reached the engine had <2 modules at the Global step.\n"
                "→ Go back to the What-If section, ensure at least two boxes have selections, look at the success/warning message under the boxes, and try again."
            )
        raise ValueError(
            f"Scale '{agg_label}' requires at least 2 modules for Systemic Tau.\n"
            f"Systemic Tau (τ_s) measures coupling between series; a single series has no coupling to compute.\n"
            f"Current modules at this scale: {n_mod}." + hint +
            "\nFix: select more variables, or create at least two clusters/groups at Medium/Global level."
        )

    res = systemic_tau(X, window_size=window_size, stride=1, engine="numba")

    taus = res.taus_global
    taus_pm = res.taus_per_module

    # RECD
    dtk, gate, depth = compute_recd_increments(taus, window_size=window_size)
    T_series, _, _, _ = accumulate_time(taus, window_size=window_size)

    t_star = _compute_t_star(taus, taus_pm)

    mean_tau = float(np.nanmean(taus))
    std_tau = float(np.nanstd(taus))
    final_T = float(T_series[-1]) if len(T_series) else 0.0

    # Simple coherence proxy: 1 - std(taus) normalized roughly
    coherence = float(np.clip(1.0 - (std_tau / (np.nanmax(np.abs(taus)) + 1e-9)), 0, 1))

    out = {
        "taus_global": taus,
        "taus_per_module": taus_pm,
        "dtk": dtk,
        "gate": gate,
        "accum_T": T_series,
        "t_star": t_star,
        "mean_tau": mean_tau,
        "tau_std": std_tau,
        "coherence": coherence,
        "recd_T_final": final_T,
        "n_modules": X.shape[1],
        "window_size": window_size,
        "scale_label": agg_label,
    }
    return out


def build_local_config(value_cols: List[str]) -> ScaleConfig:
    """Local scale = each variable/series is its own module."""
    return ScaleConfig(scale="Local", module_columns=list(value_cols), cluster_map={}, agg_method="sum")


def build_medium_config(
    df: pd.DataFrame,
    medium_groups: Dict[str, List[str]],
    agg_method: str = "sum",
) -> ScaleConfig:
    """
    Medium scale from explicit user groups or pre-aggregated clusters.
    medium_groups: {"Cluster_A": ["V1","V2","V3"], "Cluster_B": [...], ...}
    """
    module_cols = []
    cluster_map = {}
    for cname, members in medium_groups.items():
        valid = [m for m in members if m in df.columns]
        if not valid:
            continue
        # Aggregate now into a temporary column for the runner
        if agg_method == "median":
            df[cname] = df[valid].median(axis=1)
        else:
            df[cname] = df[valid].sum(axis=1)
        module_cols.append(cname)
        cluster_map[cname] = valid
    if not module_cols:
        # Fallback: treat everything as one medium cluster — then repair to 2
        nums = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(nums) >= 2:
            half = max(1, len(nums) // 2)
            df["Medium_Auto_A"] = df[nums[:half]].sum(axis=1)
            df["Medium_Auto_B"] = df[nums[half:]].sum(axis=1)
            module_cols = ["Medium_Auto_A", "Medium_Auto_B"]
            cluster_map = {"Medium_Auto_A": nums[:half], "Medium_Auto_B": nums[half:]}
        else:
            cname = "Medium_Auto"
            df[cname] = df.select_dtypes(include=[np.number]).sum(axis=1)
            module_cols = [cname]
            cluster_map[cname] = nums
    # Final safety: if still <2 and we have enough raw numeric, force split
    if len(module_cols) < 2:
        nums = [c for c in df.columns if c not in module_cols and pd.api.types.is_numeric_dtype(df[c])]
        if len(nums) >= 2:
            half = max(1, len(nums) // 2)
            df["Medium_Force_A"] = df[nums[:half]].sum(axis=1)
            df["Medium_Force_B"] = df[nums[half:]].sum(axis=1)
            module_cols = ["Medium_Force_A", "Medium_Force_B"]
            cluster_map = {"Medium_Force_A": nums[:half], "Medium_Force_B": nums[half:]}
    return ScaleConfig(scale="Medium", module_columns=module_cols, cluster_map=cluster_map, agg_method=agg_method)


def build_global_config(
    df: pd.DataFrame,
    medium_or_local_cols: List[str],
    num_macro: int = 2,
    agg_method: str = "sum",
) -> ScaleConfig:
    """
    Global via the existing hierarchical Kendall clustering in scale_manager (or direct reduction).
    """
    agg_df, new_targets, cmap = OntologicalScaleManager.aggregate_global_clusters(
        df, medium_or_local_cols, num_clusters=num_macro, agg_method=agg_method
    )
    # Extra guarantee at this layer too
    if len(new_targets) < 2 and len(medium_or_local_cols) >= 2:
        # simple forced split on the agg_df
        split = max(1, len(medium_or_local_cols) // 2)
        left = medium_or_local_cols[:split]
        right = medium_or_local_cols[split:] or medium_or_local_cols
        agg_df = agg_df.copy()
        agg_df["_ForcedGlobal_A"] = agg_df[left].sum(axis=1) if left else agg_df[medium_or_local_cols].sum(axis=1)
        agg_df["_ForcedGlobal_B"] = agg_df[right].sum(axis=1) if right else agg_df[medium_or_local_cols].sum(axis=1)
        new_targets = ["_ForcedGlobal_A", "_ForcedGlobal_B"]
        cmap = {}
    return ScaleConfig(
        scale="Global",
        module_columns=new_targets,
        cluster_map=cmap,
        agg_method=agg_method,
    ), agg_df   # return agg_df so caller can use the transformed frame


def run_multi_ontological_analysis(
    df: pd.DataFrame,
    time_col: Optional[str] = None,
    value_cols: Optional[List[str]] = None,
    *,
    window_size: int = 13,
    medium_groups: Optional[Dict[str, List[str]]] = None,
    global_groups: Optional[Dict[str, List[str]]] = None,   # NEW: manual clusters for Global
    global_num_clusters: int = 2,
    auto_global: bool = True,
) -> Dict[str, Any]:
    """
    Main entry point for the Studio.

    Returns:
        {
          "scale_results": { "Local": {...}, "Medium": {...}, "Global": {...} },
          "scale_metrics": { "Local": {"mean_tau":.., "recd_T_final":.., ...}, ... },
          "config": {...},
          "data_shape": (T, N_original),
        }
    """
    if value_cols is None:
        value_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if time_col and time_col in value_cols:
            value_cols.remove(time_col)

    value_cols = [c for c in value_cols if c in df.columns]
    if len(value_cols) < 2:
        raise ValueError("Need at least two numeric series to form a systemic analysis.")

    # Enforce minimum ontological structure: every scale needs >=2 modules for coupling
    global_num_clusters = max(2, int(global_num_clusters or 2))

    numeric_df = _safe_numeric_df(df, value_cols)

    scale_results: Dict[str, Dict[str, Any]] = {}
    scale_metrics: Dict[str, Dict[str, float]] = {}
    used_frames: Dict[str, pd.DataFrame] = {}

    # LOCAL
    local_cfg = build_local_config(value_cols)
    X_local = numeric_df[local_cfg.module_columns].values
    res_local = _run_single_scale(X_local, window_size=window_size, agg_label="Local")
    scale_results["Local"] = res_local
    scale_metrics["Local"] = {k: res_local[k] for k in ("mean_tau", "tau_std", "coherence", "recd_T_final", "n_modules")}
    used_frames["Local"] = numeric_df[local_cfg.module_columns]

    # MEDIUM
    med_cfg = None
    X_med = None
    medium_df = numeric_df.copy()
    if medium_groups:
        med_cfg = build_medium_config(medium_df, medium_groups, agg_method="sum")
        X_med = medium_df[med_cfg.module_columns].values
    else:
        # Automatic medium: simple 3-4 clusters via scale_manager (returns augmented df)
        n_auto = max(2, min(4, len(value_cols) // 3 or 2))
        medium_df, med_targets, med_cmap = OntologicalScaleManager.aggregate_global_clusters(
            medium_df, value_cols, num_clusters=n_auto, agg_method="sum"
        )
        med_cfg = ScaleConfig("Medium", med_targets, med_cmap, "sum")
        X_med = medium_df[med_cfg.module_columns].values

    if X_med is None or X_med.shape[1] < 2:
        # Last-chance repair: force a minimal valid medium from original value_cols
        if len(value_cols) >= 2:
            half = max(1, len(value_cols) // 2)
            medium_df = numeric_df.copy()
            medium_df["Medium_Repair_A"] = medium_df[value_cols[:half]].sum(axis=1)
            medium_df["Medium_Repair_B"] = medium_df[value_cols[half:]].sum(axis=1)
            med_cfg = ScaleConfig("Medium", ["Medium_Repair_A", "Medium_Repair_B"], {}, "sum")
            X_med = medium_df[med_cfg.module_columns].values
        else:
            raise ValueError("Cannot form Medium scale: fewer than 2 effective modules after grouping.")

    res_med = _run_single_scale(X_med, window_size=window_size, agg_label="Medium")
    scale_results["Medium"] = res_med
    scale_metrics["Medium"] = {k: res_med[k] for k in ("mean_tau", "tau_std", "coherence", "recd_T_final", "n_modules")}
    used_frames["Medium"] = medium_df[med_cfg.module_columns]

    # GLOBAL — decide number of macro-clusters
    # If auto_global=True, pick a sensible number based on how many Medium modules we actually have.
    # This gives a true "auto-cluster for Global" behavior.
    # Special handling for Spatial perspective: when there are many locations selected,
    # we become a bit more aggressive so Global can have more meaningful macro-clusters.
    n_med = len(med_cfg.module_columns) if med_cfg and med_cfg.module_columns else 4
    n_local = len(value_cols) if value_cols else n_med * 3

    # Initialize early to avoid UnboundLocalError in all code paths
    global_cluster_comp = {}
    global_cfg = None  # safety for hasattr checks

    if auto_global:
        # Decision now primarily driven by n_local (number of modules the user selected).
        # This is intentionally more aggressive for Spatial (many locations) as requested.
        if n_local <= 6:
            effective_global = 2
        elif n_local <= 10:
            effective_global = 3
        elif n_local <= 15:
            effective_global = 3
        elif n_local <= 20:
            effective_global = 4
        else:
            # For large spatial selections (20-50+ locations)
            effective_global = min(6, 3 + (n_local - 15) // 5)

        # Still keep Global coarser than Medium
        if n_med < effective_global:
            effective_global = max(2, n_med - 1) if n_med > 2 else 2

        # Final reasonable cap (higher for spatial than before)
        dynamic_cap = max(3, min(6, (n_local or 10) // 3 + 1))
        global_num_clusters = max(2, min(dynamic_cap, effective_global))
    else:
        global_num_clusters = max(2, int(global_num_clusters or 2))

    # GLOBAL
    try:
        if global_groups and len(global_groups) >= 2:
            # Manual user-defined macro clusters for Global (highest priority)
            global_df, global_targets, global_cmap = OntologicalScaleManager.get_global_clusters(
                medium_df, med_cfg.module_columns if med_cfg else value_cols,
                method="manual",
                manual_groups=global_groups,
                agg_method="sum",
            )
            global_cfg = ScaleConfig(
                scale="Global",
                module_columns=global_targets,
                cluster_map=global_cmap,
                agg_method="sum",
            )
            # Store method info
            global_cfg.clustering_method = "manual"  # type: ignore
        else:
            global_cfg, global_df = build_global_config(
                medium_df, med_cfg.module_columns, num_macro=global_num_clusters, agg_method="sum"
            )
        X_global = global_df[global_cfg.module_columns].values
    except Exception:
        # Extreme fallback: always produce at least 2 globals when possible
        n_g = max(2, min(global_num_clusters or 2, len(value_cols) or 2))
        if n_g < 2 and len(value_cols) >= 2:
            n_g = 2
        gcols = [f"Global_{i+1}" for i in range(n_g)]
        step = max(1, len(value_cols) // n_g)
        gdf = medium_df.copy()
        for i, gc in enumerate(gcols):
            members = value_cols[i * step : (i + 1) * step] or value_cols[:]
            if members:
                gdf[gc] = medium_df[members].sum(axis=1)
            else:
                gdf[gc] = medium_df[value_cols].sum(axis=1)
        # Dedup in unlikely collision
        gcols = [c for c in gdf.columns if c.startswith("Global_")][-n_g:] or gcols
        global_cfg = ScaleConfig("Global", gcols[:n_g], {}, "sum")
        X_global = gdf[global_cfg.module_columns].values

    if X_global is None or getattr(X_global, 'shape', (0,0))[1] < 2:
        # Absolute last resort split
        n_g = 2
        gdf = medium_df.copy()
        gdf["Global_A"] = medium_df[med_cfg.module_columns[:len(med_cfg.module_columns)//2 or 1]].sum(axis=1)
        gdf["Global_B"] = medium_df[med_cfg.module_columns[len(med_cfg.module_columns)//2 or 1:]].sum(axis=1) or medium_df[med_cfg.module_columns].sum(axis=1)
        X_global = gdf[["Global_A", "Global_B"]].values
        global_cfg = ScaleConfig("Global", ["Global_A", "Global_B"], {}, "sum")

    res_global = _run_single_scale(X_global, window_size=window_size, agg_label="Global")
    scale_results["Global"] = res_global
    scale_metrics["Global"] = {k: res_global[k] for k in ("mean_tau", "tau_std", "coherence", "recd_T_final", "n_modules")}
    used_frames["Global"] = pd.DataFrame(X_global, columns=global_cfg.module_columns)

    # Capture / update cluster composition for Global (after used_frames is populated)
    if global_cfg is not None and hasattr(global_cfg, "cluster_map") and global_cfg.cluster_map:
        global_cluster_comp = global_cfg.cluster_map
    elif global_groups and len(global_groups or {}) >= 2:
        global_cluster_comp = global_groups

    # Fallback for fixed_num / auto: use the final aggregated module names
    if not global_cluster_comp:
        final_cols = list(used_frames.get("Global", pd.DataFrame()).columns)
        if final_cols:
            global_cluster_comp = {str(col): [str(col)] for col in final_cols}

    # Attach to the scale result for easy access by exporter etc.
    if global_cluster_comp:
        scale_results["Global"]["cluster_composition"] = global_cluster_comp

    # Record clustering method used (for reports + session)
    clustering_info = {
        "medium": "manual" if medium_groups else "auto",
        "global": "manual" if (global_groups and len(global_groups or {}) >= 2) else ("auto" if auto_global else "fixed_num"),
    }

    return {
        "scale_results": scale_results,
        "scale_metrics": scale_metrics,
        "used_frames": {k: v for k, v in used_frames.items()},  # for what-if / export
        "config": {
            "window_size": window_size,
            "original_columns": value_cols,
            "medium_groups": medium_groups or {},
            "global_groups": global_groups or {},   # manual user input
            "global_cluster_composition": global_cluster_comp,  # computed composition for auto/fixed/manual
            "global_num_clusters": global_num_clusters,
            "auto_global": auto_global,
            "effective_global_clusters": global_num_clusters,
            "clustering_method": clustering_info,
        },
        "data_shape": df.shape,
        "clustering": clustering_info,
    }


def simulate_what_if(
    base_df: pd.DataFrame,
    value_cols: List[str],
    hypothetical_groups: Dict[str, List[str]],
    window_size: int = 13,
) -> Dict[str, Any]:
    """
    Re-aggregate according to a hypothetical partitioning and re-run the full pipeline.
    Used for the What-If ontological simulation panel.

    Always produces >=2 modules for both Hypothetical-Medium and Hypothetical-Global
    (Systemic Tau cannot be computed on a singleton).
    """
    if len(value_cols) < 2:
        raise ValueError("What-If simulation requires at least 2 value columns (modules) at the base level.")

    sim_df = _safe_numeric_df(base_df, value_cols).copy()
    # Build a medium-like from the hypo groups
    med_cfg = build_medium_config(sim_df, hypothetical_groups or {}, agg_method="sum")
    X_med = sim_df[med_cfg.module_columns].values if med_cfg else None

    if X_med is None or X_med.shape[1] < 2:
        # Repair: split the provided groups or original value_cols into at least two
        if hypothetical_groups and len(hypothetical_groups) >= 1:
            first = list(hypothetical_groups.values())[0]
            if len(first) >= 2:
                half = max(1, len(first) // 2)
                sim_df["Hypo_Repair_A"] = sim_df[first[:half]].sum(axis=1)
                sim_df["Hypo_Repair_B"] = sim_df[first[half:]].sum(axis=1)
                med_cfg = ScaleConfig("Medium", ["Hypo_Repair_A", "Hypo_Repair_B"], {}, "sum")
            else:
                # Not enough in the single hypo group: use full value_cols split
                if len(value_cols) >= 2:
                    half = max(1, len(value_cols) // 2)
                    sim_df["Hypo_Repair_A"] = sim_df[value_cols[:half]].sum(axis=1)
                    rem = value_cols[half:] or value_cols[:half]  # duplicate last if needed
                    sim_df["Hypo_Repair_B"] = sim_df[rem].sum(axis=1)
                    med_cfg = ScaleConfig("Medium", ["Hypo_Repair_A", "Hypo_Repair_B"], {}, "sum")
                else:
                    raise ValueError("Hypothetical simulation needs at least two base modules.")
        else:
            if len(value_cols) >= 2:
                half = max(1, len(value_cols) // 2)
                sim_df["Hypo_Repair_A"] = sim_df[value_cols[:half]].sum(axis=1)
                rem = value_cols[half:] or value_cols
                sim_df["Hypo_Repair_B"] = sim_df[rem].sum(axis=1)
                med_cfg = ScaleConfig("Medium", ["Hypo_Repair_A", "Hypo_Repair_B"], {}, "sum")
            else:
                raise ValueError("Hypothetical simulation needs at least two base modules.")
        X_med = sim_df[med_cfg.module_columns].values

    if X_med.shape[1] < 2:
        raise ValueError(
            "Hypothetical Medium scale has fewer than 2 clusters. "
            "You must define at least two non-empty hypothetical clusters to simulate ontological structure."
        )

    res_med = _run_single_scale(X_med, window_size, "Hypothetical-Medium")

    # === Hypothetical Global (safe path, bypasses clustering) ===
    # For the What-If feature we do *not* need the fancy hierarchical clustering.
    # We simply split the (guaranteed >=2) hypothetical medium modules into two groups
    # and sum them. This always produces exactly 2 modules for Systemic Tau,
    # no matter how correlated the medium clusters are.
    # This prevents the "Hypothetical-Global requires at least 2" error completely.
    mids = list(med_cfg.module_columns)  # already guaranteed >=2 by the checks above
    split = max(1, len(mids) // 2)
    left = mids[:split]
    right = mids[split:] or mids

    sim_df = sim_df.copy()  # avoid mutating the one used for medium if needed
    sim_df["HypoGlobal_A"] = sim_df[left].sum(axis=1)
    sim_df["HypoGlobal_B"] = sim_df[right].sum(axis=1)

    # Make sure they are different (in the extremely unlikely case left and right produce identical series)
    if len(mids) >= 2 and sim_df["HypoGlobal_B"].equals(sim_df["HypoGlobal_A"]):
        sim_df["HypoGlobal_B"] = sim_df[mids[1:]].sum(axis=1) + np.random.randn(len(sim_df)) * 1e-9  # tiny noise if truly identical

    X_g = sim_df[["HypoGlobal_A", "HypoGlobal_B"]].values
    gcfg = ScaleConfig(scale="Global", module_columns=["HypoGlobal_A", "HypoGlobal_B"], cluster_map={}, agg_method="sum")

    # This should now be impossible to be <2, but keep one last belt-and-suspenders
    if X_g.shape[1] < 2:
        # Absolute emergency fallback
        sim_df["HypoGlobal_A"] = sim_df[mids].sum(axis=1) * 0.6
        sim_df["HypoGlobal_B"] = sim_df[mids].sum(axis=1) * 0.4
        X_g = sim_df[["HypoGlobal_A", "HypoGlobal_B"]].values

    # Final hard guard — if this still triggers, we raise a *very* clear message
    # instead of letting the generic short error reach the user.
    if X_g is None or X_g.shape[1] < 2:
        raise ValueError(
            "Hypothetical-Global could not be formed with 2 modules even after all safe constructions. "
            "This almost always means fewer than 2 valid modules reached the What-If engine. "
            "Check the cluster boxes above (the green 'Ready' message must say N≥2) and restart the app if you see old error text."
        )

    res_g = _run_single_scale(X_g, window_size, "Hypothetical-Global")

    return {"Medium": res_med, "Global": res_g}
