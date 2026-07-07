"""
Módulo principal de análisis del paradigma Tau Sistémico.

Contiene la función de alto nivel `run_full_analysis` que orquesta
todo el pipeline (Capas 1, 2 y 3 + RECD + Dimensión Fractal).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np

from .core import compute_taus
from .recd import accumulate_time
from .layers import (
    hyper_persistence,
    rolling_rqa,
    critical_mass_metric,
    extract_joint_episodes,
    detect_reorganization_frob,
    detect_reorganization_ks,
    consensus_transition,
)
from .fractal import estimate_higuchi_dimension
from .results import OntologicalAscentResult
from .topology import compute_tda_features, HAS_RIPSER


def run_full_analysis(
    X: np.ndarray,
    window_size: int = 13,
    theta_A: float = 0.05,
    D_min: int = 10,
    validate_fractal: bool = True,
    expected_fractal_range: tuple[float, float] = (1.85, 2.15),
    component_names: Optional[List[str]] = None,
    adaptive_breathing: bool = False,
    compute_tda: bool = False,
    compute_ordinal: bool = False,
    **kwargs,
) -> OntologicalAscentResult:
    """
    Ejecuta el pipeline completo de análisis del paradigma Tau Sistémico.

    Esta función es de propósito general y puede usarse con cualquier
    serie temporal multivariada.

    Parameters
    ----------
    X : np.ndarray
        Matriz de datos de forma (T, N).
    window_size : int, default=13
        Tamaño de la ventana deslizante para el cálculo de Tau.
    theta_A : float, default=0.05
        Umbral para la extracción de Joint Episodes.
    D_min : int, default=10
        Duración mínima de un Joint Episode.
    validate_fractal : bool, default=True
        Si es True, verifica si la dimensión fractal está dentro del rango esperado.
    expected_fractal_range : tuple, default=(1.85, 2.15)
        Rango esperado para la dimensión fractal. Fuera de este rango se genera una advertencia.

    Returns
    -------
    OntologicalAscentResult
        Objeto inmutable con todos los resultados del análisis.
    """

    if component_names is None:
        component_names = [f"Component_{i}" for i in range(X.shape[1])]

    warnings: List[str] = []
    
    metadata = {
        "components": component_names,
        "parameters": {
            "window_size": window_size,
            "theta_A": theta_A,
            "D_min": D_min,
            "expected_fractal_range": expected_fractal_range
        }
    }

    if X.shape[1] < 3:
        warnings.append(
            f"Low Dimensionality: The matrix has only {X.shape[1]} components. "
            "The Systemic Tau paradigm is most robust for analyzing networks of N >= 3."
        )

    # === 1. Cálculo de Tau ===
    taus_global, taus_per_module = compute_taus(X, window_size=window_size, adaptive=adaptive_breathing)

    # === 2. Reloj Extramental Discreto (RECD) ===
    T_series, dtk_series, _, _ = accumulate_time(taus_global)
    
    # === 2b. Teorema 24v: Coherencia Ordinal Exacta (Árbol Diádico) ===
    abs_tau = np.abs(taus_global)
    k_series = np.full_like(abs_tau, np.inf, dtype=float)
    valid_mask = abs_tau > 0.5
    denominator = 2 * abs_tau[valid_mask] - 1
    k_series[valid_mask] = np.log2(abs_tau[valid_mask] / denominator)
    estimated_period_series = np.where(np.isinf(k_series), np.inf, 2 ** k_series)

    # === 3. Capa 1: Hiper-persistencia y RQA ===
    try:
        hp_z_array, _ = hyper_persistence(taus_global)
        lam_array, tt_array = rolling_rqa(taus_global)
        M_series = critical_mass_metric(hp_z_array, lam_array, tt_array)

        hp_z = float(np.asarray(np.nanmean(np.abs(hp_z_array))).item()) if len(hp_z_array) > 0 else 0.0
        lam = float(np.asarray(np.nanmean(lam_array)).item()) if len(lam_array) > 0 else 0.0
        tt = float(np.asarray(np.nanmean(tt_array)).item()) if len(tt_array) > 0 else 0.0
        M_max = float(np.asarray(np.max(M_series)).item()) if len(M_series) > 0 else 0.0
        M_mean = float(np.asarray(np.mean(M_series)).item()) if len(M_series) > 0 else 0.0
    except Exception as e:
        warnings.append(f"Error en Capa 1 (RQA): {str(e)}")
        M_series = np.zeros_like(taus_global)
        hp_z = lam = tt = M_max = M_mean = 0.0

    # === 4. Capa 2: Joint Episodes (MEJORADO) ===
    episodes = []
    try:
        raw_episodes = extract_joint_episodes(taus_global, M_series, theta_A=theta_A, D_min=D_min)
        
        for ep in raw_episodes:
            start = int(ep.get("start", 0))
            end = int(ep.get("end", start))
            duration = end - start + 1
            m_segment = M_series[start : end + 1] if len(M_series) > end else np.array([0.0])

            mean_m = float(np.mean(m_segment)) if len(m_segment) > 0 else 0.0
            total_j = float(np.sum(m_segment)) if len(m_segment) > 0 else 0.0

            if len(m_segment) > 0:
                above_threshold = np.sum(np.abs(m_segment) > 0.7)
                j_above_07 = (above_threshold / len(m_segment)) * 100
            else:
                j_above_07 = 0.0

            episodes.append({
                "start": start,
                "end": end,
                "duration": duration,
                "mean_m": round(mean_m, 4),
                "total_j": round(total_j, 4),
                "j_above_07": round(j_above_07, 2),
            })
    except Exception as e:
        warnings.append(f"Error en Capa 2 (Joint Episodes): {str(e)}")

    # === 5. Capa 3: Detección de Reorganización ===
    t_frob = None
    t_ks = None
    t_star = None
    frob_max = 0.0
    ks_max = 0.0
    
    try:
        if taus_per_module is not None and taus_per_module.ndim > 1:
            t_frob, raw_frob_max = detect_reorganization_frob(taus_per_module)
            frob_max = float(np.asarray(raw_frob_max).item())
        
        t_ks, raw_ks_max = detect_reorganization_ks(dtk_series)
        ks_max = float(np.asarray(raw_ks_max).item())
        t_star = consensus_transition(t_frob, t_ks)
    except Exception as e:
        warnings.append(f"Error en detección de reorganización: {str(e)}")

    # === 6. Dimensión Fractal ===
    fractal_D = None
    try:
        raw_fractal_D = estimate_higuchi_dimension(dtk_series, k_max=20)
        fractal_D = float(np.asarray(raw_fractal_D).item())

        if validate_fractal:
            low, high = expected_fractal_range
            if not (low <= fractal_D <= high):
                warnings.append(
                    f"Dimensión fractal atípica: {fractal_D:.3f} "
                    f"(rango esperado: {low} - {high})"
                )
    except Exception as e:
        warnings.append(f"Error calculando dimensión fractal: {str(e)}")

    # === 7. Geospatial Topology (TDA) ===
    tda_results = None
    if compute_tda:
        if HAS_RIPSER:
            try:
                stride = kwargs.get("tda_stride", 5)
                mode = kwargs.get("tda_mode", "fast")
                thresh = kwargs.get("tda_persistence_threshold", 0.1)
                tda_results = compute_tda_features(
                    X, 
                    window_size=window_size,
                    stride=stride,
                    mode=mode,
                    persistence_threshold=thresh
                )
            except Exception as e:
                warnings.append(f"Error computing TDA features: {str(e)}")
        else:
            warnings.append("TDA requested but 'ripser' library is not installed.")

    # === 8. Ordinal Memory (Feature #3) ===
    ordinal_results = None
    if compute_ordinal:
        try:
            from systemictau.ordinal_memory import compute_ordinal_features
            ordinal_stride = kwargs.get("ordinal_stride", 5)
            ordinal_mode = kwargs.get("ordinal_mode", "lite")
            ordinal_m = kwargs.get("ordinal_m", 3)
            ordinal_delay = kwargs.get("ordinal_delay", 1)
            
            ordinal_results = compute_ordinal_features(
                X,
                window_size=window_size,
                stride=ordinal_stride,
                mode=ordinal_mode,
                m=ordinal_m,
                delay=ordinal_delay
            )
        except Exception as e:
            warnings.append(f"Error computing Ordinal Memory: {str(e)}")

    return OntologicalAscentResult(
        X=X,
        window_size=window_size,
        taus_global=taus_global,
        T_series=T_series,
        hp_z=hp_z,
        lam=lam,
        tt=tt,
        M_max=M_max,
        M_mean=M_mean,
        episodes=episodes,
        t_star=t_star,
        t_frob=t_frob,
        t_ks=t_ks,
        frob_max=frob_max,
        ks_max=ks_max,
        fractal_D=fractal_D,
        warnings=warnings,
        metadata=metadata,
        dtk_series=dtk_series,
        taus_per_module=taus_per_module,
        component_names=component_names,
        dyadic_k_series=k_series,
        estimated_period_series=estimated_period_series,
        tda_results=tda_results,
        ordinal_results=ordinal_results,
        figures=None
    )
