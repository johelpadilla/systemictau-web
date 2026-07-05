"""
Módulo principal de análisis del paradigma Tau Sistémico.

Contiene la función de alto nivel `run_full_analysis` que orquesta
todo el pipeline (Capas 1, 2 y 3 + RECD + Dimensión Fractal).
"""

from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass
class TauAnalysisResults:
    """
    Contenedor de resultados del análisis completo de Tau Sistémico.

    Este objeto está diseñado para ser fácil de consumir por el generador
    de reportes académicos y por la interfaz de usuario.
    """

    # === Datos de entrada ===
    X: np.ndarray                          # Matriz original (T, N)
    window_size: int                       # Tamaño de ventana usado

    # === Capa 1: Persistencia y Criticidad ===
    taus_global: np.ndarray
    T_series: np.ndarray                   # Tiempo sistémico acumulado (RECD)
    hp_z: float                            # Z-score de hiper-persistencia
    lam: float                             # Laminaridad (RQA)
    tt: float                              # Trapping Time (RQA)
    M_max: float                           # Masa crítica máxima
    M_mean: float                          # Masa crítica media

    # === Capa 2: Joint Episodes ===
    episodes: List[Dict[str, Any]]         # Lista de episodios conjuntos detectados

    # === Capa 3: Reorganización ===
    t_star: Optional[int] = None           # Punto de reorganización consensuado
    t_frob: Optional[int] = None
    t_ks: Optional[int] = None
    frob_max: Optional[float] = None
    ks_max: Optional[float] = None

    # === Dimensión Fractal ===
    fractal_D: Optional[float] = None

    # === Metadatos y Validación ===
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    figures: Optional[Dict[str, str]] = None
    
    # === Additions for report compatibility ===
    dtk_series: np.ndarray = field(default_factory=lambda: np.array([]))
    taus_per_module: np.ndarray = field(default_factory=lambda: np.array([]))
    component_names: List[str] = field(default_factory=list)
    
    # === Teorema 24v ===
    dyadic_k_series: np.ndarray = field(default_factory=lambda: np.array([]))
    estimated_period_series: np.ndarray = field(default_factory=lambda: np.array([]))


def run_full_analysis(
    X: np.ndarray,
    window_size: int = 13,
    theta_A: float = 0.05,
    D_min: int = 10,
    validate_fractal: bool = True,
    expected_fractal_range: tuple[float, float] = (1.85, 2.15),
    component_names: Optional[List[str]] = None,
    **kwargs,
) -> TauAnalysisResults:
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
    TauAnalysisResults
        Objeto con todos los resultados del análisis.
    """

    if component_names is None:
        component_names = [f"Component_{i}" for i in range(X.shape[1])]

    results = TauAnalysisResults(
        X=X,
        window_size=window_size,
        component_names=component_names,
        taus_global=np.array([]),
        T_series=np.array([]),
        hp_z=np.nan,
        lam=np.nan,
        tt=np.nan,
        M_max=np.nan,
        M_mean=np.nan,
        episodes=[],
        metadata={
            "components": component_names,
            "parameters": {
                "window_size": window_size,
                "theta_A": theta_A,
                "D_min": D_min,
                "expected_fractal_range": expected_fractal_range
            }
        }
    )

    # === 1. Cálculo de Tau ===
    taus_global, taus_per_module = compute_taus(X, window_size=window_size)
    results.taus_global = taus_global
    results.taus_per_module = taus_per_module

    # === 2. Reloj Extramental Discreto (RECD) ===
    T_series, dtk_series, _, _ = accumulate_time(taus_global)
    results.T_series = T_series
    results.dtk_series = dtk_series
    
    # === 2b. Teorema 24v: Coherencia Ordinal Exacta (Árbol Diádico) ===
    # E[|tau|]_k = (2^(k-1)) / (2^k - 1)
    # Inverse map: k = log2( |tau| / (2|tau| - 1) )
    abs_tau = np.abs(taus_global)
    
    # Avoid division by zero or log of negative numbers
    # When |tau| <= 0.5, the system is at or beyond the Feigenbaum accumulation point (Chaos, k -> infinity)
    k_series = np.full_like(abs_tau, np.inf, dtype=float)
    
    valid_mask = abs_tau > 0.5
    denominator = 2 * abs_tau[valid_mask] - 1
    
    # Calculate k for valid values
    k_series[valid_mask] = np.log2(abs_tau[valid_mask] / denominator)
    
    results.dyadic_k_series = k_series
    # Period of stable orbit is 2^k
    results.estimated_period_series = np.where(np.isinf(k_series), np.inf, 2 ** k_series)

    # === 3. Capa 1: Hiper-persistencia y RQA ===
    try:
        hp_z, _ = hyper_persistence(taus_global)
        lam, tt = rolling_rqa(taus_global)
        M_series = critical_mass_metric(hp_z, lam, tt)

        results.hp_z = float(np.asarray(np.nanmean(np.abs(hp_z))).item()) if len(hp_z) > 0 else 0.0
        results.lam = float(np.asarray(np.nanmean(lam)).item()) if len(lam) > 0 else 0.0
        results.tt = float(np.asarray(np.nanmean(tt)).item()) if len(tt) > 0 else 0.0
        results.M_max = float(np.asarray(np.max(M_series)).item()) if len(M_series) > 0 else 0.0
        results.M_mean = float(np.asarray(np.mean(M_series)).item()) if len(M_series) > 0 else 0.0
    except Exception as e:
        results.warnings.append(f"Error en Capa 1 (RQA): {str(e)}")
        M_series = np.zeros_like(taus_global)

    # === 4. Capa 2: Joint Episodes (MEJORADO) ===
    try:
        raw_episodes = extract_joint_episodes(taus_global, M_series, theta_A=theta_A, D_min=D_min)
        
        # Enriquecer cada episodio con las claves necesarias
        enriched_episodes = []
        for ep in raw_episodes:
            start = int(ep.get("start", 0))
            end = int(ep.get("end", start))

            # Calcular métricas si no existen
            duration = end - start + 1

            # Extraer segmento de M correspondiente al episodio
            m_segment = M_series[start : end + 1] if len(M_series) > end else np.array([0.0])

            mean_m = float(np.mean(m_segment)) if len(m_segment) > 0 else 0.0
            total_j = float(np.sum(m_segment)) if len(m_segment) > 0 else 0.0

            # Porcentaje de valores por encima del umbral (ej. 0.7)
            if len(m_segment) > 0:
                above_threshold = np.sum(np.abs(m_segment) > 0.7)
                j_above_07 = (above_threshold / len(m_segment)) * 100
            else:
                j_above_07 = 0.0

            enriched = {
                "start": start,
                "end": end,
                "duration": duration,
                "mean_m": round(mean_m, 4),
                "total_j": round(total_j, 4),
                "j_above_07": round(j_above_07, 2),
            }
            enriched_episodes.append(enriched)

        results.episodes = enriched_episodes
    except Exception as e:
        results.warnings.append(f"Error en Capa 2 (Joint Episodes): {str(e)}")
        results.episodes = []

    # === 5. Capa 3: Detección de Reorganización ===
    try:
        t_frob = None
        frob_max = 0.0
        if taus_per_module is not None and taus_per_module.ndim > 1:
            t_frob, frob_max = detect_reorganization_frob(taus_per_module)
        
        t_ks, ks_max = detect_reorganization_ks(dtk_series)
        t_star = consensus_transition(t_frob, t_ks)

        results.t_frob = t_frob
        results.t_ks = t_ks
        results.t_star = t_star
        results.frob_max = float(np.asarray(frob_max).item())
        results.ks_max = float(np.asarray(ks_max).item())
    except Exception as e:
        results.warnings.append(f"Error en detección de reorganización: {str(e)}")

    # === 6. Dimensión Fractal ===
    try:
        fractal_D = estimate_higuchi_dimension(dtk_series, k_max=20)
        results.fractal_D = float(np.asarray(fractal_D).item())

        if validate_fractal:
            low, high = expected_fractal_range
            if not (low <= fractal_D <= high):
                results.warnings.append(
                    f"Dimensión fractal atípica: {fractal_D:.3f} "
                    f"(rango esperado: {low} - {high})"
                )
    except Exception as e:
        results.warnings.append(f"Error calculando dimensión fractal: {str(e)}")

    # === Metadatos ===
    results.metadata = {
        "window_size": window_size,
        "n_components": X.shape[1],
        "T": X.shape[0],
        "n_episodes": len(results.episodes),
        "components": component_names,
    }

    if X.shape[1] < 3:
        results.warnings.append(
            f"Low Dimensionality: The matrix has only {X.shape[1]} components. "
            "The Systemic Tau paradigm is most robust for analyzing networks of N >= 3."
        )

    return results
