"""
Módulo de Análisis de Sensibilidad y Robustez.
Verifica la estabilidad matemática del ascenso ontológico frente a perturbaciones.
"""

import numpy as np
from typing import Dict, List, Any
from .analysis import run_full_analysis

def inject_gaussian_noise(X: np.ndarray, noise_level: float = 0.05, seed: int = 42) -> np.ndarray:
    """
    Inyecta ruido Gaussiano a la serie temporal multivariada.
    El nivel de ruido se define como una fracción de la desviación estándar de cada canal.
    
    Parameters
    ----------
    X : np.ndarray
        Matriz original (T, N)
    noise_level : float
        Nivel de ruido (0.05 = 5% de la desviación estándar)
    seed : int
        Semilla para reproducibilidad
        
    Returns
    -------
    np.ndarray
        Matriz perturbada
    """
    np.random.seed(seed)
    X_noisy = np.copy(X)
    
    for i in range(X.shape[1]):
        std_dev = np.nanstd(X[:, i])
        if std_dev > 0:
            noise = np.random.normal(0, std_dev * noise_level, size=X.shape[0])
            X_noisy[:, i] += noise
            
    return X_noisy

def run_sensitivity_analysis(
    X: np.ndarray, 
    window_size: int = 13, 
    theta_A: float = 0.05,
    D_min: int = 10,
    noise_levels: List[float] = [0.05, 0.10, 0.15],
    iterations_per_level: int = 3,
    component_names: List[str] = None
) -> Dict[str, Any]:
    """
    Ejecuta un barrido de sensibilidad sobre el sistema, inyectando ruido
    para probar la estabilidad matemática del punto de transición crítica (t*) y la dimensión fractal.
    
    Returns
    -------
    Dict[str, Any]
        Reporte estructurado con la varianza de los indicadores topológicos.
    """
    # 1. Base (Ground Truth)
    base_res = run_full_analysis(X, window_size=window_size, theta_A=theta_A, D_min=D_min, component_names=component_names)
    base_t_star = base_res.t_star
    base_fractal_D = base_res.fractal_D
    
    report = {
        "baseline": {
            "t_star": base_t_star,
            "fractal_D": base_fractal_D,
            "episodes_count": len(base_res.episodes)
        },
        "perturbations": {}
    }
    
    # 2. Perturbation Sweeps
    for level in noise_levels:
        level_results = []
        for i in range(iterations_per_level):
            X_noisy = inject_gaussian_noise(X, noise_level=level, seed=42 + i)
            res = run_full_analysis(
                X_noisy, 
                window_size=window_size, 
                theta_A=theta_A, 
                D_min=D_min, 
                component_names=component_names,
                validate_fractal=False # Do not spam warnings during sweep
            )
            
            level_results.append({
                "t_star": res.t_star,
                "fractal_D": res.fractal_D,
                "episodes_count": len(res.episodes)
            })
            
        # Aggregate stats for this noise level
        t_stars = [r["t_star"] for r in level_results if r["t_star"] is not None]
        fractal_Ds = [r["fractal_D"] for r in level_results if r["fractal_D"] is not None]
        
        report["perturbations"][f"noise_{int(level*100)}pct"] = {
            "t_star_mean": float(np.mean(t_stars)) if t_stars else None,
            "t_star_std": float(np.std(t_stars)) if t_stars else None,
            "t_star_stability_pct": 100.0 - (float(np.std(t_stars)) / base_t_star * 100 if base_t_star and t_stars else 0.0),
            "fractal_D_mean": float(np.mean(fractal_Ds)) if fractal_Ds else None,
            "fractal_D_std": float(np.std(fractal_Ds)) if fractal_Ds else None,
            "raw_results": level_results
        }
        
    return report
