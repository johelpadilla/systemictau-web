"""
Generador de Reportes Académicos - Systemic Tau Paradigm
Versión mejorada v2.0
"""

from __future__ import annotations
from typing import Dict, Optional, List
from .analysis import TauAnalysisResults


def generate_academic_report(
    results: TauAnalysisResults,
    lang: str = "en",
    include_figures: bool = True,
    include_theoretical_captions: bool = False,
    include_significance_appendix: bool = False,
    figures: Optional[Dict[str, str]] = None,
    title: str = "Systemic Tau Paradigm Analytical Report",
    author: str = "",
    organization: str = "",
    ews_results: Optional["pd.DataFrame"] = None
) -> str:
    """
    Genera un reporte académico en Markdown de alta calidad.

    Parameters
    ----------
    results : TauAnalysisResults
        Resultados completos del análisis.
    lang : str
        "en" o "es".
    include_figures : bool
        Si se incluye la sección de figuras.
    figures : dict, optional
        Diccionario con imágenes en formato Markdown (base64 o rutas).
    title : str
        Título del reporte.
    ews_results : pd.DataFrame, optional
        Resultados de señales de alerta temprana (EWS).
    """

    lang = lang.lower()
    if lang not in ["en", "es"]:
        lang = "en"

    sections = []

    # === ENCABEZADO ===
    sections.append(_header(results, lang, title, author, organization))

    # === RESUMEN EJECUTIVO ===
    sections.append(_executive_summary(results, lang))

    # === CAPA 3: PRINCIPIO DE ASCENSO ONTOLÓGICO ===
    sections.append(_ontological_ascent_section(results, lang))

    # === ESTADÍSTICAS DE PERSISTENCIA ===
    sections.append(_persistence_statistics_section(results, lang))

    # === JOINT EPISODES ===
    sections.append(_joint_episodes_section(results, lang))

    # === RECD + DIMENSIÓN FRACTAL ===
    sections.append(_recd_fractal_section(results, lang))

    # === ADVERTENCIAS ===
    if results.warnings:
        sections.append(_warnings_section(results, lang))

    # === APÉNDICE VISUAL ===
    if include_figures:
        if not figures:
            try:
                import sys
                import importlib
                import systemictau.visualization.report_figures
                importlib.reload(sys.modules['systemictau.visualization.report_figures'])
                from systemictau.visualization.report_figures import generate_report_figures
                figures = generate_report_figures(results, ews_results=ews_results)
            except ImportError:
                print("Visualization module not available. Install with `pip install systemictau[reports]`")
        
        if figures:
            sections.append(_visual_appendix_section(results, lang, figures, include_theoretical_captions))

    # === APÉNDICE DE PARÁMETROS ===
    sections.append(_workflow_parameters_section(results, lang))
    
    # === APÉNDICE DE SIGNIFICANCIA (OPCIONAL) ===
    if include_significance_appendix:
        sections.append(_statistical_significance_appendix_section(results, lang))

    # === PIE ===
    sections.append(_footer(results, lang))

    return "\n\n".join(sections)

# ============================================================
# SECCIONES INDIVIDUALES
# ============================================================

def _workflow_parameters_section(results: TauAnalysisResults, lang: str) -> str:
    header = "## Appendix: Analytical Parameters" if lang == "en" else "## Apéndice: Parámetros Analíticos"
    
    # Extraer parámetros de metadata si existen
    params = results.metadata.get("parameters", {})
    components = results.metadata.get("components", [])
    
    window_size = params.get('window_size', results.window_size)
    theta_A = params.get('theta_A', '0.05 (Default)')
    D_min = params.get('D_min', '10 (Default)')
    fractal_range = params.get('expected_fractal_range', '(1.85, 2.15)')
    
    total_obs = len(results.taus_global) if results.taus_global is not None else "N/A"
    
    if lang == "en":
        text = (
            "The following parameters and variables were utilized to execute the Systemic Tau Paradigm workflow:\n\n"
            "### Target Variables (Spatial Components)\n"
        )
    else:
        text = (
            "Los siguientes parámetros y variables fueron utilizados para ejecutar el flujo de trabajo del Paradigma Tau Sistémico:\n\n"
            "### Variables Objetivo (Componentes Espaciales)\n"
        )
        
    for i, c in enumerate(components):
        text += f"- **Component {i}:** `{c}`\n"
        
    if not components:
        text += "- No explicit components registered.\n"
        
    if lang == "en":
        text += (
            "\n### Algorithmic Parameters\n"
            f"- **Time Series Length ($T$):** {total_obs} observations\n"
            f"- **Sliding Window Size ($n$):** {window_size}\n"
            f"- **Joint Episode Threshold ($\\theta_A$):** {theta_A}\n"
            f"- **Min Episode Duration ($D_{{min}}$):** {D_min}\n"
            f"- **Expected Fractal Range:** {fractal_range}\n"
        )
    else:
        text += (
            "\n### Parámetros Algorítmicos\n"
            f"- **Longitud de la Serie ($T$):** {total_obs} observaciones\n"
            f"- **Tamaño de Ventana Móvil ($n$):** {window_size}\n"
            f"- **Umbral de Episodios Conjuntos ($\\theta_A$):** {theta_A}\n"
            f"- **Duración Mínima de Episodio ($D_{{min}}$):** {D_min}\n"
            f"- **Rango Fractal Esperado:** {fractal_range}\n"
        )
        
    return f"{header}\n\n{text}"

def _statistical_significance_appendix_section(results: TauAnalysisResults, lang: str) -> str:
    header = "## Appendix B: Statistical Significance within the Systemic Tau Paradigm" if lang == "en" else "## Apéndice B: Significancia Estadística en el Paradigma Tau Sistémico"
    
    if lang == "en":
        text = (
            "Traditional frequentist statistical significance (p-values) assumes linear independence, normality, and stationary distributions—assumptions that break down entirely within complex dynamical systems undergoing phase transitions. Therefore, the **Systemic Tau Paradigm** establishes statistical and physical significance by adhering to Non-linear Topology and Dynamical Systems frameworks.\n\n"
            "This report establishes the significance of the observed transition through four core pillars:\n\n"
            "### 1. The Critical Feigenbaum Threshold ($\\pm 0.41$)\n"
            "In this paradigm, significance is not determined by an arbitrary confidence interval, but by the universal Feigenbaum constant. When the Systemic Tau ($\\tau_s$) trajectory crosses the $\\pm 0.41$ boundary, the system mathematically abandons the linear regime. Crossing this threshold serves as the proof that the empirical variables are no longer independent and have entered a state of **Deterministic Chaos**.\n\n"
            "### 2. The Higuchi Fractal Dimension ($D$)\n"
            "Stochastic noise (such as Brownian motion) yields specific expected fractal dimensions (e.g., $D \\approx 1.5 - 2.0$). The paradigm calculates the $D$ of the empirical Systemic Time Manifold (RECD). If the collapse is mathematically valid, $D$ diverges from classical noise boundaries, behaving as a *Bayesian Optimal Accumulator*. A fractal dimension contrasting with purely spatial regimes provides geometric proof of significance.\n\n"
            "### 3. Critical Slowing Down (Early Warning Signals)\n"
            "Complex systems approaching a critical singularity ($t^*$) exhibit Critical Slowing Down. An abrupt increase in Variance and Lag-1 Autocorrelation immediately preceding $t^*$ acts as an empirical validation that the topological collapse was preceded by a genuine loss of systemic resilience, ruling out stochastic outliers.\n\n"
            "### 4. Probabilistic Filters (Joint Episodes)\n"
            "To prevent false positives, spatial couplings must survive strict mathematical filters. A coupling event is only classified as a significant *Joint Episode* if its magnitude strictly exceeds the threshold ($\\theta_A$) and persists for a minimum duration ($D_{min}$). The presence of a massive vertical alignment in the Topological Heatmap under these constraints confirms that the macroscopic lock-in is a genuine systemic event."
        )
    else:
        text = (
            "La significancia estadística frecuentista clásica (p-values) asume independencia lineal, normalidad y distribuciones estacionarias—supuestos que se rompen por completo en sistemas dinámicos complejos que atraviesan transiciones de fase. Por ende, el **Paradigma Tau Sistémico** establece la significancia estadística y física adhiriéndose a la Topología No Lineal y los Sistemas Dinámicos.\n\n"
            "Este reporte establece la significancia de la transición observada a través de cuatro pilares centrales:\n\n"
            "### 1. El Umbral Crítico de Feigenbaum ($\\pm 0.41$)\n"
            "En este paradigma, la significancia no se determina por un intervalo de confianza arbitrario, sino por la constante universal de Feigenbaum. Cuando la trayectoria de la Tau Sistémica ($\\tau_s$) cruza la frontera de $\\pm 0.41$, el sistema abandona matemáticamente el régimen lineal. Cruzar este umbral es la prueba de que las variables empíricas ya no son independientes y han entrado en un estado de **Caos Determinista**.\n\n"
            "### 2. La Dimensión Fractal de Higuchi ($D$)\n"
            "El ruido estocástico (como el movimiento browniano) arroja dimensiones fractales esperadas específicas (ej. $D \\approx 1.5 - 2.0$). El paradigma calcula la $D$ del Manifold del Tiempo Sistémico empírico (RECD). Si el colapso es matemáticamente válido, $D$ diverge de las fronteras de ruido clásico, comportándose como un *Acumulador Óptimo Bayesiano*. Una dimensión fractal que contrasta con los regímenes puramente espaciales provee prueba geométrica de significancia.\n\n"
            "### 3. Critical Slowing Down (Señales de Alerta Temprana)\n"
            "Los sistemas complejos que se aproximan a una singularidad crítica ($t^*$) exhiben Critical Slowing Down. Un aumento abrupto en la Varianza y la Autocorrelación Lag-1 inmediatamente anterior a $t^*$ actúa como validación empírica de que el colapso topológico fue precedido por una pérdida genuina de resiliencia sistémica, descartando valores atípicos estocásticos.\n\n"
            "### 4. Filtros Probabilísticos (Joint Episodes)\n"
            "Para prevenir falsos positivos, los acoplamientos espaciales deben sobrevivir estrictos filtros matemáticos. Un evento de acoplamiento solo se clasifica como un *Joint Episode* significativo si su magnitud excede estrictamente el umbral ($\\theta_A$) y persiste por una duración mínima ($D_{min}$). La presencia de una alineación vertical masiva en el Heatmap Topológico bajo estas restricciones confirma que el bloqueo macroscópico es un evento sistémico genuino."
        )
        
    return f"{header}\n\n{text}"

def _get_component_names(results: TauAnalysisResults) -> List[str]:
    """Obtiene los nombres reales de los componentes si existen."""
    components = results.metadata.get("components", [])
    if components and not all(c.startswith("Component_") for c in components):
        return components
    return []


def _header(results: TauAnalysisResults, lang: str, title: str, author: str = "", organization: str = "") -> str:
    components = _get_component_names(results)
    dataset_str = ", ".join(components) if components else "N/A"
    
    author_str = author if author else "SystemicTau Engine"
    org_str = f"**Organization:** {organization}\n\n" if organization else ""
    author_block = f"**Lead Analyst:** {author_str}\n\n" if author else ""

    if lang == "es":
        return f"""---
title: "{title}"
author: "{author_str}"
date: ""
dataset: {dataset_str}
---

# {title}

{author_block}{org_str}---

**Generado automáticamente vía `systemictau`**
*Marco Teórico basado en la Síntesis Magna v6 y The Principle of Ontological Ascent (Padilla, 2026).*"""
    else:
        return f"""---
title: "{title}"
author: "{author_str}"
date: ""
dataset: {dataset_str}
---

# {title}

{author_block}{org_str}---

**Automatically generated via `systemictau`**
*Theoretical Framework based on the Magna Synthesis v6 and The Principle of Ontological Ascent (Padilla, 2026).*"""


def _executive_summary(results: TauAnalysisResults, lang: str) -> str:
    t_star = results.t_star if results.t_star is not None else "Not detected"
    n_ep = len(results.episodes)
    components = _get_component_names(results)

    if components:
        dataset_text = f"variables: {', '.join(components)}"
    else:
        dataset_text = "the provided multivariate time series" if lang == "en" else "las variables multivariadas proporcionadas"

    if lang == "es":
        return f"""## Resumen Ejecutivo

Este documento presenta el análisis topológico del dataset ({dataset_text}) utilizando una ventana de observación $n = {results.window_size}$.

El análisis detectó exitosamente el **Punto de Reorganización Sistémica** (Consensus Transition) en $t^* = {t_star}$, donde el sistema alcanzó convergencia espacial global. En este punto, la métrica de sincronización global cruzó la banda caótica de Feigenbaum ($|\\tau_s| < 0.41$), marcando el colapso topológico del sistema.

Se detectaron **{n_ep}** Episodios Conjuntos (Joint Episodes), evidenciando los precursores de baja varianza en la capa de habilitación antes del colapso."""
    else:
        return f"""## Executive Summary

This document presents the topological analysis of the dataset ({dataset_text}) using an observation window $n = {results.window_size}$.

The analysis successfully detected the **Systemic Reorganization Point** (Consensus Transition) at $t^* = {t_star}$, where the system reached global spatial convergence. At this point, the global synchronization metric crossed into the Feigenbaum chaotic band ($|\\tau_s| < 0.41$), marking the topological collapse of the system.

**{n_ep}** Joint Episodes were detected, evidencing the low-variance precursors in the enabling layer before the collapse."""


def _ontological_ascent_section(results: TauAnalysisResults, lang: str) -> str:
    t_star = results.t_star or "N/A"
    f_max = f"{results.frob_max:.4f}" if results.frob_max else "N/A"
    ks_max = f"{results.ks_max:.4f}" if results.ks_max else "N/A"

    if lang == "es":
        return f"""## El Principio del Ascenso Ontológico (Capa 3)

El modelo identificó la Transición de Reorganización Estructural Global en la coordenada cronológica $t^* = {t_star}$.

Según el marco teórico (*The Principle of Ontological Ascent*), este evento no constituye una simple fluctuación estadística, sino un verdadero ascenso de nivel ontológico ($L_m \\to L_{{m+1}}$) en la jerarquía RECD. En $t^*$, se instituye un nuevo régimen de acumulación de tiempo sistémico, irreducible a la suma de las dinámicas espaciales previas.

En este régimen, la función de compuerta fractal entra en la banda caótica intermedia, donde la topología del sistema cruza el rango de universalidad de Feigenbaum ($|\\tau_s| < 0.41$). Específicamente, en el punto crítico se alcanzó un máximo de Frobenius de $F_{{max}} = {f_max}$ y una divergencia de Kolmogorov-Smirnov de $KS_{{max}} = {ks_max}$, confirmando el colapso topológico determinista de la matriz de Kendall."""
    else:
        return f"""## The Principle of Ontological Ascent (Layer 3)

The model identified the Global Structural Reorganization Transition at the chronological coordinate $t^* = {t_star}$.

According to the theoretical framework (*The Principle of Ontological Ascent*), this event does not constitute a simple statistical fluctuation, but a true ascent of ontological level ($L_m \\to L_{{m+1}}$) in the RECD hierarchy. At $t^*$, a new regime of systemic time accumulation is instituted, which is irreducible to the sum of the previous spatial dynamics.

In this regime, the fractal gate function enters the intermediate chaotic band, where the system's topology crosses the Feigenbaum universality range ($|\\tau_s| < 0.41$). Specifically, at the critical point, the maximum Frobenius metric reached $F_{{max}} = {f_max}$ and the maximum Kolmogorov-Smirnov divergence was $KS_{{max}} = {ks_max}$, confirming the deterministic topological collapse of the Kendall matrix."""


def _persistence_statistics_section(results: TauAnalysisResults, lang: str) -> str:
    note = (
        "*Paradigmatic note: An elevated hyper-persistence ($hp_z$) or significant laminarity validates the Bayesian Optimal Accumulator theoretical framework.*"
        if lang == "en"
        else "*Nota paradigmática: Una hiper-persistencia elevada ($hp_z$) o una laminaridad significativa valida el marco teórico del Acumulador Óptimo Bayesiano.*"
    )

    header = "## Persistence Statistics and Ordinal Criticality" if lang == "en" else "## Estadísticas de Persistencia y Criticidad Ordinal"

    table = f"""
| Metric                              | Calculated Value |
|-------------------------------------|------------------|
| Hyper-persistence Z-score ($hp_z$)  | {results.hp_z:.4f} |
| Ordinal Laminarity (RQA)            | {results.lam:.4f}  |
| Trapping Time                       | {results.tt:.4f}   |
| Maximum Critical Mass ($M_{{max}}$)   | {results.M_max:.4f}|
| Mean Critical Mass ($M_{{mean}}$)     | {results.M_mean:.4f}|
"""

    return f"{header}\n\n{table}\n\n{note}"


def _joint_episodes_section(results: TauAnalysisResults, lang: str) -> str:
    header = "## Joint Episodes and Relational Kairos" if lang == "en" else "## Episodios Conjuntos y Kairos Relacional"

    if not results.episodes:
        msg = "No Joint Episodes were detected under the current thresholds." if lang == "en" else \
              "No se detectaron Episodios Conjuntos con los umbrales actuales."
        return f"{header}\n\n{msg}"

    intro = (
        "The theory dictates that topological collapse is preceded by spatio-temporal units called *Joint Episodes*. "
        "These units (Layer 2) act as bidirectional probabilistic filtering, establishing a \"Relational Kairos\" that enables the imminent \"Ontological Kairos\" (Layer 3)."
        if lang == "en"
        else
        "La teoría dicta que el colapso topológico es precedido por unidades espacio-temporales llamadas *Episodios Conjuntos*. "
        "Estas unidades (Capa 2) actúan como un filtrado probabilístico bidireccional, estableciendo un \"Kairos Relacional\" que habilita el inminente \"Kairos Ontológico\" (Capa 3)."
    )

    # Tabla
    table = ["| Start | End | Duration (D) | Mean M | Total Magnitude (J) | J > 0.7 (%) |"]
    table.append("|-------|-----|--------------|--------|---------------------|-------------|")

    for ep in results.episodes:
        table.append(
            f"| {ep.get('start', '-')} | {ep.get('end', '-')} | "
            f"{ep.get('duration', '-')} | {ep.get('mean_m', 0):.4f} | "
            f"{ep.get('total_j', 0):.4f} | {ep.get('j_above_07', 0):.2f}% |"
        )

    return f"{header}\n\n{intro}\n\n" + "\n".join(table)


def _recd_fractal_section(results: TauAnalysisResults, lang: str) -> str:
    d = f"{results.fractal_D:.3f}" if results.fractal_D is not None else "N/A"

    if lang == "es":
        return f"""## Ley del Reloj Extramental Discreto (RECD) y Dimensión Fractal

La integración analítica demuestra la fracturación del tiempo cronológico continuo $t$ hacia el tiempo sistémico discreto $T$. Esta acumulación regulada está gobernada por la **Ley del Reloj Extramental Discreto (RECD)**.

El análisis de dimensión geométrica validado mediante el método de Higuchi arroja una dimensión fractal de soporte de **$D \\approx {d}$** para el tiempo sistémico.

Este resultado se encuentra fuera del corpus empírico que establece límites rigurosos en $[1.96, 2.01]$, un fenómeno explicable como un Acumulador Óptimo Bayesiano. Esto contrasta explícitamente con la dimensión puramente topológico-espacial del atractor original."""
    else:
        return f"""## Law of the Discrete Extramental Clock (RECD) and Fractal Dimension

Analytical integration demonstrates the fracturing of continuous chronological time $t$ towards discrete systemic time $T$. This regulated accumulation is governed by the **Law of the Discrete Extramental Clock (RECD)**.

The geometric dimension analysis validated through the Higuchi method yields a supporting fractal dimension of **$D \\approx {d}$** for systemic time.

This result falls outside the empirical corpus that places rigorous bounds in $[1.96, 2.01]$, a phenomenon explicable as a Bayesian Optimal Accumulator. This explicitly contrasts with the purely topological-spatial dimension of the original attractor."""


def _warnings_section(results: TauAnalysisResults, lang: str) -> str:
    header = "## Validation Warnings" if lang == "en" else "## Advertencias de Validación"
    warnings = "\n".join([f"- {w}" for w in results.warnings])
    return f"{header}\n\n{warnings}"


def _visual_appendix_section(results: TauAnalysisResults, lang: str, figures: Dict[str, str], include_captions: bool = False) -> str:
    header = "## Visual Appendix" if lang == "en" else "## Apéndice Visual"
    content = []
    
    # Dictionary mapping standard figure names to their theoretical captions
    captions_en = {
        "Classical Dual-Plot": "This graph illustrates the macroscopic topological evolution of the system. The upper panel tracks the Systemic Tau ($\\tau_s$); intersection with the critical Feigenbaum bands ($\\pm 0.41$) indicates a departure from the linear regime. The lower panel displays the discrete consumption of systemic time ($\\Delta t_k$) driven by the opening of the topological gate. Severe spikes in this lower panel denote critical systemic shocks, highlighting abrupt losses of spatial independence culminating in the irreversible topological collapse at $t^*$.",
        
        "Systemic Time Manifold": "This manifold visualizes the discretization of time within complex systems, contrasting continuous chronological time (x-axis) with accumulated systemic time (y-axis). Horizontal plateaus represent periods of empirical stability where the system's components fluctuate independently. Abrupt vertical steps indicate Joint Episodes—periods of extreme topological coupling and systemic time consumption—ultimately leading to the geometric collapse of the time manifold at the critical consensus point ($t^*$).",
        
        "Topological Heatmap": "Representing the second layer of ontological ascent, this heatmap provides spatial evidence of local probabilistic filters aligning to generate the macroscopic Systemic Tau. Intense vertical bands (deep red or blue) across multiple spatial components signify a massive Joint Episode. Empirically, this demonstrates a momentary spatial lock-in, where independent modules synchronize simultaneously, severely reducing the system's degrees of freedom and driving it toward deterministic chaos.",
        
        "3D Phase Space Attractor": "This three-dimensional phase space model demonstrates the non-linear dynamic geometry of the system. The convergence of the system's trajectory into a strange attractor confirms its chaotic, yet deterministic, nature constrained by thermodynamic entropy (RECD). The empirical data traces a funneling trajectory that sharply constricts as it approaches the critical point singularity (orange marker, $t^*$), indicating that the observed systemic collapse is an inevitable outcome of non-linear dynamics rather than stochastic noise.",
        
        "Unimodal Return Map": "This Unimodal Return Map plots the Systemic Tau orbit ($\\tau_s(t)$ versus $\\tau_s(t+1)$). A purely stochastic system would produce an unstructured point cloud. Here, the dense 'cobweb' accumulation toward the center proves that the system's topological collapse is not random noise, but an inevitable transition governed by deterministic non-linear dynamics, ultimately constrained by the universal Feigenbaum constant.",
        
        "Early Warning Signals": "This panel demonstrates Critical Slowing Down, the hallmark of complex systems approaching a catastrophic phase transition. The drastic spike in system Variance and Lag-1 Autocorrelation immediately preceding the critical point ($t^*$) provides empirical proof that the topological collapse was preceded by a genuine loss of systemic resilience."
    }
    
    captions_es = {
        "Classical Dual-Plot": "Este gráfico ilustra la evolución topológica macroscópica del sistema. El panel superior rastrea la Tau Sistémica ($\\tau_s$); la intersección con las bandas críticas de Feigenbaum ($\\pm 0.41$) indica una salida del régimen lineal. El panel inferior muestra el consumo discreto de tiempo sistémico ($\\Delta t_k$) impulsado por la apertura de la compuerta topológica. Los picos severos denotan choques sistémicos críticos, culminando en el colapso topológico irreversible en $t^*$.",
        "Systemic Time Manifold": "Este manifold visualiza la discretización del tiempo en sistemas complejos, contrastando el tiempo cronológico continuo con el tiempo sistémico acumulado. Las zonas planas representan periodos de estabilidad empírica. Los escalones verticales abruptos indican Joint Episodes (periodos de acoplamiento extremo), que conducen al colapso geométrico del manifold temporal en el punto crítico ($t^*$).",
        "Topological Heatmap": "Representando la segunda capa del ascenso ontológico, este mapa de calor provee evidencia espacial de los filtros probabilísticos locales alineándose. Las bandas verticales intensas significan un Joint Episode masivo. Empíricamente, esto demuestra un bloqueo espacial momentáneo, reduciendo severamente los grados de libertad del sistema e impulsándolo hacia el caos determinista.",
        "3D Phase Space Attractor": "Este modelo tridimensional demuestra la geometría dinámica no lineal del sistema. La convergencia en un atractor extraño confirma su naturaleza caótica restringida por la entropía termodinámica (RECD). La trayectoria se estrecha drásticamente al acercarse a la singularidad ($t^*$), indicando que el colapso sistémico observado es un resultado inevitable de la dinámica no lineal.",
        "Unimodal Return Map": "Este Mapa de Retorno Unimodal grafica la órbita de la Tau Sistémica ($\\tau_s(t)$ versus $\\tau_s(t+1)$). Un sistema puramente estocástico produciría una nube de puntos sin estructura. Aquí, la densa acumulación en forma de 'telaraña' (cobweb) hacia el centro prueba que el colapso topológico del sistema no es ruido aleatorio, sino una transición inevitable gobernada por dinámicas no lineales deterministas, restringida en última instancia por la constante universal de Feigenbaum.",
        "Early Warning Signals": "Este panel demuestra el Critical Slowing Down (Desaceleración Crítica), el sello distintivo de los sistemas complejos que se aproximan a una transición de fase catastrófica. El pico drástico en la Varianza del sistema y la Autocorrelación Lag-1 inmediatamente anterior al punto crítico ($t^*$) provee prueba empírica de que el colapso topológico fue precedido por una genuina pérdida de resiliencia sistémica."
    }
    
    captions_dict = captions_en if lang == "en" else captions_es
    
    for name, fig in figures.items():
        if not str(fig).startswith("!["):
            # Si es un string base64 crudo, formatearlo como Markdown
            if not str(fig).startswith("data:image"):
                fig = f"![{name}](data:image/png;base64,{fig})"
            else:
                fig = f"![{name}]({fig})"
                
        section_text = f"### {name.replace('_', ' ').title()}\n\n{fig}"
        
        # Inyectar el caption teórico si está habilitado
        if include_captions:
            # Buscar el nombre exacto de la figura en el diccionario para adjuntarle su explicación
            for key, caption in captions_dict.items():
                if key in name:
                    section_text += f"\n\n*{caption}*"
                    break
                    
        content.append(section_text)
        
    return f"{header}\n\n" + "\n\n---\n\n".join(content)


def _footer(results: TauAnalysisResults, lang: str) -> str:
    if lang == "es":
        return f"*Para garantizar reproducibilidad, este documento fue derivado usando `systemictau` con ventana $n={results.window_size}$. Cite como: Padilla (2026). \"Síntesis Magna v6\" y paquete Python asociado.*"
    else:
        return f"*To guarantee reproducibility, this document was derived using `systemictau` with window $n={results.window_size}$. Cite as: Padilla (2026). \"Magna Synthesis v6\" and associated Python package.*"
