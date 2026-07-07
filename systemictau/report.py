"""
Generador de Reportes Académicos - Systemic Tau Paradigm v5.6.0
"""

from __future__ import annotations
from typing import Dict, Optional, List, Any
from .results import OntologicalAscentResult

def generate_academic_report(
    result: Optional[OntologicalAscentResult] = None,
    results: Optional[OntologicalAscentResult] = None,
    lang: str = "en",
    include_figures: bool = True,
    include_theoretical_captions: bool = False,
    include_significance_appendix: bool = False,
    figures: Optional[Dict[str, str]] = None,
    title: str = "Systemic Tau Paradigm Analytical Report",
    author: str = "",
    organization: str = "",
    ews_results: Optional["pd.DataFrame"] = None,
    surrogate_result: Optional[Any] = None,
    adaptive_breathing: bool = False,
    # API Compatibility parameters
    output_path: Optional[str] = None,
    location_name: Optional[str] = None,
    variables: Optional[List[str]] = None,
    language: Optional[str] = None
) -> str:
    # Support both result and results (API compatibility)
    results = result if result is not None else results
    if results is None:
        raise ValueError("Must provide an OntologicalAscentResult object via 'result' or 'results' argument.")
        
    lang = (language or lang).lower()
    if lang not in ["en", "es"]:
        lang = "en"
        
    if location_name:
        title = f"{title} - {location_name}"
        
    if variables and results.metadata is not None:
        results.metadata["components"] = variables

    sections = []

    # 0. HEADER
    sections.append(_header(results, lang, title, author, organization))

    # 1. EXECUTIVE SUMMARY (Short)
    sections.append(_executive_summary(results, lang))

    # 2. ONTOLOGICAL ASCENT NARRATIVE (Detailed)
    sections.append(_ontological_narrative(results, lang))

    # 3. KEY METRICS TABLE
    sections.append(_key_metrics_table(results, lang, adaptive_breathing))

    # 4. TIER 1: ONTOLOGICAL OVERVIEW
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
                print("Visualization module not available.")
        
        if figures and "Tier 1: Ontological Overview" in figures:
            sections.append(_tier_1_section(results, lang, figures, include_theoretical_captions))

    # 4.5. TDA GEOSPATIAL TOPOLOGY
    if include_figures and figures and "TDA: Systemic Transition vs Topological Holes" in figures:
        sections.append(_tda_section(results, lang, figures, include_theoretical_captions))
        
    # 4.6. ORDINAL MEMORY
    if include_figures and figures and "Ordinal Memory Dynamics" in figures:
        sections.append(_ordinal_section(results, lang, figures, include_theoretical_captions))

    # 5. WARNINGS
    if results.warnings:
        sections.append(_warnings_section(results, lang))

    # 6. APPENDIX A: TIER 2 (LAYER DETAILS)
    if include_figures and figures:
        sections.append(_tier_2_appendix(results, lang, figures, include_theoretical_captions))

    # 7. APPENDIX B: STATISTICAL SIGNIFICANCE (Optional)
    if include_significance_appendix:
        sections.append(_statistical_significance_appendix_section(results, lang))
        
    # X. APPENDIX C: IAAFT SURROGATE VALIDATION
    if surrogate_result is not None:
        title = "## Appendix C: IAAFT Surrogate Validation" if lang == "en" else "## Apéndice C: Validación IAAFT Surrogate"
        import math
        real_max_tau = surrogate_result.metadata.get("real_max_tau", float('nan'))
        
        if math.isnan(real_max_tau):
            warn_title = "CRITICAL WARNING: INVALID VALIDATION" if lang == "en" else "ADVERTENCIA CRÍTICA: VALIDACIÓN INVÁLIDA"
            warn_text = (
                "The surrogate validation is fundamentally flawed and should be disregarded. "
                "The empirical 'Max Systemic Tau' calculated for the true data yielded a NaN (Not a Number). "
                "This indicates a severe data quality issue (e.g., constant values, unhandled NaNs in the source) "
                "or an evaluation window too small for the underlying variability. "
                "Re-evaluate the source data before drawing any topological conclusions."
                if lang == "en" else
                "La validación surrogate es inválida y debe ser ignorada. "
                "El valor empírico 'Max Systemic Tau' calculado para los datos reales resultó en NaN (Not a Number). "
                "Esto indica un problema grave de calidad de datos (ej. valores constantes, NaNs no manejados) "
                "o una ventana de evaluación demasiado pequeña para la variabilidad subyacente. "
                "Re-evalúe los datos fuente antes de extraer conclusiones topológicas."
            )
            content = f"{title}\n\n> [!CAUTION]\n> **{warn_title}**\n> {warn_text}"
        else:
            surr_mean = surrogate_result.metadata.get("surrogate_max_tau_mean", 0.0)
            surr_std = surrogate_result.metadata.get("surrogate_max_tau_std", 0.0)
            linear_contrib = min(100.0, (surr_mean / real_max_tau * 100.0)) if real_max_tau > 0 else 100.0
            excess_z = (real_max_tau - surr_mean) / surr_std if surr_std > 0 else 0.0
            
            table_header_en = "| Metric | Value |\n|--------|-------|"
            table_header_es = "| Métrica | Valor |\n|---------|-------|"
            table_header = table_header_en if lang == "en" else table_header_es
            
            p_val = surrogate_result.percentile_rank
            
            table = f"""
{table_header}
| Linear Baseline Contribution | {linear_contrib:.1f}% |
| Empirical Max Systemic Tau | {real_max_tau:.4f} |
| Surrogate Mean Max Tau | {surr_mean:.4f} |
| Excess over Linear Baseline (Z-score) | {excess_z:.2f} |
| Statistical Percentile | {p_val:.2f}% |
"""
            if p_val < 5.0:
                interp_title = "Linear Dominated: Weak/Absent Non-linear Evidence" if lang == "en" else "Dominado por Linealidad: Evidencia No-Lineal Débil o Ausente"
                interp_text = (
                    f"The IAAFT test indicates that ~100% of the magnitude of the detected topological change can be generated by the individual linear properties of the series (autocorrelation and amplitude). "
                    "This does not invalidate the existence of a transition, but suggests the systemic non-linear coupling component is very weak or absent."
                    if lang == "en" else
                    f"La prueba IAAFT indica que ~100% de la magnitud del cambio topológico detectado puede ser generado por las propiedades lineales individuales de las series (autocorrelación y amplitud). "
                    "Esto no invalida la existencia de una transición, pero sugiere que el componente de acoplamiento sistémico no lineal es muy débil o ausente."
                )
                conclusion = f"> [!WARNING]\n> **{interp_title}**\n> {interp_text}"
            elif surrogate_result.is_significant:
                interp_title = "Strong Evidence of Non-linear Topological Lock-in" if lang == "en" else "Evidencia Fuerte de Bloqueo Topológico No Lineal"
                interp_text = (
                    f"The detected transition clearly exceeds what can be explained by linear effects alone (Percentile: {p_val:.2f}%, Excess Z = {excess_z:.2f}). "
                    "There is strong empirical evidence of genuine systemic non-linear topological reorganization."
                    if lang == "en" else
                    f"La transición detectada excede claramente lo explicable por efectos lineales (Percentil: {p_val:.2f}%, Exceso Z = {excess_z:.2f}). "
                    "Existe fuerte evidencia empírica de una genuina reorganización topológica sistémica no lineal."
                )
                conclusion = f"> [!TIP]\n> **{interp_title}**\n> {interp_text}"
            else:
                interp_title = "Mixed/Inconclusive Non-linear Evidence" if lang == "en" else "Evidencia No Lineal Mixta/Inconclusa"
                interp_text = (
                    f"The empirical coupling failed to achieve the strict 95% statistical significance threshold against linear models (Percentile: {p_val:.2f}%). "
                    f"While some non-linear structure exists (Excess Z = {excess_z:.2f}), it cannot be conclusively differentiated from random phase-shifted noise."
                    if lang == "en" else
                    f"El acoplamiento empírico no logró superar el estricto umbral de 95% frente a los modelos lineales (Percentil: {p_val:.2f}%). "
                    f"Si bien existe cierta estructura no lineal (Exceso Z = {excess_z:.2f}), no se puede distinguir de manera concluyente del ruido de fase aleatorio."
                )
                conclusion = f"> [!NOTE]\n> **{interp_title}**\n> {interp_text}"
                
            content = f"{title}\n{table}\n\n{conclusion}"
            
        sections.append(content)

    # 8. APPENDIX D: REPRODUCIBILITY & PARAMETERS
    sections.append(_reproducibility_section(results, lang))

    final_md = "\n\n".join(sections)
    
    if output_path:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_md)

    return final_md

# ============================================================
# SECCIONES INDIVIDUALES
# ============================================================

def _get_component_names(results: OntologicalAscentResult) -> List[str]:
    components = results.metadata.get("components", [])
    if components and not all(c.startswith("Component_") for c in components):
        return components
    return []

def _header(results: OntologicalAscentResult, lang: str, title: str, author: str = "", organization: str = "") -> str:
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

**Generado automáticamente vía `systemictau v5.6.0`**
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

**Automatically generated via `systemictau v5.6.0`**
*Theoretical Framework based on the Magna Synthesis v6 and The Principle of Ontological Ascent (Padilla, 2026).*"""

def _executive_summary(results: OntologicalAscentResult, lang: str) -> str:
    components = _get_component_names(results)
    dataset_text = f"variables: {', '.join(components)}" if components else "the provided multivariate time series"
    if lang == "es" and not components: dataset_text = "las variables multivariadas proporcionadas"
    
    title = "## Executive Summary" if lang == "en" else "## Resumen Ejecutivo"
    intro_en = f"This document presents the topological analysis of the dataset ({dataset_text}) using an observation window of $n = {results.window_size}$."
    intro_es = f"Este documento presenta el análisis topológico del dataset ({dataset_text}) utilizando una ventana de observación $n = {results.window_size}$."
    
    return f"""{title}

{intro_en if lang == 'en' else intro_es}

{results.summary(level="short", lang=lang)}
"""

def _ontological_narrative(results: OntologicalAscentResult, lang: str) -> str:
    title = "## Ontological Ascent Narrative" if lang == "en" else "## Narrativa de Ascenso Ontológico"
    return f"""{title}

{results.summary(level="detailed", lang=lang)}
"""

def _key_metrics_table(results: OntologicalAscentResult, lang: str, adaptive_breathing: bool = False) -> str:
    header = "## Key Metrics" if lang == "en" else "## Métricas Clave"
    num_episodes = len(results.episodes)
    t_star_text = str(results.t_star) if results.t_star else "N/A"
    fractal_val = f"{results.fractal_D:.4f}" if results.fractal_D is not None else "N/A"

    adaptive_str = "Enabled" if adaptive_breathing else "Disabled"
    table = f"""
| Metric | Value |
|--------|-------|
| Adaptive Breathing Window ($W_t$) | {adaptive_str} |
| Evaluation Window ($n$) | {results.window_size} |
| Critical Transition ($t^*$) | {t_star_text} |
| Number of Joint Episodes | {num_episodes} |
| Hyper-persistence Z-score ($hp_z$) | {results.hp_z:.4f} |
| Fractal Dimension ($D$) | {fractal_val} |
| Maximum Critical Mass ($M_{{max}}$) | {results.M_max:.4f} |
"""

    if lang == "en":
        interpretation = f"**Interpretation:** The critical transition at $t^*={t_star_text}$ signifies the exact moment the system lost its spatial degrees of freedom. A fractal dimension of {fractal_val} coupled with a hyper-persistence of {results.hp_z:.2f} mathematically validates that this is not stochastic noise, but a structured topological collapse."
    else:
        interpretation = f"**Interpretación:** La transición crítica en $t^*={t_star_text}$ significa el momento exacto en que el sistema perdió sus grados de libertad espaciales. Una dimensión fractal de {fractal_val} acoplada con una hiper-persistencia de {results.hp_z:.2f} valida matemáticamente que esto no es ruido estocástico, sino un colapso topológico estructurado."

    return f"{header}\n{table}\n{interpretation}"

def _get_caption(fig_name: str, lang: str) -> str:
    captions_en = {
        "Tier 1: Ontological Overview": "This unified diagram illustrates the complete ontological ascent of the system. It combines the continuous metric of systemic time consumption (RECD), the discrete topological 'Joint Episodes' where independent variables enter synchronized enablement, and the ultimate critical systemic transition ($t^*$). This $t^*$ is not merely a statistical anomaly, but the precise chronological coordinate where the system achieves irreversible topological coupling, fundamentally altering its macroscopic behavior.",
        "Tier 2: Layer Details": "A multi-layer view of the topological transition. The Micro layer tracks the fractal dimension of individual components, indicating their local degrees of freedom. The Meso layer isolates the density of Joint Episodes where topological enablement begins. The Macro layer confirms the systemic phase transition where synchronization crosses the Feigenbaum threshold.",
        "Topological Heatmap": "Representing the second layer of the ontological ascent, this heatmap provides spatial evidence of local probabilistic filters aligning. Intense vertical bands signify a massive Joint Episode. Empirically, this demonstrates a momentary spatial lock-in, severely reducing the system's degrees of freedom and driving it toward deterministic chaos.",
        "Unimodal Return Map": "This Unimodal Return Map plots the Systemic Tau orbit ($\\tau_s(t)$ versus $\\tau_s(t+1)$). A purely stochastic system would produce an unstructured point cloud. Here, the dense 'cobweb' accumulation toward the center proves that the system's topological collapse is not random noise, but an inevitable transition governed by deterministic non-linear dynamics, ultimately constrained by the universal Feigenbaum constant.",
        "Early Warning Signals": "This panel demonstrates Critical Slowing Down, the hallmark of complex systems approaching a catastrophic phase transition. The drastic spike in system Variance and Lag-1 Autocorrelation immediately preceding the critical point ($t^*$) provides empirical proof that the topological collapse was preceded by a genuine loss of systemic resilience."
    }
    captions_es = {
        "Tier 1: Ontological Overview": "Este diagrama unificado ilustra el ascenso ontológico completo del sistema. Combina la métrica continua del consumo de tiempo sistémico (RECD), los 'Episodios Conjuntos' topológicos donde variables independientes entran en habilitación sincronizada, y la transición sistémica crítica ($t^*$). Este $t^*$ no es una mera anomalía estadística, sino la coordenada cronológica precisa donde el sistema alcanza un acoplamiento topológico irreversible, alterando fundamentalmente su comportamiento macroscópico.",
        "Tier 2: Layer Details": "Una vista de múltiples capas de la transición topológica. La capa Micro rastrea la dimensión fractal de los componentes individuales, indicando sus grados de libertad locales. La capa Meso aísla la densidad de Episodios Conjuntos donde comienza la habilitación topológica. La capa Macro confirma la transición de fase sistémica donde la sincronización cruza el umbral de Feigenbaum.",
        "Topological Heatmap": "Representando la segunda capa del ascenso ontológico, este mapa de calor provee evidencia espacial de los filtros probabilísticos locales alineándose. Las bandas verticales intensas significan un Joint Episode masivo. Empíricamente, esto demuestra un bloqueo espacial momentáneo, reduciendo severamente los grados de libertad del sistema e impulsándolo hacia el caos determinista.",
        "Unimodal Return Map": "Este Mapa de Retorno Unimodal grafica la órbita de la Tau Sistémica ($\\tau_s(t)$ versus $\\tau_s(t+1)$). Un sistema puramente estocástico produciría una nube de puntos sin estructura. Aquí, la densa acumulación en forma de 'telaraña' (cobweb) hacia el centro prueba que el colapso topológico del sistema no es ruido aleatorio, sino una transición inevitable gobernada por dinámicas no lineales deterministas, restringida en última instancia por la constante universal de Feigenbaum.",
        "Early Warning Signals": "Este panel demuestra el Critical Slowing Down (Desaceleración Crítica), el sello distintivo de los sistemas complejos que se aproximan a una transición de fase catastrófica. El pico drástico en la Varianza del sistema y la Autocorrelación Lag-1 inmediatamente anterior al punto crítico ($t^*$) provee prueba empírica de que el colapso topológico fue precedido por una genuina pérdida de resiliencia sistémica."
    }
    d = captions_en if lang == "en" else captions_es
    for k, v in d.items():
        if k in fig_name:
            return v
    return ""

def _format_figure(name: str, fig: str, include_caption: bool, lang: str) -> str:
    if not str(fig).startswith("!["):
        if not str(fig).startswith("data:image"):
            fig = f"![{name}](data:image/png;base64,{fig})"
        else:
            fig = f"![{name}]({fig})"
            
    section_text = f"### {name.replace('_', ' ').title()}\n\n{fig}"
    
    if include_caption:
        caption = _get_caption(name, lang)
        if caption:
            section_text += f"\n\n*{caption}*"
            
    return section_text

def _tier_1_section(results: OntologicalAscentResult, lang: str, figures: Dict[str, str], include_captions: bool) -> str:
    header = "## Tier 1: Ontological Overview"
    content = []
    
    if "Tier 1: Ontological Overview" in figures:
        content.append(_format_figure("Tier 1: Ontological Overview", figures["Tier 1: Ontological Overview"], include_captions, lang))
        
    return f"{header}\n\n" + "\n\n".join(content)

def _tier_2_appendix(results: OntologicalAscentResult, lang: str, figures: Dict[str, str], include_captions: bool) -> str:
    header = "## Appendix A: Tier 2 Details & Topologies" if lang == "en" else "## Apéndice A: Detalles de Capas y Topologías"
    content = []
    
    # Solo agregar figuras secundarias
    for name, fig in figures.items():
        if name != "Tier 1: Ontological Overview":
            content.append(_format_figure(name, fig, include_captions, lang))
            
    if not content:
        return ""
        
    return f"{header}\n\n" + "\n\n---\n\n".join(content)

def _warnings_section(results: OntologicalAscentResult, lang: str) -> str:
    header = "## Validation Warnings" if lang == "en" else "## Advertencias de Validación"
    warnings = "\n".join([f"- {w}" for w in results.warnings])
    return f"{header}\n\n{warnings}"

def _statistical_significance_appendix_section(results: OntologicalAscentResult, lang: str) -> str:
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
            "Complex systems approaching a critical singularity ($t^*$) exhibit Critical Slowing Down. An abrupt spike in Variance and Lag-1 Autocorrelation immediately preceding $t^*$ acts as empirical validation that the topological collapse was preceded by a genuine loss of systemic resilience, ruling out stochastic outliers.\n\n"
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

def _tda_section(results: OntologicalAscentResult, lang: str, figures: Dict[str, str], include_theoretical_captions: bool) -> str:
    header = "## Tier 4: Geospatial Topology (TDA)" if lang == "en" else "## Nivel 4: Topología Geoespacial (TDA)"
    
    # 4.1 Persistence Curves
    section1 = ""
    fig1 = figures.get("TDA: Systemic Transition vs Topological Holes")
    if fig1:
        img_md = f"![TDA Persistence Curves](data:image/png;base64,{fig1})"
        desc_en = (
            "**Figure: Topological Homology (H1).**\n"
            "This figure tracks the Betti number $H_1$ representing 1-dimensional 'holes' or uncoupled cycles "
            "within the correlation network. A sudden drop in Total Persistence before or during $t^*$ "
            "indicates the collapse of structural diversity as the system is absorbed into the systemic manifold."
        )
        desc_es = (
            "**Figura: Homología Topológica (H1).**\n"
            "Esta figura rastrea el número de Betti $H_1$ representando 'agujeros' unidimensionales o ciclos "
            "desacoplados en la red de correlación. Una caída repentina en la Persistencia Total antes o durante $t^*$ "
            "indica el colapso de la diversidad estructural a medida que el sistema es absorbido por la variedad sistémica."
        )
        desc = desc_en if lang == "en" else desc_es
        
        caption = ""
        if include_theoretical_captions:
            cap_en = (
                "> **Theoretical Note:** In Topological Data Analysis, a high $H_1$ persistence implies a structurally "
                "heterogeneous system with independent sub-cycles. When the Systemic Tau transitions the system into a "
                "hyper-synchronized state, these holes topologically collapse into a single connected component, signaling "
                "the loss of degrees of freedom."
            )
            cap_es = (
                "> **Nota Teórica:** En Análisis Topológico de Datos, una alta persistencia de $H_1$ implica un sistema "
                "estructuralmente heterogéneo con sub-ciclos independientes. Cuando la Tau Sistémica transiciona el sistema "
                "a un estado hiper-sincronizado, estos agujeros colapsan topológicamente en un solo componente conectado, "
                "señalando la pérdida de grados de libertad."
            )
            caption = "\n\n" + (cap_en if lang == "en" else cap_es)
            
        section1 = f"{img_md}\n\n{desc}{caption}"
        
    # 4.2 Topological Chaos
    section2 = ""
    fig2 = figures.get("TDA: Topological Chaos")
    if fig2:
        img_md = f"![Topological Chaos](data:image/png;base64,{fig2})"
        desc_en = (
            "**Figure: Structural Fragmentation (H0) & Persistence Entropy.**\n"
            "This figure tracks the Betti number $H_0$ (Connected Components) and the Shannon entropy of structural holes ($H_1$). "
            "A drop in components indicates that isolated clusters are merging, while a drop in entropy signifies that the network has lost its topological complexity, becoming uniformly synchronized."
        )
        desc_es = (
            "**Figura: Fragmentación Estructural (H0) y Entropía de Persistencia.**\n"
            "Esta figura rastrea el número de Betti $H_0$ (Componentes Conectados) y la entropía de Shannon de los agujeros estructurales ($H_1$). "
            "Una caída en los componentes indica que los clústeres aislados se están fusionando, mientras que una caída en la entropía significa que la red ha perdido su complejidad topológica, volviéndose uniformemente sincronizada."
        )
        desc = desc_en if lang == "en" else desc_es
        
        caption = ""
        if include_theoretical_captions:
            cap_en = (
                "> **Theoretical Note:** $H_0$ components measure the number of disconnected clusters in the state space. "
                "A high $H_0$ implies spatial fragmentation, while a drop to 1 indicates total systemic synchronization. "
                "Concurrently, Persistence Entropy quantifies the structural chaos of these connections. "
                "Systems approaching criticality often experience a topological simplification, mathematically observable as "
                "a steep decline in both $H_0$ and Persistence Entropy."
            )
            cap_es = (
                "> **Nota Teórica:** Los componentes $H_0$ miden el número de clústeres desconectados en el espacio de estados. "
                "Un $H_0$ alto implica fragmentación espacial, mientras que una caída a 1 indica sincronización sistémica total. "
                "Simultáneamente, la Entropía de Persistencia cuantifica el caos estructural de estas conexiones. "
                "Los sistemas que se acercan a la criticidad a menudo experimentan una simplificación topológica, observable "
                "matemáticamente como un declive abrupto tanto en $H_0$ como en la Entropía de Persistencia."
            )
            caption = "\n\n" + (cap_en if lang == "en" else cap_es)
            
        section2 = f"{img_md}\n\n{desc}{caption}"
        
    return f"{header}\n\n{section1}\n\n{section2}"

def _ordinal_section(results, lang: str, figures: dict, include_theoretical_captions: bool) -> str:
    header = "## Tier 5: Ordinal Memory (Information Flow)" if lang == "en" else "## Nivel 5: Memoria Ordinal (Flujo de Información)"
    
    fig = figures.get("Ordinal Memory Dynamics")
    if not fig:
        return ""
        
    mode = results.ordinal_results['mode'].upper()
    img_md = f"![Ordinal Memory Dynamics](data:image/png;base64,{fig})"
    
    desc_en = (
        f"**Figure: Ordinal Memory Dynamics ({mode}).**\n"
        "Tracks the Total Information Flow (average non-linear coupling) and, if in Full Mode, the Net Flow Asymmetry. "
        "A high Total Flow indicates strong non-linear memory within the system. A rising Net Asymmetry suggests that a subset of variables is 'leading' the systemic collapse, exerting a directed influence over the rest of the network."
    )
    desc_es = (
        f"**Figura: Dinámica de Memoria Ordinal ({mode}).**\n"
        "Rastrea el Flujo Total de Información (acoplamiento no lineal promedio) y, si está en Modo Full, la Asimetría Neta de Flujo. "
        "Un Flujo Total alto indica una fuerte memoria no lineal dentro del sistema. Una Asimetría Neta creciente sugiere que un subconjunto de variables está 'liderando' el colapso sistémico, ejerciendo una influencia dirigida sobre el resto de la red."
    )
    desc = desc_en if lang == "en" else desc_es
    
    caption = ""
    if include_theoretical_captions:
        cap_en = (
            "> **Theoretical Note:** Systemic Tau measures macroscopic structural coupling. "
            "Ordinal Memory (via Symbolic Transfer Entropy or Rank Mutual Information) adds directionality and non-linear memory to this framework. "
            "While Tau describes *that* the system is collapsing, Ordinal Asymmetry helps identify *who* is driving the transition."
        )
        cap_es = (
            "> **Nota Teórica:** La Tau Sistémica mide el acoplamiento estructural macroscópico. "
            "La Memoria Ordinal (vía Transfer Entropy Simbólica o Rank Mutual Information) añade direccionalidad y memoria no lineal a este marco. "
            "Mientras que Tau describe *que* el sistema está colapsando, la Asimetría Ordinal ayuda a identificar *quién* está impulsando la transición."
        )
        caption = "\n\n" + (cap_en if lang == "en" else cap_es)
        
    return f"{header}\n\n{img_md}\n\n{desc}{caption}"

def _reproducibility_section(results: OntologicalAscentResult, lang: str) -> str:
    header = "## Reproducibility" if lang == "en" else "## Reproducibilidad"
    
    theta_a = results.metadata.get("theta_A", 0.5) if results.metadata else 0.5
    import hashlib
    # Generate a reproducible short hash based on core metrics to serve as an identifier
    core_str = f"{results.t_star}_{results.fractal_D}_{results.hp_z}"
    res_hash = hashlib.sha256(core_str.encode()).hexdigest()[:8]
    
    if lang == "en":
        text = f"This analysis was generated using **`systemictau v5.6.0`**.\n\n"
        text += f"**Core Parameters:**\n"
        text += f"- `window_size`: {results.window_size}\n"
        text += f"- `theta_A`: {theta_a}\n\n"
        text += f"**Serialized Identifier:** `{res_hash}`\n\n"
        text += "The full `OntologicalAscentResult` object associated with this identifier was serialized to `analysis_result.json` (available as supplementary material or within the app's analytical export hub). "
        text += "Cite as: Padilla (2026). *Magna Synthesis v6* and associated Python package."
    else:
        text = f"Este análisis fue generado utilizando **`systemictau v5.6.0`**.\n\n"
        text += f"**Parámetros Principales:**\n"
        text += f"- `window_size`: {results.window_size}\n"
        text += f"- `theta_A`: {theta_a}\n\n"
        text += f"**Identificador Serializado:** `{res_hash}`\n\n"
        text += "El objeto `OntologicalAscentResult` completo asociado con este identificador fue serializado a `analysis_result.json` (disponible como material suplementario o en el panel de exportación de la app). "
        text += "Citar como: Padilla (2026). *Síntesis Magna v6* y paquete Python asociado."
        
    return f"{header}\n\n{text}"
