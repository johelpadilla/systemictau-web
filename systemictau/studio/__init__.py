"""
Systemic Tau Studio
A modern, research-grade web application for ontological systems analysis.

Heavy dependencies (matplotlib, streamlit) are optional.
Importing the top-level studio package is safe; the actual functions
are available after `pip install 'systemictau[studio]'` (or [visualization] + streamlit).
"""

from __future__ import annotations
from typing import Any

__all__ = [
    "apply_publication_style",
    "get_scale_palette",
    "plot_multi_scale_tau_trajectories",
    "plot_ontological_decay",
    "plot_recd_discretization",
    "plot_scale_detail",
    "export_figure",
    "create_report_pdf",
    "HAS_STUDIO_VIZ",
]

HAS_STUDIO_VIZ = False

# Soft imports so the package can be imported without optional deps
try:
    from .viz import (  # noqa: F401
        apply_publication_style,
        get_scale_palette,
        plot_multi_scale_tau_trajectories,
        plot_ontological_decay,
        plot_recd_discretization,
        plot_scale_detail,
        export_figure,
        create_report_pdf,
    )
    HAS_STUDIO_VIZ = True
except Exception:
    # Users will get a clear message when they actually call a viz function
    def _missing_viz(*args: Any, **kwargs: Any) -> Any:
        raise ImportError(
            "Systemic Tau Studio visualizations require matplotlib. "
            "Install with: pip install 'systemictau[studio]' or "
            "pip install 'systemictau[visualization]' matplotlib"
        )

    apply_publication_style = _missing_viz  # type: ignore
    get_scale_palette = _missing_viz  # type: ignore
    plot_multi_scale_tau_trajectories = _missing_viz  # type: ignore
    plot_ontological_decay = _missing_viz  # type: ignore
    plot_recd_discretization = _missing_viz  # type: ignore
    plot_scale_detail = _missing_viz  # type: ignore
    export_figure = _missing_viz  # type: ignore
    create_report_pdf = _missing_viz  # type: ignore
