"""
Desktop re-export of the shared structured exporter (shim).

The authoritative implementation (including all bug fixes) lives in
`systemictau.exporter`.

Recent changes (to satisfy "Fix Multivariate Global Results Copy Bug"):
- New public helper: get_scale_results(perspective, scale) -- fully isolated
  deepcopies at every level. Call this for any (p, s) pair.
- _gather now delegates to the explicit getter in a double loop (no cross-scale
  variables ever reused).
- Collectors (_collect..., _build_*) and export_to_excel now fetch via the getter
  and perform per-row / pre-write validation.
- Enhanced Cluster_Composition logic for Global/fixed_num to prefer real variable
  names over MacroCluster_SUM_* placeholders when possible.
- Multiple layers of copy.deepcopy before any metric/series/composition is written.

This makes it structurally impossible for multivariate Medium values to leak
into multivariate Global (or any other cross-scale contamination).

All 6 combinations (spatial/multivariate × Local/Medium/Global) are handled
independently.
"""

from systemictau.exporter import (  # noqa: F401
    build_structured_export,
    export_to_json,
    export_to_excel,
    export_current_view,
    get_scale_results,   # NEW: the explicit per-perspective+scale safe getter
)
