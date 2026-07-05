"""
Systemic Tau Studio
A modern, research-grade platform for ontological systems analysis.

Run with:
    PYTHONPATH=src streamlit run src/systemictau/studio/app.py

Or after install:
    systemictau-studio
"""

from __future__ import annotations
import io
import os
import tempfile
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st


# ------------------------------------------------------------------
# Direct file saving helpers (critical for pywebview desktop app)
# ------------------------------------------------------------------
# st.download_button for JSON, Excel, PDF etc. often causes the webview
# to navigate and display the content inline (or in a frame), replacing
# the UI with no back button. We bypass Streamlit's download mechanism
# completely by writing directly to ~/Downloads/SystemicTauStudio/
def _save_to_downloads(data: bytes, suggested_filename: str) -> str:
    """Save bytes directly to ~/Downloads/SystemicTauStudio/ with timestamp.
    Preserves the original extension. Returns the full path.
    """
    downloads = Path.home() / "Downloads" / "SystemicTauStudio"
    downloads.mkdir(parents=True, exist_ok=True)

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    p = Path(suggested_filename)
    stem = p.stem
    suffix = p.suffix or ""
    filename = f"{stem}_{ts}{suffix}"
    path = downloads / filename

    with open(path, "wb") as f:
        f.write(data)

    return str(path)


def _reveal_in_finder(path: str):
    """Reveal the file in Finder (macOS). Non-fatal if it fails."""
    try:
        import subprocess
        subprocess.run(["open", "-R", path], check=False)
    except Exception:
        pass



# Core engine
from systemictau import ChaosGenerator
from systemictau.studio.analysis import run_multi_ontological_analysis, simulate_what_if
from systemictau.studio.viz import (
    plot_multi_scale_tau_trajectories,
    plot_ontological_decay,
    plot_recd_discretization,
    plot_scale_detail,
    export_figure,
    create_report_pdf,
    get_scale_palette,
)

# Sensitivity & robustness analysis (new in v4.x)
try:
    from systemictau.sensitivity import run_parameter_sensitivity, SensitivityReport
    HAS_SENSITIVITY = True
except Exception:
    HAS_SENSITIVITY = False
    run_parameter_sensitivity = None  # type: ignore
    SensitivityReport = None  # type: ignore

# Structured export (JSON + Excel)
try:
    from systemictau.exporter import (
        export_to_json,
        export_to_excel,
        build_structured_export,
        reconstruct_analyses_from_structured_export,
    )
    HAS_EXPORTER = True
except Exception:
    HAS_EXPORTER = False
    export_to_json = None
    export_to_excel = None
    reconstruct_analyses_from_structured_export = None


# Optional desktop scale manager reuse for more advanced partitioning
try:
    from systemictau.desktop.scale_manager import OntologicalScaleManager
except Exception:
    OntologicalScaleManager = None

# Session save/load (shared with desktop, works for Streamlit too)
try:
    from systemictau.desktop.session_manager import SessionManager
except Exception:
    SessionManager = None


st.set_page_config(
    page_title="Systemic Tau Studio",
    page_icon="◐",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _check_optional_dependencies():
    """Expert UX: proactively warn users about missing common packages."""
    missing = []
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        missing.append("openpyxl (needed for .xlsx uploads)")

    if missing:
        with st.container():
            st.warning(
                "**Some features may be limited.**\n\n"
                "Missing packages: " + ", ".join(missing) + "\n\n"
                "Install them with:\n"
                "```bash\n"
                "uv pip install --python .venv/bin/python " + " ".join(missing) + "\n"
                "```"
            )


_check_optional_dependencies()

# =============================================================================
# Elegant minimal theming (light, scientific)
# =============================================================================
st.markdown(
    """
    <style>
    .main .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1180px; }
    h1, h2, h3 { font-weight: 600; letter-spacing: -0.2px; }
    .stButton > button {
        background: linear-gradient(180deg, #1565C0 0%, #0D47A1 100%);
        color: white; border: none; border-radius: 6px; font-weight: 600;
        padding: 0.5rem 1.1rem;
    }
    .stButton > button:hover { filter: brightness(1.05); }
    .metric-card {
        background: #FAFAFA;
        border: 1px solid #E8E8E8;
        border-radius: 8px;
        padding: 0.75rem 1rem;
    }
    .insight-box {
        background: #F5F7FA;
        border-left: 4px solid #1565C0;
        padding: 0.9rem 1rem;
        border-radius: 6px;
        font-size: 0.95rem;
        line-height: 1.45;
    }
    .scale-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

PALETTE = get_scale_palette()


def _badge(text: str, color: str) -> str:
    return f'<span class="scale-badge" style="background:{color}">{text}</span>'


def generate_ontological_insight(
    scale_metrics: Dict[str, Dict[str, float]],
    scale_results: Optional[Dict[str, Dict[str, Any]]] = None,
    perspective: Optional[str] = None,
) -> str:
    """
    Produce a rich, paradigm-aligned interpretive discussion.
    perspective: "spatial" | "multivariate" | None
    When provided, the language is adapted to the meaning of the modules.
    """
    if "Local" not in scale_metrics:
        return "Run the multi-ontological analysis to receive scale-aware interpretation."

    lines = []

    l = scale_metrics.get("Local", {})
    m = scale_metrics.get("Medium", {})
    g = scale_metrics.get("Global", l)  # fallback

    # Relative changes
    def _rel_change(a, b):
        denom = abs(a) + 1e-9
        return (b - a) / denom

    tau_l = l.get("mean_tau", 0.0)
    tau_m = m.get("mean_tau", tau_l)
    tau_g = g.get("mean_tau", tau_l)

    tau_lg = _rel_change(tau_l, tau_g)
    tau_lm = _rel_change(tau_l, tau_m)

    recd_l = l.get("recd_T_final", 0.0)
    recd_g = g.get("recd_T_final", recd_l)
    recd_growth = _rel_change(recd_l, recd_g)

    var_l = l.get("tau_std", 0.0)
    var_g = g.get("tau_std", var_l)
    var_change = _rel_change(var_l, var_g)

    lines.append(
        "<b>Overall Interpretation of the Multi-Ontological Analysis</b><br>"
        "The analysis reveals how systemic properties transform under the Principle of Ontological Ascent. "
        "Aggregation from Local to Global does not merely average signals — it induces qualitative shifts in the dominant regime."
    )

    if perspective == "spatial":
        lines.append(
            "<i>Perspective: Spatial (locations as modules)</i> — Results describe coupling and reorganization of incidence patterns across geographic units."
        )
    elif perspective == "multivariate":
        lines.append(
            "<i>Perspective: Multivariate (variables as modules)</i> — Results describe co-evolution and coupling among the measured signals (incidence + environmental drivers)."
        )

    # Coupling behavior
    if tau_lg > 0.20:
        lines.append(
            f"<b>Strengthened macro-coupling</b>: Mean Systemic Tau rose substantially from Local ({tau_l:.3f}) "
            f"to Global ({tau_g:.3f}), a relative increase of ~{tau_lg*100:.0f}%. "
            "At coarser scales the system exhibits more consistent, organized coupling."
        )
    elif tau_lg < -0.15:
        lines.append(
            f"<b>Decaying topological coupling</b>: Mean τ_s declined ~{abs(tau_lg)*100:.0f}% from Local to Global "
            f"({tau_l:.3f} → {tau_g:.3f}). Fine-grained correlations are filtered by aggregation."
        )
    else:
        lines.append(
            f"Mean Systemic Tau remained relatively stable across scales (Local {tau_l:.3f}, Global {tau_g:.3f})."
        )

    # Variability
    if var_change > 0.15:
        lines.append(
            f"Variability of τ_s increased at higher scales (σ rose ~{var_change*100:.0f}%), indicating that while mean behavior may stabilize or strengthen, the system still expresses structured fluctuations at macro levels."
        )
    elif var_change < -0.15:
        lines.append("Variability decreased with ascent, consistent with smoothing of micro-dynamics.")

    # RECD / temporal rigidity (the strongest signature in most runs)
    if recd_growth > 0.25:
        lines.append(
            f"<b>Emergent temporal rigidity</b>: RECD accumulated time increased dramatically "
            f"({recd_l:.3f} → {recd_g:.3f}, ~{recd_growth*100:.0f}% growth). "
            "Higher ontological levels reveal (or generate) markedly more discretized temporal structure. "
            "The system becomes progressively more 'locked' into discrete patterns as stochastic noise is filtered."
        )
    elif recd_growth > 0.08:
        lines.append(
            f"RECD shows moderate growth (~{recd_growth*100:.0f}%), suggesting gradual emergence of temporal discretization with scale."
        )
    elif recd_growth < -0.1:
        lines.append("RECD accumulation declined at macro scale.")

    # Include Medium if meaningful
    if m and abs(tau_lm) > 0.08:
        lines.append(
            f"At the Medium scale, mean τ_s was {tau_m:.3f} (Local→Medium change ~{tau_lm*100:.0f}%). "
            "Intermediate aggregation already produces noticeable reorganization of systemic behavior."
        )

    # t* and breakpoints
    if scale_results:
        tstars = []
        for sc in ("Local", "Medium", "Global"):
            ts = scale_results.get(sc, {}).get("t_star")
            if ts is not None and not (isinstance(ts, float) and np.isnan(ts)):
                tstars.append(f"{sc} t*≈{int(ts)}")
        if tstars:
            lines.append(
                f"<b>Scale-dependent transitions</b>: {'; '.join(tstars)}. "
                "The timing and expression of reorganization points shift across ontological levels — an important consideration for early-warning applications."
            )

    lines.append(
        "These patterns — whether coupling strengthens or decays, combined with the clear rise in RECD — constitute a signature of the Principle of Ontological Ascent: the system undergoes a transformation in its dominant dynamical character rather than simple rescaling."
    )

    return "<br><br>".join(lines)


# =============================================================================
# Data loading helpers
# =============================================================================
def load_sample_panel() -> pd.DataFrame:
    """Load the raw Aedes mock panel (long format).
    Returns the original long table (NID × EpiWeek × Count + climate covariates).
    NO auto-pivoting here. The user explicitly chooses the analytical perspective afterwards.
    This gives full control over whether modules = locations or modules = variables.
    """
    here = Path(__file__).resolve().parent.parent.parent.parent
    sample_path = here / "Aedes_Mock_Panel.csv"
    try:
        if not sample_path.exists():
            sample_path = Path("/Users/johelpadilla/Investigaciones/systemictau_v4/Aedes_Mock_Panel.csv")
        df = pd.read_csv(sample_path)
        return df
    except Exception:
        # Synthetic long-style fallback
        X = ChaosGenerator.logistic_map_coupled(320, 4, coupling=0.22, r=3.9)
        df = pd.DataFrame(X, columns=["Count", "Temp", "Humidity", "Precipitation"])
        df.insert(0, "EpiWeek", pd.date_range("2020-01-01", periods=len(df), freq="W"))
        df.insert(0, "NID", "Loc_Synth")
        return df


def prepare_spatial_view(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Explicit transformation: long panel → wide where each location is a module.
    Uses Count as the measure. Answers: coupling of incidence across space.
    """
    cols_lower = {c.lower(): c for c in raw_df.columns}
    idc = cols_lower.get("nid") or cols_lower.get("id") or list(raw_df.columns)[0]
    tc = (cols_lower.get("epitime") or cols_lower.get("epiweek") or
          cols_lower.get("time") or cols_lower.get("week") or list(raw_df.columns)[1])
    vc = cols_lower.get("count") or "Count"

    wide = raw_df.pivot_table(index=tc, columns=idc, values=vc, aggfunc="mean").reset_index()
    time_like = [c for c in wide.columns if str(c).lower() in ("epitime", "epiweek", "time", "week", "date", "index")]
    for t in time_like:
        if t in wide.columns:
            wide = wide.drop(columns=[t])
    wide = wide.dropna(axis=1, thresh=max(5, int(0.5 * len(wide))))
    return wide.reset_index(drop=True)


def prepare_multivariate_view(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Explicit transformation: create one time series per measured variable.
    Averages across locations (system-level view). 
    Answers: how Count, Temp, Humidity and Precipitation couple as a system.
    """
    num_cols = raw_df.select_dtypes(include=[np.number]).columns.tolist()
    # Try to find time column
    cols_lower = {c.lower(): c for c in raw_df.columns}
    tc = (cols_lower.get("epitime") or cols_lower.get("epiweek") or
          cols_lower.get("time") or cols_lower.get("week") or None)

    if tc and tc in raw_df.columns:
        grouped = raw_df.groupby(tc)[num_cols].mean().reset_index()
    else:
        grouped = raw_df[num_cols].copy()
        grouped.insert(0, "time", range(len(grouped)))

    # Drop any time-like from the variable list
    timeish = ("time", "week", "epi", "date")
    value_cols = [c for c in grouped.columns if not any(t in str(c).lower() for t in timeish)]
    # Keep a clean frame with only the variable series
    mv = grouped[value_cols].copy()
    # Make column names clear
    mv = mv.rename(columns={c: f"System_{c}" for c in mv.columns if c in num_cols})
    return mv.reset_index(drop=True)


def simple_pivot_panel(df: pd.DataFrame, id_col: str, time_col: str, value_col: str) -> pd.DataFrame:
    """Pivot long panel (multiple locations × time) into wide systemic format."""
    wide = df.pivot_table(index=time_col, columns=id_col, values=value_col, aggfunc="mean").reset_index()
    wide = wide.dropna(axis=1, thresh=max(5, int(0.5 * len(wide))))
    return wide


# =============================================================================
# Session Save / Load using Structured JSON (official format, replaces .stausession)
# =============================================================================
def _save_session_as_json():
    """Save current full session state as structured JSON.
    Reuses the exporter so both perspectives, scales, series and clusters are persisted.
    """
    if not HAS_EXPORTER or export_to_json is None:
        st.error("Structured exporter not available.")
        return

    analyses = st.session_state.get("analyses", {}) or {}
    if not analyses:
        st.warning("No analysis results to save. Run both perspectives first.")
        return

    df_raw = st.session_state.get("df_raw")
    active_view = st.session_state.get("active_view")
    window_size = st.session_state.get("last_window")

    try:
        json_bytes = export_to_json(
            analyses=analyses,
            df_raw=df_raw,
            active_view=active_view,
            window_size=window_size,
        )
        filename = f"systemic_tau_session_{dt.datetime.now().strftime('%Y%m%d_%H%M')}.json"
        saved_path = _save_to_downloads(json_bytes, filename)
        st.success(f"✅ Session saved to: {saved_path}")
        st.caption("Contains both perspectives + all scales + clusters.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📂 Reveal in Finder", key="reveal_session_json"):
                _reveal_in_finder(saved_path)
        with col2:
            if st.button("📁 Open Downloads", key="open_downloads_session"):
                try:
                    import subprocess
                    subprocess.run(["open", str(Path.home() / "Downloads")], check=False)
                except Exception:
                    pass
    except Exception as e:
        st.error(f"Failed to save session JSON: {e}")


def _load_session_from_json(uploaded_file):
    """Load a structured JSON session.
    Rebuilds analyses for both perspectives so switching works without re-running.
    """
    if not HAS_EXPORTER or reconstruct_analyses_from_structured_export is None:
        st.error("JSON session loader not available.")
        return

    try:
        content = uploaded_file.getvalue()
        data = json.loads(content)

        reconstructed = reconstruct_analyses_from_structured_export(data)

        if not reconstructed:
            st.error("No valid perspectives found in JSON.")
            return

        st.session_state.analyses = reconstructed
        st.session_state.prepared_views = {p: None for p in reconstructed}  # markers so view selector works

        # Set active
        active = st.session_state.get("active_view")
        if active not in reconstructed:
            active = "spatial" if "spatial" in reconstructed else next(iter(reconstructed))
        st.session_state.active_view = active
        st.session_state.analysis = reconstructed.get(active)

        # Config hints
        any_cfg = next(iter(reconstructed.values())).get("config", {}) or {}
        st.session_state.last_window = any_cfg.get("window_size", 13)

        for view, a in reconstructed.items():
            vcfg = a.get("config", {}) or {}
            cm = vcfg.get("clustering_method", {}) or {}
            auto_med = (cm.get("medium", "auto") == "auto") if isinstance(cm, dict) else True
            auto_g = (cm.get("global", "auto") == "auto") if isinstance(cm, dict) else True
            st.session_state[f"auto_medium_{view}"] = auto_med
            st.session_state[f"auto_global_{view}"] = auto_g
            if view == active:
                st.session_state["auto_medium"] = auto_med
                st.session_state["auto_global"] = auto_g
            if vcfg.get("global_groups") or vcfg.get("medium_groups"):
                st.session_state["had_manual_global"] = True

        st.session_state.pop("sensitivity_report", None)

        st.success(f"Session loaded from JSON. Perspectives: {list(reconstructed.keys())}. Switch views freely.")
        st.rerun()

    except Exception as e:
        st.error(f"Failed to load session JSON: {e}")
        st.exception(e)


# =============================================================================
# Sidebar — controls
# =============================================================================
with st.sidebar:
    st.markdown("### Systemic Tau Studio")
    st.caption("Ontological Analysis • Principle of Ontological Ascent")

    st.divider()

    st.markdown("**Data**")
    uploaded = st.file_uploader("Upload CSV or XLSX", type=["csv", "xlsx", "xls"], key="uploader")

    use_sample = st.button("Load sample panel data (raw Aedes long format)", width="stretch")

    st.divider()

    st.markdown("**Analysis Parameters**")
    window_size = st.slider("Window size (w)", 5, 41, 13, step=2,
                            help="Primary analysis window. 13 is the canonical Systemic Tau value.")
    st.session_state.last_window = window_size
    st.caption("RECD uses the same w and Feigenbaum δ ≈ 4.669")

    st.divider()
    st.markdown("**Ontological Partitioning**")
    auto_medium = st.checkbox("Auto-cluster for Medium scale", value=st.session_state.get("auto_medium", True))
    auto_global = st.checkbox("Auto-cluster for Global scale", value=st.session_state.get("auto_global", True),
                              help="When on, the number of Global macro-clusters is chosen automatically based on how many Medium modules exist (recommended for most analyses).")
    if not auto_global:
        global_macros = st.slider("Global macro-clusters", 2, 4, 2,
                                  help="Number of macro-clusters to force at Global scale. Only used when 'Auto-cluster for Global' is off.")
    else:
        global_macros = 2  # will be overridden by auto logic inside the runner
        st.caption("Global will automatically pick 2–3 macro-clusters based on the Medium scale size.")

    st.divider()
    st.markdown("**Export**")
    st.caption("All figures are rendered at publication DPI. Use the buttons in the main panel after analysis.")

# =============================================================================
# Header
# =============================================================================
col_title, col_sub = st.columns([3, 2])
with col_title:
    st.markdown("# Systemic Tau Studio")
    st.markdown(
        "<span style='color:#555; font-size:1.02rem'>A research instrument for multi-scale ontological analysis "
        "of complex systems using <b>Systemic Tau (τ_s)</b> and <b>RECD</b>.</span>",
        unsafe_allow_html=True,
    )
with col_sub:
    st.markdown(
        f"<div style='text-align:right; color:#666; font-size:0.9rem'>"
        f"{_badge('Local', PALETTE['Local'])} &nbsp; {_badge('Medium', PALETTE['Medium'])} &nbsp; {_badge('Global', PALETTE['Global'])}"
        f"<br><span style='font-size:0.8rem'>Local → Medium → Global</span></div>",
        unsafe_allow_html=True,
    )

st.divider()

# =============================================================================
# Session Save & Load — Structured JSON (official, replaces .stausession)
# =============================================================================
json_col1, json_col2, json_col3 = st.columns([1, 1, 3])
with json_col1:
    if st.button("💾 Save Session (JSON)", width="stretch"):
        if st.session_state.get("analyses"):
            _save_session_as_json()
        else:
            st.warning("Run analysis on at least one perspective to save a session.")
with json_col2:
    load_json = st.file_uploader("Load Session from JSON", type=["json"], label_visibility="collapsed", key="load_json_sess")
    if load_json is not None:
        _load_session_from_json(load_json)
with json_col3:
    st.caption("Uses the structured JSON format. Saves/loads both perspectives, all scales, series and manual clusters. Simpler and more reliable than .stausession.")

st.divider()

# =============================================================================
# Session state (more flexible to support multiple analytical perspectives)
# =============================================================================
if "df_raw" not in st.session_state:
    st.session_state.df_raw = None
if "prepared_views" not in st.session_state:
    # "spatial": wide df (locations as modules), "multivariate": variables as modules, "custom": user-defined
    st.session_state.prepared_views = {}
if "active_view" not in st.session_state:
    st.session_state.active_view = None
if "analyses" not in st.session_state:
    # Allow multiple analyses: e.g. {"spatial": {...}, "multivariate": {...}}
    st.session_state.analyses = {}
if "analysis" not in st.session_state:  # legacy single result, kept for compatibility
    st.session_state.analysis = None

# =============================================================================
# Data loading section
# =============================================================================
st.markdown("## 1. Data")

# Load actions (always start from raw / neutral)
if use_sample:
    raw = load_sample_panel()
    st.session_state.df_raw = raw
    st.session_state.prepared_views = {}
    st.session_state.active_view = None
    st.session_state.analyses = {}
    st.session_state.analysis = None
    st.success("Raw Aedes panel loaded (long format: 50 locations × 100 weeks + climate covariates). You now decide how to interpret it.")

if uploaded is not None:
    try:
        fname = uploaded.name.lower()
        if fname.endswith((".xlsx", ".xls")):
            try:
                df = pd.read_excel(uploaded, engine="openpyxl")
            except ImportError:
                st.error("Missing openpyxl. Install with: uv pip install --python .venv/bin/python openpyxl")
                st.stop()
            except Exception as e:
                st.error(f"Failed to read Excel: {e}")
                st.stop()
        else:
            df = pd.read_csv(uploaded)
        st.session_state.df_raw = df
        st.session_state.prepared_views = {}
        st.session_state.active_view = None
        st.session_state.analyses = {}
        st.session_state.analysis = None
        st.success(f"Loaded {uploaded.name} — {df.shape[0]} rows × {df.shape[1]} columns (raw).")
    except Exception as e:
        st.error(f"Failed to read file: {e}")

df_raw = st.session_state.df_raw

# Show raw preview always when available
if df_raw is not None:
    with st.expander("Raw data preview (always available)", expanded=False):
        st.dataframe(df_raw.head(6), width="stretch")
        st.caption(f"Shape: {df_raw.shape} — This is the neutral starting point. Nothing has been decided yet about modules.")

st.divider()

# =============================================================================
# 1b. Explicit Analytical Perspective (the key user-controlled step)
# =============================================================================
st.markdown("## 1b. Choose your analytical perspective (what are your 'modules'?)")

st.markdown(
    "In Systemic Tau the most important decision is **what counts as a module**. "
    "The app should never auto-decide this for you. Choose the perspective that matches the scientific question you want to answer."
)

colA, colB = st.columns(2)

with colA:
    st.markdown("**Spatial View — Locations as modules**")
    st.caption(
        "Pivot the data so each location (Loc_*) becomes one time series.\n"
        "**Question answered**: How do the incidence patterns couple / synchronize across different places? "
        "(Recommended for the Aedes-style spatial panel. This implements multi-location ontological ascent.)"
    )
    if st.button("Prepare Spatial View\n(locations = modules)", disabled=(df_raw is None), width="stretch"):
        try:
            spatial_df = prepare_spatial_view(df_raw)
            st.session_state.prepared_views["spatial"] = spatial_df
            st.session_state.active_view = "spatial"
            # Do NOT clear analyses here — we want to retain previous runs for other perspectives
            st.success(f"Spatial view ready: {spatial_df.shape[1]} location modules (time × locations). Count is now the signal of each location.")
        except Exception as e:
            st.error(f"Failed to prepare spatial view: {e}")

with colB:
    st.markdown("**Multivariate View — Variables as modules**")
    st.caption(
        "Create system-level series for each measured variable (average across locations).\n"
        "**Question answered**: How do Count, temperature, humidity and precipitation couple with each other as a system?"
    )
    if st.button("Prepare Multivariate View\n(variables = modules)", disabled=(df_raw is None), width="stretch"):
        try:
            mv_df = prepare_multivariate_view(df_raw)
            st.session_state.prepared_views["multivariate"] = mv_df
            st.session_state.active_view = "multivariate"
            # Do NOT clear analyses here — retain results from other perspectives (e.g. spatial)
            st.success(f"Multivariate view ready: {mv_df.shape[1]} variable modules.")
        except Exception as e:
            st.error(f"Failed to prepare multivariate view: {e}")

st.markdown("---")
st.caption("You can prepare **both views** in the same session and run separate analyses for each perspective. Switch using the selector below.")

# View selector + custom pivot
prepared_views = st.session_state.prepared_views
active_view = st.session_state.active_view

if prepared_views:
    view_options = list(prepared_views.keys())
    default_idx = view_options.index(active_view) if active_view in view_options else 0
    chosen = st.selectbox(
        "Active data view (this decides which columns you can pick as modules)",
        view_options,
        index=default_idx,
        help="Switching here changes the pool of columns offered below as modules. You can run analyses on both views."
    )
    st.session_state.active_view = chosen
    df_for_modules = prepared_views.get(chosen)
else:
    df_for_modules = None
    st.info("Prepare at least one view above (Spatial or Multivariate), or use the manual pivot below for full custom control.")

# Manual / custom pivot always available (full user power)
with st.expander("Manual pivot & custom module preparation (full control)", expanded=bool(df_raw is not None and not prepared_views)):
    if df_raw is not None:
        cols = df_raw.columns.tolist()
        idc = st.selectbox("ID / Location column", cols, index=0, key="manual_id")
        tc = st.selectbox("Time column", cols, index=min(1, len(cols)-1), key="manual_time")
        possible_values = [c for c in cols if c not in (idc, tc)]
        vc = st.selectbox("Value column to pivot (or leave for direct use)", possible_values, index=0, key="manual_val")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            if st.button("Pivot this value → wide (chosen column becomes modules)"):
                try:
                    wide = simple_pivot_panel(df_raw, idc, tc, vc)
                    st.session_state.prepared_views["custom"] = wide
                    st.session_state.active_view = "custom"
                    st.success(f"Custom pivoted view ready with {wide.shape[1]-1} modules.")
                except Exception as e:
                    st.error(f"Pivot failed: {e}")
        with col_p2:
            if st.button("Use raw numeric columns directly (no pivot)"):
                st.session_state.prepared_views["custom"] = df_raw.copy()
                st.session_state.active_view = "custom"
                st.success("Using raw data columns directly as potential modules.")

# Current active dataframe for module selection
df = prepared_views.get(st.session_state.active_view) if prepared_views else None

if df is not None:
    with st.expander(f"Preview of current active view: {st.session_state.active_view}", expanded=False):
        st.dataframe(df.head(5), width="stretch")
        st.caption(f"Shape: {df.shape}. These are the series available as modules for this perspective.")
else:
    st.info("Load data + prepare a view (Spatial or Multivariate) above. The app will never auto-decide for you what a 'module' is.")

st.divider()

# =============================================================================
# 2. Ontological Configuration & Analysis
# =============================================================================
st.markdown("## 2. Ontological Configuration & Analysis")

# Show context about the current perspective
if st.session_state.active_view:
    perspective_help = {
        "spatial": "Current view: **Spatial** — columns are locations. Selecting them means you are studying coupling of incidence across space.",
        "multivariate": "Current view: **Multivariate** — columns are system-level variables (Count, climate...). Selecting them studies coupling between signals.",
        "custom": "Current view: **Custom** — you defined the columns yourself."
    }
    st.info(perspective_help.get(st.session_state.active_view, ""))

# Show which perspectives have retained analyses (so user knows switching back is safe)
analyses = getattr(st.session_state, "analyses", {})
if len(analyses) >= 1:
    retained = ", ".join(f"**{k}**" for k in analyses.keys())
    st.caption(f"Retained analyses in this session: {retained}  —  Switch views above to see previous results without re-running.")

if df is None:
    try:
        st.stop()
    except Exception:
        pass
    all_num = []
    value_cols = []
else:
    all_num = df.select_dtypes(include=[np.number]).columns.tolist()

if df is None or not all_num:
    value_cols = []
else:
    timeish = ("time", "week", "epi", "date", "index")
    all_num = [c for c in all_num if not any(t in str(c).lower() for t in timeish)]

    value_cols = st.multiselect(
        "Modules / Variables at Local scale (these are the components whose coupling you want to study)",
        all_num,
        default=all_num[: min(12, len(all_num))],
        help="You decide. In Spatial view these are locations. In Multivariate view these are the measured variables (or derived series)."
    )

    # Live preview of what Auto-Global will decide (very useful in spatial with dozens of locations)
    auto_g = locals().get('auto_global', True)
    if auto_g and len(value_cols) >= 2:
        n_loc = len(value_cols)
        if n_loc <= 6:
            eff_g = 2
        elif n_loc <= 10:
            eff_g = 3
        elif n_loc <= 15:
            eff_g = 3
        elif n_loc <= 20:
            eff_g = 4
        else:
            eff_g = min(6, 3 + (n_loc - 15) // 5)

        # quick n_med for caption only
        n_med_preview = max(2, min(4, n_loc // 3 or 2))
        st.caption(f"Auto-Global preview (spatial-friendly): with your current selection of {n_loc} modules, Global will automatically use **{eff_g} macro-clusters** (based on n_local; Medium ~{n_med_preview}).")

    # Only enforce module selection (and stop) when there are NO loaded results for this view.
    # After loading a session we may have analyses[view] but no current "df" prepared for module picking.
    # In that case we want to show the restored results without forcing the user to pick modules again.
    _analyses_now = getattr(st.session_state, "analyses", {}) or {}
    _active_now = st.session_state.get("active_view")
    _has_loaded_for_view = bool(_analyses_now.get(_active_now, {}).get("scale_results")) if _active_now else False

    if len(value_cols) < 2 and not _has_loaded_for_view:
        st.warning("Select at least two columns to form a systemic analysis (τ_s requires coupling between series).")
        try:
            st.stop()
        except Exception:
            pass
        value_cols = []


# Medium grouping UI (simple)
medium_groups: Dict[str, List[str]] = {}
if df is not None and value_cols and not auto_medium:
    st.markdown("**Define Medium clusters manually** (optional)")
    remaining = list(value_cols)
    cluster_i = 1
    while remaining and cluster_i <= 5:
        sel = st.multiselect(f"Medium cluster {cluster_i}", remaining, default=remaining[: max(1, len(remaining)//3)])
        if sel:
            medium_groups[f"Cluster_{cluster_i}"] = sel
            remaining = [x for x in remaining if x not in sel]
        cluster_i += 1
        if not sel:
            break
    if len(medium_groups) == 1:
        st.warning("You defined only one manual Medium cluster. Analysis will still run (auto-repair to ≥2), but for true ontological contrast define 2+ clusters.")

# NEW: Manual Global clusters (per Implementation Plan)
global_groups: Dict[str, List[str]] = {}
use_manual_global = False
if df is not None and value_cols and not auto_global:
    use_manual_global = st.checkbox(
        "Define manual macro-clusters for Global scale",
        value=False,
        help="Instead of automatic hierarchical clustering, explicitly group variables into 2–6 macro-clusters."
    )
    if use_manual_global:
        st.markdown("**Define Global macro-clusters manually** (2 to 6 required)")
        g_remaining = list(value_cols)
        g_i = 1
        while g_remaining and g_i <= 6:
            g_sel = st.multiselect(
                f"Global macro-cluster {g_i}",
                g_remaining,
                default=g_remaining[:max(1, len(g_remaining)//3)],
                key=f"global_manual_{g_i}"
            )
            if g_sel:
                global_groups[f"Macro_{g_i}"] = g_sel
                g_remaining = [x for x in g_remaining if x not in g_sel]
            g_i += 1
            if not g_sel:
                break
        if len(global_groups) < 2:
            st.warning("Define at least 2 macro-clusters for Global to use manual mode. Otherwise automatic clustering will be used.")
        elif len(global_groups) > 6:
            st.warning("Maximum 6 macro-clusters supported.")

# Run button - support running the same view multiple times or different views
if df is not None and value_cols:
    col_run, col_info = st.columns([2, 3])
    with col_run:
        run_clicked = st.button("🚀 RUN MULTI-ONTOLOGICAL ANALYSIS", type="primary", width="stretch")

    with col_info:
        view_label = st.session_state.active_view or "custom"
        st.caption(
            f"Running on **{view_label}** view. "
            "Computes Systemic Tau (τ_s) + RECD at Local, Medium, and Global scales. "
            "The meaning of the results depends on what you chose as modules above."
        )
else:
    run_clicked = False

if run_clicked and df is not None and value_cols:
    view_key = st.session_state.active_view or "custom"
    with st.spinner(f"Running ontological analysis on the '{view_key}' perspective..."):
        try:
            analysis = run_multi_ontological_analysis(
                df,
                value_cols=value_cols,
                window_size=window_size,
                medium_groups=medium_groups if medium_groups else None,
                global_groups=global_groups if global_groups and len(global_groups) >= 2 else None,
                global_num_clusters=int(global_macros),
                auto_global=auto_global,
            )
            # Store per view so both perspectives can coexist
            # Attach clean insight for the report
            try:
                clean_insight = generate_ontological_insight(
                    analysis.get("scale_metrics", {}),
                    scale_results=analysis.get("scale_results", {}),
                    perspective=view_key
                )
                analysis["insight_text"] = clean_insight
            except Exception:
                pass

            st.session_state.analyses[view_key] = analysis
            st.session_state.analysis = analysis
            st.session_state.last_value_cols = list(value_cols)
            st.session_state.last_window = window_size
            st.success(f"Analysis complete for **{view_key}** view. Results stored separately.")
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.exception(e)

# Results now come from the active view's analysis (or legacy)
analyses = getattr(st.session_state, "analyses", {})
active_view = st.session_state.active_view
analysis = analyses.get(active_view) if active_view else getattr(st.session_state, "analysis", None)

if not analyses and analysis is None:
    try:
        st.info("Prepare a view in section 1b, choose your modules, then click RUN. You can prepare and analyze **both Spatial and Multivariate** perspectives in the same session.")
        st.stop()
    except Exception:
        pass
    scale_results = {}
    scale_metrics = {}
    _has_results = False
else:
    if analysis:
        scale_results = analysis.get("scale_results", {})
        scale_metrics = analysis.get("scale_metrics", {})
        _has_results = True
    else:
        # Fallback if only legacy exists
        scale_results = analysis.get("scale_results", {}) if analysis else {}
        scale_metrics = analysis.get("scale_metrics", {}) if analysis else {}
        _has_results = bool(scale_results)

if not _has_results:
    # Skip heavy UI on bare import
    metrics_df = pd.DataFrame()
    value_cols = []
    df = None
    scale_results = {}
    pass
else:
    st.divider()

    # =============================================================================
    # Results Dashboard
    # =============================================================================
    st.markdown("## 3. Ontological Results")

    # Show all available analyses so user can compare perspectives
    if len(analyses) > 1:
        st.markdown("**Analyses run in this session** — use the selector in 1b or the tabs below to explore each fully.")

    st.markdown("### Comparative Metrics")

    metrics_df = pd.DataFrame(scale_metrics).T
    if not metrics_df.empty:
        try:
            metrics_df = metrics_df[["mean_tau", "recd_T_final", "coherence", "n_modules"]]
        except Exception:
            pass
    metrics_df = metrics_df.round(4) if not metrics_df.empty else metrics_df
    metrics_df.index.name = "Scale"

    if "Local" in metrics_df.index:
        base = metrics_df.loc["Local"]
        for col in ["mean_tau", "recd_T_final", "coherence"]:
            if col in metrics_df.columns:
                metrics_df[f"Δ{col}_vs_Local"] = (metrics_df[col] - base[col]).round(4)

    tstar_row = {}
    for sc in ["Local", "Medium", "Global"]:
        if sc in scale_results:
            ts = scale_results[sc].get("t_star")
            tstar_row[sc] = int(ts) if ts is not None and not (isinstance(ts, float) and np.isnan(ts)) else "—"
    if tstar_row:
        metrics_df["t* (reorg)"] = pd.Series(tstar_row)

    st.dataframe(metrics_df, width="stretch")

    cfg = analysis.get("config", {}) if analysis else {}
    parts = []
    if cfg.get("auto_global"):
        parts.append(f"Global: auto → {cfg.get('effective_global_clusters', cfg.get('global_num_clusters', '?'))} macro-clusters")
    else:
        parts.append(f"Global: manual {cfg.get('global_num_clusters', '?')} macro-clusters")
    if cfg.get("medium_groups"):
        parts.append("Medium: manual")
    else:
        parts.append("Medium: auto")
    if parts:
        st.caption("Partitioning used: " + "  •  ".join(parts) + f"   |   View: {active_view or '—'}")

# Automatic side-by-side comparison when both perspectives have been run
if len(analyses) >= 2:
    st.markdown("#### Side-by-side Metrics Comparison")
    comp_cols = st.columns(2)
    view_names = list(analyses.keys())[:2]
    for idx, vname in enumerate(view_names):
        with comp_cols[idx]:
            a = analyses[vname]
            mdf = pd.DataFrame(a.get("scale_metrics", {})).T
            if not mdf.empty:
                try:
                    mdf = mdf[["mean_tau", "recd_T_final", "n_modules"]]
                except Exception:
                    pass
            mdf = mdf.round(4)
            st.markdown(f"**{vname} view**")
            st.dataframe(mdf, width="stretch")
            # Simple delta between views for key metric
            if len(view_names) == 2 and idx == 0:
                other_m = pd.DataFrame(analyses[view_names[1]].get("scale_metrics", {})).T
                if not other_m.empty and "Local" in mdf.index and "Local" in other_m.index:
                    delta_tau = (mdf.loc["Local", "mean_tau"] - other_m.loc["Local", "mean_tau"])
                    st.caption(f"Δ Local τ vs other view: {delta_tau:+.4f}")

# =============================================================================
# SENSITIVITY & ROBUSTNESS ANALYSIS (new in v4.x)
# =============================================================================
if HAS_SENSITIVITY and "df_raw" in st.session_state and st.session_state.get("df_raw") is not None:
    with st.expander("🔬 Sensitivity & Robustness Analysis (Parameter Sweep)", expanded=False):
        st.markdown(
            "Evaluate how stable **t***, mean **τ_s**, and **RECD** are when key parameters change. "
            "Essential for scientific credibility."
        )

        current_value_cols = st.session_state.get("last_value_cols") or (value_cols if 'value_cols' in locals() else [])

        col_a, col_b = st.columns(2)
        with col_a:
            test_windows = st.multiselect(
                "Test Window sizes (w)",
                options=list(range(7, 25)),
                default=[9, 11, 13, 15, 17],
                help="Run the full analysis at these window sizes."
            )
        with col_b:
            test_global_k = st.multiselect(
                "Test Global macro-clusters (k)",
                options=[2, 3, 4, 5, 6],
                default=[2, 3, 4],
                help="Only applies when using automatic Global clustering."
            )

        max_runs = st.slider("Maximum combinations to run", 4, 25, 12, help="Higher = more thorough but slower.")

        if st.button("🚀 Run Sensitivity Analysis", type="secondary", width="stretch"):
            if not current_value_cols:
                st.warning("Run a main analysis first (or select value columns) so we know which variables to use.")
            else:
                with st.spinner("Running parameter sweep... this can take a while"):
                    try:
                        sens_report = run_parameter_sensitivity(
                            st.session_state.df_raw,
                            value_cols=current_value_cols,
                            base_window_size=st.session_state.get("last_window", 13),
                            base_medium_groups=analysis.get("config", {}).get("medium_groups") if analysis else None,
                            base_global_groups=analysis.get("config", {}).get("global_groups") if analysis else None,
                            base_auto_global=analysis.get("config", {}).get("auto_global", True) if analysis else True,
                            window_sizes=test_windows or [13],
                            global_cluster_counts=test_global_k or [3],
                            max_combinations=max_runs,
                        )
                        st.session_state.sensitivity_report = sens_report
                        st.success(f"Sensitivity complete: {len(sens_report.results)} runs.")
                    except Exception as e:
                        st.error(f"Sensitivity run failed: {e}")
                        st.exception(e)

        # Display previous sensitivity results
        if st.session_state.get("sensitivity_report"):
            sens: SensitivityReport = st.session_state.sensitivity_report
            st.markdown("#### Sensitivity Results")

            # Summary
            if sens.summary:
                st.info(sens.summary)

            # Stability table
            stab = sens.stability
            if stab:
                stab_rows = []
                for sc in ["Local", "Medium", "Global"]:
                    ts = stab.get(f"{sc}_t_star")
                    if ts:
                        stab_rows.append({
                            "Scale": sc,
                            "t* median": ts.get("median"),
                            "t* std": ts.get("std"),
                            "Stable (±3 steps)": f"{ts.get('stability_score_within_3', 0)*100:.0f}%",
                            "t* range": f"{ts.get('min')}–{ts.get('max')}",
                        })
                if stab_rows:
                    st.dataframe(pd.DataFrame(stab_rows), width="stretch")

            # Detailed table of all runs
            if sens.results:
                rows = []
                for r in sens.results:
                    if r.success:
                        for sc in ["Local", "Medium", "Global"]:
                            m = r.metrics.get(sc, {})
                            rows.append({
                                "window": r.params.get("window_size"),
                                "global_k": r.params.get("global_num_clusters"),
                                "scale": sc,
                                "t*": m.get("t_star"),
                                "mean_τ": round(m.get("mean_tau", float("nan")), 4),
                                "RECD_final": round(m.get("recd_T_final", float("nan")), 4),
                            })
                if rows:
                    sens_df = pd.DataFrame(rows)
                    st.dataframe(sens_df, width="stretch", height=280)

            if st.button("Clear Sensitivity Results", key="clear_sens"):
                st.session_state.pop("sensitivity_report", None)
                st.rerun()
else:
    if not HAS_SENSITIVITY:
        st.caption("Sensitivity module not available in this build.")

# Insight
st.markdown("### Ontological Interpretation")
insight_html = generate_ontological_insight(scale_metrics, scale_results=scale_results, perspective=active_view)
st.markdown(f'<div class="insight-box">{insight_html}</div>', unsafe_allow_html=True)

st.divider()

# =============================================================================
# High-Quality Visualizations
# =============================================================================
st.markdown("### Publication-Ready Visualizations")

if len(analyses) > 1:
    # Pestañas separadas por perspectiva (user request)
    perspective_viz_tabs = st.tabs([f"{v} visualizations" for v in analyses.keys()])
    for ptab, (pview, pan) in zip(perspective_viz_tabs, analyses.items()):
        with ptab:
            psr = pan.get("scale_results", {})
            psm = pan.get("scale_metrics", {})
            st.caption(f"Visualizations for the **{pview}** perspective (modules = {'locations' if pview=='spatial' else 'variables' if pview=='multivariate' else 'custom'})")
            ptab1, ptab2, ptab3, ptab4 = st.tabs(["τ Trajectories", "Ontological Decay", "RECD", "Per-Scale Detail"])
            with ptab1:
                fig = plot_multi_scale_tau_trajectories(psr)
                st.pyplot(fig, width="stretch")
            with ptab2:
                fig = plot_ontological_decay(psm)
                st.pyplot(fig, width="stretch")
            with ptab3:
                fig = plot_recd_discretization(psr)
                st.pyplot(fig, width="stretch")
            with ptab4:
                sel = st.selectbox("Scale", ["Local","Medium","Global"], key=f"sel_{pview}")
                if sel in psr:
                    st.pyplot(plot_scale_detail(sel, psr[sel]), width="stretch")
    st.markdown("---")
    st.caption("Above: separate visualization tabs per perspective. Below: visualizations for the currently active view (switch in section 1b).")

# Normal visualizations for the active / current selection
tab1, tab2, tab3, tab4 = st.tabs(["τ_s Trajectories", "Ontological Decay", "RECD Discretization", "Per-Scale Detail"])

with tab1:
    fig_tau = plot_multi_scale_tau_trajectories(scale_results)
    st.pyplot(fig_tau, width="stretch")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("📥 Download PNG (τ trajectories)", key="dl_tau_png"):
            data = export_figure(fig_tau, fmt="png", dpi=400)
            saved_path = _save_to_downloads(data, "tau_trajectories.png")
            st.success(f"✅ PNG saved to: {saved_path}")
            if st.button("📂 Reveal in Finder", key="reveal_png"):
                _reveal_in_finder(saved_path)
    with c2:
        if st.button("📥 Download SVG", key="dl_tau_svg"):
            data = export_figure(fig_tau, fmt="svg")
            saved_path = _save_to_downloads(data, "tau_trajectories.svg")
            st.success(f"✅ SVG saved to: {saved_path}")
            if st.button("📂 Reveal in Finder", key="reveal_svg"):
                _reveal_in_finder(saved_path)
    with c3:
        if st.button("📥 Download PDF", key="dl_tau_pdf"):
            data = export_figure(fig_tau, fmt="pdf", dpi=300)
            saved_path = _save_to_downloads(data, "tau_trajectories.pdf")
            st.success(f"✅ PDF saved to: {saved_path}")
            if st.button("📂 Show in Finder", key="reveal_tau_pdf"):
                _reveal_in_finder(saved_path)

with tab2:
    fig_decay = plot_ontological_decay(scale_metrics)
    st.pyplot(fig_decay, width="stretch")
    if st.button("📥 Download high-res decay plot (PDF)"):
        data = export_figure(fig_decay, fmt="pdf", dpi=350)
        saved_path = _save_to_downloads(data, "decay_ontological.pdf")
        st.success(f"✅ PDF saved to: {saved_path}")
        if st.button("📂 Show in Finder", key="reveal_decay_pdf"):
            _reveal_in_finder(saved_path)

with tab3:
    fig_recd = plot_recd_discretization(scale_results)
    st.pyplot(fig_recd, width="stretch")
    if st.button("📥 Download RECD (PDF)"):
        data = export_figure(fig_recd, fmt="pdf", dpi=350)
        saved_path = _save_to_downloads(data, "recd_discretization.pdf")
        st.success(f"✅ PDF saved to: {saved_path}")
        if st.button("📂 Show in Finder", key="reveal_recd_pdf"):
            _reveal_in_finder(saved_path)

with tab4:
    sel_scale = st.selectbox("Select scale for detailed view", ["Local", "Medium", "Global"], index=0)
    if sel_scale in scale_results:
        fig_detail = plot_scale_detail(sel_scale, scale_results[sel_scale])
        st.pyplot(fig_detail, width="stretch")
        if st.button(f"📥 Export {sel_scale} detail (high-res PDF)"):
            data = export_figure(fig_detail, fmt="pdf", dpi=350)
            saved_path = _save_to_downloads(data, f"{sel_scale.lower()}_detail.pdf")
            st.success(f"✅ PDF saved to: {saved_path}")
            if st.button("📂 Show in Finder", key=f"reveal_{sel_scale.lower()}_pdf"):
                _reveal_in_finder(saved_path)

st.divider()

# =============================================================================
# 4. Ontological What-If Simulator
# =============================================================================
if not _has_results:
    pass
else:
    view_for_whatif = active_view or "current"
    st.markdown(f"## 4. Ontological What-If Simulator — on **{view_for_whatif}** view")

st.caption(
    "Experiment with alternative ways of grouping your chosen modules. The what-if is always performed on the modules of the currently active view. "
    "You can switch views (Spatial vs Multivariate) above, run a new main analysis, and then run what-if for that perspective."
)

st.info(
    "**If you are seeing 'Scale ... requires at least 2 modules' for Hypothetical-Global right now:** "
    "Stop the Streamlit command completely (Ctrl+C) and restart it. "
    "The Python process must reload the fixed code. A browser refresh is not enough."
)

st.markdown("**Define your hypothetical Medium clusters** (pick different modules for each box):")

# ROBUST UI: always render the same 4 stable multiselects using the *full* list of options.
# Previous dynamic "rem" logic caused selections for clusters 2/3/4 to be ignored or lost.
# Now each box always offers all modules; we clean overlaps afterwards (first cluster wins).
_hypo_raw: Dict[str, List[str]] = {}
for ii in range(1, 5):
    # Try to give non-overlapping sensible defaults for the first two clusters on initial load
    n = len(value_cols)
    if ii == 1:
        d = value_cols[: max(1, n // 2)] if n >= 2 else value_cols
    elif ii == 2:
        d = value_cols[max(1, n // 2):] if n >= 2 else []
    else:
        d = []
    sel = st.multiselect(
        f"Hypothetical cluster {ii}",
        options=value_cols,
        default=d,
        key=f"hypo_{ii}",
        help="Select a non-empty subset of modules for this hypothetical medium cluster."
    )
    if sel:
        _hypo_raw[f"Hypo_{ii}"] = list(sel)

# Clean overlaps: first declared cluster keeps a variable
assigned = set()
hypo_groups: Dict[str, List[str]] = {}
for name, members in _hypo_raw.items():
    clean = [m for m in members if m in value_cols and m not in assigned]
    if clean:
        hypo_groups[name] = clean
        assigned.update(clean)

n_hypo = len(hypo_groups)
assigned_count = len(assigned)

# Very clear feedback
if n_hypo < 2:
    st.warning(
        f"**You currently have only {n_hypo} valid hypothetical cluster(s)** (with {assigned_count} unique modules assigned).\n\n"
        "To run the simulation you need **at least 2 clusters**. "
        "Systemic Tau (τ_s) is a measure of *coupling between* series — a single group gives nothing to correlate.\n\n"
        "👉 Assign variables to at least **Hypothetical cluster 1** and **Hypothetical cluster 2** (use the boxes above)."
    )
else:
    # Nice summary of what will be used
    summary = "  •  ".join(f"**{k}** ({len(v)} vars)" for k, v in hypo_groups.items())
    st.success(f"Ready: {n_hypo} hypothetical medium clusters → {summary}. Global will always use ≥2 macro aggregates.")

if st.button("Simulate hypothetical ontological structure", disabled=(n_hypo < 2), type="primary"):
    with st.spinner("Re-aggregating according to your structure and recomputing Systemic Tau + RECD..."):
        try:
            hypo = simulate_what_if(df, value_cols, hypo_groups, window_size=window_size)
            st.session_state.hypo_results = hypo
            st.session_state.last_hypo_groups = hypo_groups
            st.session_state.last_hypo_view = active_view
            st.success(f"Hypothetical structure simulated on the current view ({active_view or 'active'}).")
        except Exception as e:
            err_msg = str(e)
            st.error("**Simulation failed**")
            st.markdown(
                f"""
**What happened:**  
The What-If simulator could not build a valid Global scale (>= 2 modules) from the hypothetical clusters you defined.

**Raw technical detail:**  
`{err_msg}`

**What to do:**
- Make sure the green "Ready" box above says at least 2 clusters with distinct variables.
- You are currently working on the **{active_view or 'active'}** view.
- Switch views in section 1b if you want to test the other perspective.
"""
            )
            with st.expander("Groups sent to engine"):
                st.json(hypo_groups if hypo_groups else {"(none)": ""})

if "hypo_results" in st.session_state and st.session_state.hypo_results:
    st.markdown(f"**Hypothetical vs Real Global — on the {active_view or 'current'} view**")
    st.caption("What-if always uses the modules from the active view you selected above.")
    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Real Global** (from main analysis)")
        if "Global" in scale_results:
            f = plot_scale_detail("Global", scale_results["Global"])
            st.pyplot(f, width="stretch")
    with colB:
        st.markdown("**Hypothetical Global**")
        if "Global" in st.session_state.hypo_results:
            f = plot_scale_detail("Hypothetical-Global", st.session_state.hypo_results["Global"])
            st.pyplot(f, width="stretch")

# Cautious Cross-View What-If (user request)
prepared_views = st.session_state.get("prepared_views", {})
if (len(prepared_views) > 1 and 
    "last_hypo_groups" in st.session_state and 
    st.session_state.get("last_hypo_view")):
    last_view = st.session_state.last_hypo_view
    other_candidates = [v for v in prepared_views if v != last_view]
    if other_candidates:
        other_v = other_candidates[0]
        if st.button(f"Cross-view: Apply analogous grouping from '{last_view}' to '{other_v}'", 
                     help="Uses proportional split sizes from your last what-if groups on the columns of the other perspective. This lets you explore similar organizational hypotheses across different module definitions."):
            try:
                other_df = prepared_views[other_v]
                other_all_cols = [c for c in other_df.select_dtypes(include=[np.number]).columns 
                                  if not any(t in str(c).lower() for t in ("time","week","epi","date","index"))]
                last_groups = st.session_state.last_hypo_groups
                total_size = sum(len(m) for m in last_groups.values()) or 1
                translated = {}
                idx = 0
                for gname, mems in last_groups.items():
                    sz = max(1, int(round(len(other_all_cols) * len(mems) / total_size)))
                    translated[f"From_{last_view}_{gname}"] = other_all_cols[idx:idx+sz]
                    idx += sz
                if len(translated) < 2 and len(other_all_cols) >= 2:
                    h = len(other_all_cols) // 2
                    translated = {"Cross_A": other_all_cols[:h], "Cross_B": other_all_cols[h:]}
                cross_res = simulate_what_if(other_df, other_all_cols, translated, window_size=window_size)
                st.session_state.cross_hypo = {"target_view": other_v, "groups": translated, "results": cross_res}
                st.success(f"Cross simulation completed targeting the {other_v} view.")
            except Exception as ex:
                st.error(f"Cross-view simulation error: {ex}")

if st.session_state.get("cross_hypo"):
    ch = st.session_state.cross_hypo
    st.markdown(f"**Cross-View Hypothetical (groups from {st.session_state.get('last_hypo_view')} applied to {ch['target_view']})**")
    if "Global" in ch.get("results", {}):
        st.pyplot(plot_scale_detail("Hypothetical-Global", ch["results"]["Global"]), width="stretch")

st.divider()

# =============================================================================
# Export & Reporting
# =============================================================================
st.markdown("## 5. Export & Reporting")

df_raw_for_export = st.session_state.get("df_raw")

# PDF Report - use direct file save (avoids pywebview replacing the UI with the PDF)
if st.button("📄 Generate & Save Full Report (multi-page PDF)", width="stretch"):
    with st.spinner("Assembling publication-quality multi-perspective report..."):
        try:
            views_to_report = dict(analyses) if analyses else ({active_view: analysis} if analysis else {})
            raw_shape = df_raw_for_export.shape if df_raw_for_export is not None else None
            ws = window_size

            try:
                sens_report = st.session_state.get("sensitivity_report")
                pdf_bytes = create_report_pdf(
                    views_to_report,
                    raw_shape=raw_shape,
                    window_size=ws,
                    sensitivity_report=sens_report,
                )
            except TypeError as te:
                if "unexpected keyword argument" in str(te) or "got an unexpected keyword" in str(te).lower():
                    st.warning(
                        "Old cached version of report function detected. "
                        "Please fully stop Streamlit (Ctrl+C) and run: rm -rf src/systemictau/studio/__pycache__ "
                        "then restart for the full multi-perspective report."
                    )
                    first = next(iter(views_to_report.values()), {})
                    sr = first.get("scale_results", {})
                    sm = first.get("scale_metrics", {})
                    it = first.get("insight_text")
                    pdf_bytes = create_report_pdf(sr, sm, insight_text=it)
                else:
                    raise

            saved_path = _save_to_downloads(pdf_bytes, "systemic_tau_multi_perspective_report.pdf")
            st.success(f"✅ Full report saved successfully!")
            st.code(saved_path, language=None)
            st.caption("The PDF is now in your Downloads / SystemicTauStudio folder.")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("📂 Reveal in Finder", key="reveal_full_report"):
                    _reveal_in_finder(saved_path)
            with col2:
                if st.button("📁 Open Downloads folder", key="open_downloads"):
                    try:
                        import subprocess
                        subprocess.run(["open", str(Path.home() / "Downloads")], check=False)
                    except Exception:
                        pass

        except Exception as e:
            st.error(f"Report generation failed: {e}")

st.divider()

# Structured Exports (JSON + Excel)
st.markdown("**Structured Data Export** (machine-readable)")

col_json, col_excel = st.columns(2)

with col_json:
    if HAS_EXPORTER:
        if st.button("📤 Export Structured Results (JSON)", width="stretch"):
            try:
                json_bytes = export_to_json(
                    analyses=analyses or ({"active": analysis} if analysis else {}),
                    df_raw=df_raw_for_export,
                    active_view=active_view,
                    window_size=window_size,
                )
                saved_path = _save_to_downloads(json_bytes, f"systemic_tau_structured_{dt.datetime.now().strftime('%Y%m%d_%H%M')}.json")
                st.success(f"✅ JSON saved to: {saved_path}")
                if st.button("📂 Reveal in Finder", key="reveal_structured_json"):
                    _reveal_in_finder(saved_path)
            except Exception as e:
                st.error(f"JSON export failed: {e}")
    else:
        st.caption("Structured exporter not available.")

with col_excel:
    if HAS_EXPORTER:
        if st.button("📊 Export Structured Results (Excel)", width="stretch"):
            try:
                excel_bytes = export_to_excel(
                    analyses=analyses or ({"active": analysis} if analysis else {}),
                    df_raw=df_raw_for_export,
                    active_view=active_view,
                    window_size=window_size,
                    include_raw_data=True,
                )
                saved_path = _save_to_downloads(excel_bytes, f"systemic_tau_structured_{dt.datetime.now().strftime('%Y%m%d_%H%M')}.xlsx")
                st.success(f"✅ Excel saved to: {saved_path}")
                if st.button("📂 Reveal in Finder", key="reveal_structured_excel"):
                    _reveal_in_finder(saved_path)
            except Exception as e:
                st.error(f"Excel export failed: {e}")
    else:
        st.caption("Structured exporter not available.")

# Quick preview of what will be exported
if analyses or analysis:
    with st.expander("Preview export structure", expanded=False):
        try:
            preview = build_structured_export(
                analyses=analyses or ({"active": analysis} if analysis else {}),
                df_raw=df_raw_for_export,
                active_view=active_view,
                window_size=window_size,
            )
            st.json({k: (v if k != "results" else {vk: list(vv.get("scales", {}).keys()) for vk, vv in v.items()})
                     for k, v in preview.items() if k in ("metadata", "results")}, expanded=False)
        except Exception:
            st.write("Preview unavailable")

st.caption(
    "JSON contains full metrics + arrays + cluster composition. "
    "Excel (improved): Summary → Cluster_Composition → Metrics_by_Scale → "
    "spatial_Local/Medium/Global + multivariate_* sheets (with header metrics + Time_Step series) + Raw_Data."
)

st.divider()

# Footer / philosophy note
st.markdown(
    "<div style='color:#666; font-size:0.82rem; text-align:center; padding-top:1rem'>"
    "Systemic Tau Studio — Treating ontological level as a first-class analytical dimension. "
    "Built for researchers who need to understand how structure and temporal organization co-evolve across scales."
    "</div>",
    unsafe_allow_html=True,
)


def main():
    """Entry point for console script."""
    # Streamlit apps are run via `streamlit run`, this is a no-op placeholder
    pass


if __name__ == "__main__":
    # Streamlit apps should be launched with `streamlit run`.
    # This block is intentionally quiet to avoid spam on reruns.
    pass
