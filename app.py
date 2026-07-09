import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import base64
import io
import pickle
import plotly.graph_objects as go

import systemictau as systau
import importlib
importlib.reload(systau.report)
importlib.reload(systau.analysis)
import systemictau.panel
importlib.reload(systemictau.panel)
import systemictau.visualization.monumental_viz
importlib.reload(systemictau.visualization.monumental_viz)
from systemictau import run_full_analysis, prepare_multivariate_timeseries
from systemictau.report import generate_academic_report
HAS_REPORTS = True

from engine.scales import OntologicalScaleManager
from engine.diagnostics import compute_ews
from systemictau.visualization.viz import plot_scale_detail
from systemictau.visualization.monumental_viz import plot_3d_strange_attractor, plot_discretization_manifold, plot_topological_heatmap, plot_ews_dashboard, plot_dyadic_cascade, plot_unimodal_return_map
from utils.export_pdf import convert_markdown_to_pdf

st.set_page_config(
    page_title="Systemic Tau Paradigm", 
    page_icon="🧬", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/johelpadilla/systemictau-web',
        'Report a bug': "https://github.com/johelpadilla/systemictau-web/issues",
        'About': """
# Systemic Tau Paradigm 🧬
**Version 5.6.0 (The Synthesis Release)**

An advanced topological time-series engine designed to detect **Early Warning Signals (EWS)** of systemic collapse in complex non-linear environments. 

**Core Technologies:**
- Persistent Homology (Algebraic Topology)
- Symbolic Transfer Entropy (Ordinal Memory)
- Adaptive Temporal Breathing Windows

*Theoretical Framework based on the Magna Synthesis v6 (Padilla, 2026).*
        """
    }
)

def inject_custom_css():
    st.markdown("""
    <style>
    /* Global Typography */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Roboto', sans-serif;
    }
    
    /* Header Gradient */
    h1 {
        background: -webkit-linear-gradient(45deg, #1E88E5, #8E24AA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        letter-spacing: -1px;
    }
    
    /* Tabs styling (Premium Pills) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: pre-wrap;
        background-color: #FFFFFF;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        color: #64748B;
        font-weight: 600;
        padding: 0 16px;
        transition: all 0.2s ease-in-out;
    }
    .stTabs [data-baseweb="tab"]:hover {
        border-color: #1E88E5;
        color: #1E88E5;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E88E5 !important;
        color: #FFFFFF !important;
        border-color: #1E88E5 !important;
        box-shadow: 0 4px 6px -1px rgba(30, 136, 229, 0.2), 0 2px 4px -1px rgba(30, 136, 229, 0.1) !important;
    }
    /* Hide the default animated underline */
    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        border-color: #1E88E5;
        color: #1E88E5;
    }
    .stDownloadButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stDownloadButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
    
    /* Metric Cards */
    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        transition: all 0.2s ease;
    }
    [data-testid="stMetric"]:hover {
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        border-color: #1E88E5;
    }
    [data-testid="stMetricValue"] {
        color: #1E293B;
        font-weight: 700;
    }
    
    /* Info/Success/Warning Cards (Glassmorphism / Shadow) */
    div[data-testid="stAlert"] {
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -1px rgba(0,0,0,0.03);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        border-right: 1px solid #E2E8F0;
        background-color: #FFFFFF;
    }
    
    /* DataFrame Header */
    thead tr th {
        background-color: #F1F5F9 !important;
        color: #334155 !important;
        font-weight: 600 !important;
    }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# ==============================================================================
# CACHED COMPUTATION ENGINE (PERFORMANCE LEAP)
# ==============================================================================
@st.cache_data(show_spinner=False)
def generate_macro_clusters(df_numeric, cols, num_clusters, agg_method):
    return OntologicalScaleManager.get_global_clusters(
        df_numeric, cols, method="auto", num_clusters=num_clusters, agg_method=agg_method
    )

@st.cache_data(show_spinner=False)
def run_core_math(X, window_size, component_names=None, adaptive_breathing=False, compute_tda=False, compute_ordinal=False, compute_nested_recd=False, **kwargs):
    """
    v5.6.0: Added adaptive_breathing flag and compute_tda flag.
    v5.1.0: Added compute_ordinal flag.
    """
    if adaptive_breathing:
        st.info("🫁 **Adaptive Breathing Window (W_t) engaged** — Dynamic regime-aware window sizing is active. The engine will automatically expand $W_t$ in hyper-persistent regimes and contract it during chaotic periods.")
    
    res_obj = run_full_analysis(X, window_size=window_size, component_names=component_names, adaptive_breathing=adaptive_breathing, compute_tda=compute_tda, compute_ordinal=compute_ordinal, compute_nested_recd=compute_nested_recd, **kwargs)
    
    # Check for warnings
    if res_obj.warnings:
        for w in res_obj.warnings:
            st.warning(f"Engine Validation: {w}")
            
    # Bridge to AnalysisResults dictionary format expected by reports and plots
    mean_dtk_open = float(np.nanmean(res_obj.dtk_series[res_obj.dtk_series > 0])) if len(res_obj.dtk_series) > 0 and np.any(res_obj.dtk_series > 0) else 0.0
    res_dict = {
        "X": res_obj.X,
        "taus_global": res_obj.taus_global,
        "taus_per_module": getattr(res_obj, "taus_per_module", np.array([])),
        "T_series": res_obj.T_series,
        "dtk_series": getattr(res_obj, "dtk_series", np.array([])),
        "dyadic_k_series": getattr(res_obj, "dyadic_k_series", np.array([])),
        "estimated_period_series": getattr(res_obj, "estimated_period_series", np.array([])),
        "gate_series": np.array([]),
        "depths": np.array([]),
        "episodes": res_obj.episodes,
        "t_frob": res_obj.t_frob,
        "t_ks": res_obj.t_ks,
        "t_star": res_obj.t_star,
        "max_dist": res_obj.frob_max if res_obj.frob_max is not None else 0.0,
        "max_ks": res_obj.ks_max if res_obj.ks_max is not None else 0.0,
        "fractal_D": res_obj.fractal_D,
        "metadata": res_obj.metadata,
        "figures": None,
        "tda_results": getattr(res_obj, "tda_results", None),
        "ordinal_results": getattr(res_obj, "ordinal_results", None),
        "nested_recd_results": getattr(res_obj, "nested_recd_results", None),
        "nonlinear_stats": {
            "hp_z": res_obj.hp_z,
            "laminarity": res_obj.lam,
            "trapping_time": res_obj.tt,
            "max_M": res_obj.M_max,
            "mean_M": res_obj.M_mean,
            "mean_dtk_open": mean_dtk_open
        },
        "episodes_table": res_obj.episodes,
        "warnings": res_obj.warnings
    }
    return res_dict, res_obj

@st.cache_data(show_spinner=False)
def get_ews(taus_global, window_size):
    return compute_ews(taus_global, window_size)

# ==============================================================================
# STATE MANAGEMENT & SERIALIZATION
# ==============================================================================
def export_workspace():
    state_dict = {
        "raw_df": st.session_state.get("raw_df"),
        "clean_df": st.session_state.get("clean_df"),
        "data_cols": st.session_state.get("data_cols", []),
        "time_cols": st.session_state.get("time_cols", []),
        "scale_dict": st.session_state.get("scale_dict", {}),
        "clustered_df": st.session_state.get("clustered_df"),
        "analysis_results": st.session_state.get("analysis_results"),
        "raw_results": st.session_state.get("raw_results"),
        "ews_results": st.session_state.get("ews_results")
    }
    return pickle.dumps(state_dict)

def import_workspace(file_bytes):
    state_dict = pickle.loads(file_bytes)
    for key, value in state_dict.items():
        st.session_state[key] = value

AUTOSAVE_PATH = "/tmp/systemictau_autosave.tau"

def autosave_workspace():
    try:
        with open(AUTOSAVE_PATH, "wb") as f:
            f.write(export_workspace())
    except Exception as e:
        print(f"Autosave failed: {e}")

def load_autosave():
    if os.path.exists(AUTOSAVE_PATH):
        try:
            with open(AUTOSAVE_PATH, "rb") as f:
                import_workspace(f.read())
            return True
        except:
            return False
    return False

if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "clean_df" not in st.session_state:
    st.session_state.clean_df = None
if "data_cols" not in st.session_state:
    st.session_state.data_cols = []
if "time_cols" not in st.session_state:
    st.session_state.time_cols = []
if "scale_dict" not in st.session_state:
    st.session_state.scale_dict = {}
if "clustered_df" not in st.session_state:
    st.session_state.clustered_df = None
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "ews_results" not in st.session_state:
    st.session_state.ews_results = None

# Sidebar
with st.sidebar:
    st.image("logo.jpg", width="stretch")
    st.markdown("<h3 style='text-align: center;'>Systemic Tau 🧬 v5.6.1</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888;'>Adaptive • Topological • Ordinal Memory</p>", unsafe_allow_html=True)
    
    with st.expander("⚙️ Core Settings", expanded=True):
        window_size = st.slider(
            "Window Size (n)", 
            min_value=5, max_value=150, value=13, step=1, 
            help="**What is the Sliding Window?**\nIt is the number of consecutive temporal observations (t) evaluated simultaneously to calculate the level of systemic entanglement.\n\n*   **Short Windows (e.g., 5-10):** Capture ultra-fast dynamics and abrupt changes, but are highly susceptible to statistical 'noise'.\n*   **Long Windows (e.g., 20+):** Smooth the signal revealing the macroscopic trend, but mathematically *delay* the detection of Early Warning Signals (EWS).\n*Recommendation*: 13 is an empirically proven Pareto optimum in biological and financial cycles."
        )
        
        # v5.6.0 NEW: Adaptive Breathing Window (Urgent #1)
        adaptive_breathing = st.checkbox(
            "🫁 Adaptive Breathing Window (W_t) — v5.6", 
            value=True,
            help="""**Dynamic and Adaptive Evaluation Window**
            
    In systems with **hyper-persistence** (e.g., dengue environmental variables in San Juan, chaotic streaks of 400-800+ weeks), a fixed window generates:
    • Too much noise in long stable periods, or
    • Excessive delay near transitions.

    **How it works (Breathing W_t):**
    1. First pass with base window → calculates τ_s and hyper-persistence streaks.
    2. Automatically detects the local length of regimes (using run-length of |τ_s| < 0.41 and local variance).
    3. Dynamically adjusts W_t:
       - **Long (↑)** in prolonged hyper-persistence → more smoothing, fewer false alarms.
       - **Short (↓)** near transitions or high local volatility → greater temporal sensitivity.
    4. Re-calculates key metrics with the adapted window per regime.

    This immediately improves: Layer 1 detectors (hp_z, Joint Episodes), Layer 3 (t*), and EWS. It is the highest impact, lowest risk improvement for your daily workflow."""
        )
        st.session_state.adaptive_breathing = adaptive_breathing
    
    with st.expander("🧬 Extended Features", expanded=False):
        compute_tda = st.checkbox(
            "🌐 Geospatial Topology (TDA)", 
            value=True,
            help="""**Topological Data Analysis (Persistent Homology)**
            
    Computes Betti Numbers ($H_0$, $H_1$) over time to detect the formation and collapse of structural 'holes' or correlation cycles in the network.
    - Uses $D = 1 - |\tau|$ as distance metric.
    - Includes Total Persistence, Max Persistence, and Persistence Entropy of $H_1$.
    - Calculates in 'fast' mode (stride = 5) to save computation.
    Requires the 'ripser' library."""
        )
        st.session_state.compute_tda = compute_tda
        
        compute_ordinal = st.checkbox(
            "🧠 Ordinal Memory (STE)", 
            value=True,
            help="""**Ordinal Memory / Transfer Entropy**
            
    Computes Rank Mutual Information (Lite) or Symbolic Transfer Entropy (Full) to detect directionality and non-linear memory in the system.
    - Lite Mode measures average non-linear coupling.
    - Full Mode measures directed information flow (O(N²) complexity)."""
        )
        st.session_state.compute_ordinal = compute_ordinal
        
        compute_nested_recd = st.checkbox(
            "🕰️ Nested RECD (Extramental Clock)", 
            value=True,
            help=r"""**Nested Ordinal RECD**
            
    Computes nested ordinal conjunction levels ($\Phi_1$, $\Phi_2$, $\Phi_3$) and the $\lambda$-weighted Discrete Extramental Clock.
    - Extracts deep structural memory and synergistic patterns before critical transitions.
    - Essential for Cardiac and Physiological data mapping."""
        )
        st.session_state.compute_nested_recd = compute_nested_recd
    
    with st.expander("⚙️ Advanced Parameters"):
        st.markdown("<small>Probabilistic Filters</small>", unsafe_allow_html=True)
        theta_A = st.number_input(
            "Magnitude Threshold (θ_A)", 
            value=0.05, step=0.01, 
            help="**Magnitude Threshold (θ_A)**\nFilters minor fluctuations. Defines the minimum aggressiveness required for a relational change to be cataloged as a 'Systemic Episode' (Critical Lock-in). Historical values like 0.41 represent extreme thermodynamic barriers, while 0.05 is ideal for high sensitivity."
        )
        D_min = st.number_input(
            "Minimum Duration (D_min)", 
            value=10, step=1, 
            help="**Minimum Duration (D_min)**\nTemporal boundary condition. A topological lock-in only acquires mathematical validation if the system remains anchored in hyper-synchronization for at least *D_min* time steps. Acts as a low-pass filter against false positives and transient micro-shocks."
        )
        
        st.markdown("<small>Geospatial TDA Parameters</small>", unsafe_allow_html=True)
        tda_stride = st.number_input("TDA Stride (Windows)", value=5, step=1, help="Number of windows to skip between Persistent Homology computations. Higher is faster.")
        tda_mode = st.selectbox("TDA Mode", ["fast", "full"], help="Fast respects stride, Full computes every window (slow).")
        tda_persistence_threshold = st.number_input("TDA Persistence Threshold", value=0.1, step=0.01, help="Minimum lifespan to consider a topological hole significant.")
        
        st.session_state.tda_stride = tda_stride
        st.session_state.tda_mode = tda_mode
        st.session_state.tda_persistence_threshold = tda_persistence_threshold
        
        st.markdown("<small>Ordinal Memory Parameters</small>", unsafe_allow_html=True)
        ordinal_mode = st.selectbox("Ordinal Mode", ["lite", "full"], help="Lite (Rank MI) is fast. Full (Symbolic Transfer Entropy) is extremely slow and captures directionality.")
        ordinal_stride = st.number_input("Ordinal Stride (Windows)", value=5, step=1, help="Skip windows for faster computation.")
        ordinal_m = st.slider("Embedding Dimension (m)", 2, 5, 3, help="Length of ordinal patterns. Higher m requires exponentially more computation.")
        ordinal_delay = st.slider("Embedding Delay", 1, 5, 1, help="Time delay between points in the pattern.")
        if ordinal_mode == "full" and ordinal_m >= 4:
            st.warning("⚠️ Full Mode with m >= 4 is extremely slow. Ensure stride is high or dataset is small.")
            
        st.session_state.ordinal_mode = ordinal_mode
        st.session_state.ordinal_stride = ordinal_stride
        st.session_state.ordinal_m = ordinal_m
        st.session_state.ordinal_delay = ordinal_delay
        
        st.markdown("<small>Nested RECD Parameters</small>", unsafe_allow_html=True)
        nested_m = st.slider("Nested Embedding Dimension (m)", 2, 5, 3, help="Length of ordinal patterns for Level 1-3 conjunctions.")
        nested_d = st.slider("Alphabet Size (d)", 2, 6, 4, help="Alphabet size for discretization. Typically 4.")
        nested_theta3 = st.number_input("Theta3 (Level 3 Surplus Threshold)", value=0.10, step=0.01, help="Threshold for Level 3 synergistic information surplus.")
        
        st.session_state.nested_m = nested_m
        st.session_state.nested_d = nested_d
        st.session_state.nested_theta3 = nested_theta3
    with st.expander("📝 Report Branding"):
        report_title = st.text_input("Project Title", value="Systemic Tau Analysis")
        report_author = st.text_input("Lead Analyst / Author", value="")
        report_org = st.text_input("Organization", value="")
        
        # Save these to session state so they can be exported/imported and accessed in Tab 5
        st.session_state.report_title = report_title
        st.session_state.report_author = report_author
        st.session_state.report_org = report_org
        
    st.markdown("---")
    st.info("Upload your dataset in the Data Hub to begin analysis.")

    # ------------------ LEGAL DISCLAIMER ------------------
    st.markdown(
        """
        <div style='font-size: 0.75em; color: #666; margin-top: 2em; line-height: 1.2;'>
        <b>Disclaimer:</b> Systemic Tau is designed for academic research and retrospective topological analysis. 
        It does not constitute financial, medical, or life-critical diagnostic advice. 
        Decisions made based on this engine's outputs are at the user's own risk.
        </div>
        """, 
        unsafe_allow_html=True
    )

# Main Layout
tab1, tab2, tab3, tab8, tab9, tab10, tab4, tab7, tab5, tab6 = st.tabs([
    "📊 Data Hub & Health", 
    "🌐 Ontological Scales", 
    "📈 Systemic Analysis", 
    "🗺️ Geospatial Topology",
    "🧠 Ordinal Memory",
    "🕰️ Nested RECD",
    "🌀 EWS & Diagnostics", 
    "🛡️ Robustness & Tiers",
    "💾 Export Center",
    "🔬 Meta-Analysis"
])

# ----------------- TAB 1: DATA HUB & HEALTH -----------------
with tab1:
    st.header("Data Ingestion & Preprocessing")
    st.markdown("The Systemic Tau engine requires unbroken, continuous time-series data without NaNs.")
    
    with st.expander("📂 Load Previous Session (.tau)", expanded=False):
        if os.path.exists(AUTOSAVE_PATH):
            st.info("We detected an unsaved session from your last run.")
            if st.button("🔄 Restore Last Unsaved Session"):
                if load_autosave():
                    st.success("Autosave restored successfully! Go to Tab 3 or 4.")
                else:
                    st.error("Failed to restore autosave.")
        
        tau_file = st.file_uploader("Upload a Systemic Tau workspace file", type=["tau"], key="tau_uploader")
        if tau_file is not None:
            try:
                import_workspace(tau_file.getvalue())
                st.success("✅ Workspace loaded successfully! All your variables, clusters, and mathematical results have been restored.")
                st.info("👉 **Next Step:** You can now click directly on **Tab 3 (Systemic Analysis)** or **Tab 4 (EWS)** to view your charts without recalculating.")
            except Exception as e:
                st.error(f"Error loading workspace: {e}")
                
    st.markdown("---")
    
    col_up, col_tpl = st.columns([2, 1])
    with col_up:
        uploaded_file = st.file_uploader("Upload CSV or Excel Data", type=["csv", "xlsx", "xls"], help="Upload your multivariate time-series data here.")
    with col_tpl:
        st.write("")
        st.write("")
        # Create a tiny mock template
        template_df = pd.DataFrame({
            "Time": ["2023-01", "2023-02", "2023-03", "2023-04"],
            "Variable_A": [12.4, 15.1, 14.2, 18.0],
            "Variable_B": [100, 105, 98, 110],
            "Variable_C": [0.45, 0.52, 0.48, 0.61]
        })
        st.download_button(
            label="⬇️ Download Data Template",
            data=template_df.to_csv(index=False).encode('utf-8'),
            file_name="systemic_tau_template.csv",
            mime="text/csv",
            help="Download a perfectly formatted example CSV file to see how to structure your own data."
        )
        
        st.write("")
        st.markdown("### 🧪 Built-in Test Datasets")
        st.info("**Don't have data on hand?** Explore the potential of the Systemic Tau engine by loading our methodologically validated dataset.")
        
        if st.button("🦠 Load Dataset: Dengue Outbreak (DengAI)"):
            if os.path.exists("data/datos_dengai_completo.csv"):
                st.session_state.raw_df = pd.read_csv("data/datos_dengai_completo.csv")
                st.success("✅ **Dengue Epidemic Dataset loaded.** This dataset contains climate and vegetation variables. Go to 'Data Health' to start preprocessing and then to 'Ontological Scales'.")
                st.rerun()
            else:
                st.error("Example dataset file not found in the 'data/' path.")
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            else:
                try:
                    df = pd.read_csv(uploaded_file)
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding='latin1')
            
            # 1. Structural cleaning: remove completely empty rows/cols
            df.dropna(how='all', axis=1, inplace=True)
            df.dropna(how='all', axis=0, inplace=True)
            
            if df.empty or len(df.columns) == 0:
                st.error("❌ **Critical Error:** The uploaded dataset is completely empty or lacks valid columns.")
            else:
                st.session_state.raw_df = df
                st.success(f"✅ Data loaded successfully! Detected {df.shape[0]} rows and {df.shape[1]} columns.")
                
        except pd.errors.EmptyDataError:
            st.error("❌ **File Empty:** The uploaded CSV file contains no data.")
        except pd.errors.ParserError:
            st.error("❌ **Format Error:** The CSV file appears corrupted or uses an invalid delimiter. Please ensure it is comma-separated.")
        except Exception as e:
            st.error(f"❌ **Unexpected Error:** Failed to read the file. Technical details: {e}")
        
    # Empty State Onboarding if completely empty
    if st.session_state.raw_df is None:
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.info("**Step 1**\n\nUpload a continuous time-series dataset without missing values.")
        c2.info("**Step 2**\n\nScale variables top-down or let the engine cluster them.")
        c3.info("**Step 3**\n\nRun the mathematical engine to detect systemic phase transitions.")
        
    if st.session_state.raw_df is not None:
        df = st.session_state.raw_df
        
        # Select variables
        all_cols = df.columns.tolist()
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Check for temporal panel structure heuristically for defaults
        time_candidates = ['EpiYear', 'EpiWeek', 'Year', 'Week', 'Date', 'Time']
        found_time_cols = [c for c in time_candidates if c in all_cols]
        default_time_cols = found_time_cols[:2] if found_time_cols else []
        
        # Respect loaded session state for time columns
        saved_time = st.session_state.get("time_cols", [])
        if saved_time and all(c in all_cols for c in saved_time):
            default_time_cols = saved_time
        
        st.markdown("### Data Configuration")
        
        time_cols_to_use = st.multiselect(
            "Select Time/Chronological Columns (e.g. Year, Week)", 
            options=all_cols, 
            default=default_time_cols
        )
        st.session_state.time_cols = time_cols_to_use
        
        possible_values = [c for c in num_cols if c not in time_cols_to_use and c.lower() not in ['nid', 'id', 'trap']]
        
        # Respect loaded session state for data columns
        saved_data = st.session_state.get("data_cols", [])
        if saved_data and all(c in num_cols for c in saved_data):
            default_data_cols = saved_data
        else:
            default_data_cols = possible_values[:3] if possible_values else []
            
        data_cols = st.multiselect(
            "Select Value/Component Columns to Analyze", 
            options=num_cols, 
            default=default_data_cols
        )
        st.session_state.data_cols = data_cols
        
        if len(data_cols) > 0:
            for col in data_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df_subset = df[data_cols].copy()
            
            st.markdown("### Data Health & Aggregation Profiler")
            if len(time_cols_to_use) > 0:
                st.info(f"✅ Time columns selected: {time_cols_to_use}. The engine will aggregate panel data chronologically.")
                
                # Allow user to pick aggregation method
                agg_method_target = st.selectbox("Aggregation for main target (first column)", ["sum", "mean", "median", "max"])
                agg_method_covs = st.selectbox("Aggregation for secondary columns", ["mean", "sum", "median", "max"])
                
                if st.button("Aggregate & Prepare Matrix"):
                    try:
                        # Build aggregation dictionary dynamically
                        target_col = data_cols[0]
                        agg_dict = {target_col: agg_method_target}
                        for c in data_cols[1:]:
                            agg_dict[c] = agg_method_covs
                            
                        X_prepared, weekly_df = prepare_multivariate_timeseries(
                            df, 
                            time_cols=time_cols_to_use,
                            value_cols=data_cols,
                            agg=agg_dict,
                            sort=True,
                            dropna=True
                        )
                        st.session_state.clean_df = weekly_df
                        # Update the registered data_cols so downstream tabs only see the aggregated variables
                        st.session_state.data_cols = data_cols
                        # Automatically set clustered_df so the user can skip Tab 2 if they want
                        st.session_state.clustered_df = weekly_df[data_cols]
                        
                        st.success(f"Data aggregated successfully: Reduced panel to a robust mathematical matrix of shape {X_prepared.shape} (T={X_prepared.shape[0]} points).")
                    except Exception as e:
                        st.error(f"Aggregation failed: {e}")
            else:
                nan_count = df_subset.isna().sum().sum()
                if nan_count > 0:
                    st.warning(f"⚠️ Warning: Found {nan_count} missing values (NaNs) in the selected variables. Numba will fail.")
                    impute_method = st.radio("Select Imputation Method", ["Linear Interpolation", "Forward Fill", "Drop Rows with NaNs"])
                    
                    if st.button("Apply Preprocessing"):
                        if impute_method == "Linear Interpolation":
                            df_subset = df_subset.interpolate(method='linear').ffill().bfill()
                        elif impute_method == "Forward Fill":
                            df_subset = df_subset.ffill().bfill()
                        else:
                            df_subset = df_subset.dropna()
                            
                        st.session_state.clean_df = df_subset
                        st.success("Data cleaned and ready for the engine.")
                else:
                    st.success("✅ Data is perfectly clean (0 NaNs detected). Ready for analysis.")
                    st.session_state.clean_df = df_subset
                
            if st.session_state.clean_df is not None:
                # Plot max 5 strictly numeric columns to avoid Streamlit rendering crashes on massive/mixed datasets
                numeric_cols = [c for c in st.session_state.data_cols if c in st.session_state.clean_df.columns]
                cols_to_plot = numeric_cols[:5]
                if len(cols_to_plot) > 0:
                    st.line_chart(st.session_state.clean_df[cols_to_plot].head(500))
                    if len(numeric_cols) > 5:
                        st.caption(f"Showing preview of first 5 variables (out of {len(numeric_cols)}).")

# ----------------- TAB 2: ONTOLOGICAL SCALES -----------------
with tab2:
    st.header("Ontological Scaling")
    st.markdown("Variables must be clustered hierarchically before calculating global tau.")
    
    if st.session_state.clean_df is not None and len(st.session_state.data_cols) > 0:
        # Filter out stale variables from previous sessions that aren't in the new dataset
        valid_cols = [c for c in st.session_state.data_cols if c in st.session_state.clean_df.columns]
        
        if not valid_cols:
            st.warning("No valid numeric columns selected. Please go back to Tab 1 and configure your data.")
        else:
            col1, col2 = st.columns(2)
            
            n_vars = len(valid_cols)
            if n_vars <= 2:
                st.info(f"Only {n_vars} valid variables detected. Clustering is bypassed. You can proceed directly to Tab 3.")
                st.session_state.clustered_df = st.session_state.clean_df[valid_cols]
            else:
                with col1:
                    num_clusters = st.number_input("Number of Macro-Clusters", min_value=2, max_value=n_vars, value=min(3, n_vars))
                with col2:
                    agg_method = st.selectbox("Aggregation Method", ["sum", "median", "mean"])
                    
                with st.spinner("Clustering spatial variables..."):
                    try:
                        clustered_df, target_names, cluster_dict = generate_macro_clusters(
                            st.session_state.clean_df, valid_cols, num_clusters, agg_method
                        )
                        st.session_state.scale_dict = cluster_dict
                        # Only keep the newly generated macro-cluster columns for the math engine
                        st.session_state.clustered_df = clustered_df[target_names]
                        
                        st.success(f"Generated {len(target_names)} macro-clusters successfully.")
                        st.json(cluster_dict)
                    except Exception as e:
                        st.error(f"Clustering error: {e}")
    else:
        st.warning("Please configure and clean variables in the Data Hub first.")

# ----------------- TAB 3: SYSTEMIC ANALYSIS -----------------
with tab3:
    st.header("Systemic Transition Analysis")
    
    if "clustered_df" in st.session_state and st.session_state.clustered_df is not None:
        with st.status("Running full systemic tau mathematical pipeline...", expanded=True) as status:
            try:
                X = st.session_state.clustered_df.values.astype(np.float64)
            except ValueError as ve:
                st.error(f"❌ **Critical Data Error:** Numba requires strictly numeric data. Exception detail: {ve}")
                st.stop()
                
            c_names = st.session_state.clustered_df.columns.tolist()
            st.write("Calculating global taus and scaling constraints...")
            adaptive_flag = st.session_state.get("adaptive_breathing", False)
            tda_flag = st.session_state.get("compute_tda", False)
            ordinal_flag = st.session_state.get("compute_ordinal", False)
            nested_recd_flag = st.session_state.get("compute_nested_recd", False)
            analysis_kwargs = {
                "tda_stride": st.session_state.get("tda_stride", 5),
                "tda_mode": st.session_state.get("tda_mode", "fast"),
                "tda_persistence_threshold": st.session_state.get("tda_persistence_threshold", 0.1),
                "ordinal_stride": st.session_state.get("ordinal_stride", 5),
                "ordinal_mode": st.session_state.get("ordinal_mode", "lite"),
                "ordinal_m": st.session_state.get("ordinal_m", 3),
                "ordinal_delay": st.session_state.get("ordinal_delay", 1),
                "nested_m": st.session_state.get("nested_m", 3),
                "nested_d": st.session_state.get("nested_d", 4),
                "nested_theta3": st.session_state.get("nested_theta3", 0.10)
            }
            res_dict, res_obj = run_core_math(
                X, window_size, 
                component_names=c_names, 
                adaptive_breathing=adaptive_flag, 
                compute_tda=tda_flag, 
                compute_ordinal=ordinal_flag,
                compute_nested_recd=nested_recd_flag,
                **analysis_kwargs
            )
            st.session_state.analysis_results = res_dict
            st.session_state.raw_results = res_obj
            autosave_workspace()
            status.update(label="Systemic Analysis Complete", state="complete", expanded=False)
            
        # Metrics Row
        c1, c2, c3 = st.columns(3)
        c1.metric("Consensus Critical Point (t*)", f"{res_dict['t_star']}")
        c2.metric("Max Systemic Tau", f"{res_dict['max_dist']:.3f}")
        final_recd = res_dict['T_series'][~np.isnan(res_dict['T_series'])][-1] if len(res_dict['T_series']) > 0 else 0
        c3.metric("Final Accumulated RECD", f"{final_recd:.4f}")
        
        st.markdown("### I. Dual-Plot: Systemic Tau vs RECD")
        fig1 = plot_scale_detail("Global Scale", res_dict)
        st.pyplot(fig1)
        
        st.markdown("### II. Systemic Time Manifold (The RECD Staircase)")
        fig2 = plot_discretization_manifold(res_dict)
        st.plotly_chart(fig2, width="stretch")
        
        st.markdown("### III. Topological Heatmap (Spatial Synchronization)")
        if res_dict.get("taus_per_module") is not None:
            fig3 = plot_topological_heatmap(res_dict["taus_per_module"], res_dict.get("t_star", 0))
            st.plotly_chart(fig3, width="stretch")
            
        st.markdown("### IV. Unimodal Return Map (Teorema 24v)")
        st.markdown("Unimodal return map reconstructed from the orbit of $\\tau_s(t)$ vs $\\tau_s(t+1)$. The topology of the system transitions into the central Feigenbaum accumulation region.")
        fig5 = plot_unimodal_return_map(res_dict)
        st.plotly_chart(fig5, width="stretch")
    else:
        st.warning("⚠️ **Action Required:** Please complete Step 1 (Data Hub) and Step 2 (Ontological Scales) before proceeding to Systemic Analysis.")

# ----------------- TAB 8: GEOSPATIAL TOPOLOGY (TDA) -----------------
with tab8:
    st.header("Geospatial Topology (TDA) 🗺️")
    if "analysis_results" not in st.session_state or st.session_state.analysis_results is None:
        st.info("Please run the Systemic Analysis (Tab 3) first to generate topological data.")
    else:
        res_dict = st.session_state.analysis_results
        tda = res_dict.get("tda_results")
        
        if tda is None:
            st.warning("TDA was not computed. Please enable 'Geospatial Topology (TDA)' in the side panel and re-run the global analysis.")
        else:
            st.markdown("### Persistent Homology Metrics ($H_1$)")
            st.markdown("This tracks the formation and collapse of **Topological Holes** (cycles of correlation) across the network.")
            
            c1, c2, c3 = st.columns(3)
            # Find the max value of total persistence
            valid_pers = tda['total_persistence_h1'][~np.isnan(tda['total_persistence_h1'])]
            max_h1_total = np.max(valid_pers) if len(valid_pers) > 0 else 0
            
            # Find systemic t*
            t_star = res_dict.get('t_star')
            
            # Find the time when H1 collapses or peaks
            t_h1_peak = None
            if len(valid_pers) > 0:
                try:
                    t_h1_peak = int(np.nanargmax(tda['total_persistence_h1']))
                except ValueError:
                    t_h1_peak = None
                    
            c1.metric("Peak H1 Total Persistence", f"{max_h1_total:.3f}")
            c2.metric("Systemic $t^*$", f"{t_star}")
            c3.metric("H1 Peak Time", f"{t_h1_peak}" if t_h1_peak is not None else "N/A")
            
            if t_star is not None and t_h1_peak is not None:
                offset = t_star - t_h1_peak
                if offset > 0:
                    st.success(f"**Insight:** The topological holes reached maximum persistence **{offset} steps BEFORE** the systemic transition $t^*$. This suggests network fragmentation preceded the global lock-in.")
                elif offset < 0:
                    st.info(f"**Insight:** The topological holes reached maximum persistence **{abs(offset)} steps AFTER** the systemic transition $t^*$.")
                else:
                    st.warning(f"**Insight:** Maximum topological complexity exactly coincided with the systemic transition.")
            
            st.markdown("### TDA Persistence Curves")
            import plotly.graph_objects as go
            
            fig = go.Figure()
            
            # Add Systemic Tau as a background reference
            tau = res_dict["taus_global"]
            fig.add_trace(go.Scatter(
                y=tau,
                mode='lines',
                name='Systemic Tau',
                line=dict(color='rgba(200, 200, 200, 0.4)', width=3, dash='dot')
            ))
            
            # Add TDA curves
            x_idx = tda['computation_windows']
            y_total = [tda['total_persistence_h1'][i] for i in x_idx]
            y_max = [tda['max_persistence_h1'][i] for i in x_idx]
            y_holes = [tda['n_significant_holes'][i] for i in x_idx]
            y_ent = [tda['persistence_entropy_h1'][i] for i in x_idx]
            
            fig.add_trace(go.Scatter(
                x=x_idx,
                y=y_total,
                mode='lines+markers',
                name='Total Persistence (H1)',
                line=dict(color='#ff4b4b', width=2),
                marker=dict(size=4)
            ))
            
            fig.add_trace(go.Scatter(
                x=x_idx,
                y=y_max,
                mode='lines+markers',
                name='Max Persistence (H1)',
                line=dict(color='#4b4bff', width=2),
                marker=dict(size=4)
            ))
            
            if t_star is not None:
                fig.add_vline(x=t_star, line_width=2, line_dash="dash", line_color="red", annotation_text="t*")
                
            fig.update_layout(
                title="Systemic Transition vs Topological Holes",
                xaxis_title="Time Step (t)",
                yaxis_title="Metric Value",
                height=500,
                template="plotly_white"
            )
            st.plotly_chart(fig, width="stretch")
            
            st.markdown("### Structural Fragmentation (H0 Components) & Entropy")
            
            y_h0_comp = [tda['n_connected_components_h0'][i] for i in x_idx]
            
            fig2 = go.Figure()
            # H0 Components
            fig2.add_trace(go.Scatter(
                x=x_idx,
                y=y_h0_comp,
                mode='lines+markers',
                name='Connected Components (H0)',
                line=dict(color='#00d2d3', width=2),
                yaxis='y1'
            ))
            # Persistence Entropy
            fig2.add_trace(go.Scatter(
                x=x_idx,
                y=y_ent,
                mode='lines+markers',
                name='Persistence Entropy',
                line=dict(color='#ff9f43', width=2),
                yaxis='y2'
            ))
            
            if t_star is not None:
                fig2.add_vline(x=t_star, line_width=2, line_dash="dash", line_color="red", annotation_text="t*")
                
            fig2.update_layout(
                title="Topological Fragmentation vs Chaos",
                xaxis_title="Time Step (t)",
                yaxis=dict(title="H0 Components", side="left", showgrid=False),
                yaxis2=dict(title="Entropy", side="right", overlaying="y", showgrid=False),
                height=400,
                template="plotly_white",
                legend=dict(x=0.01, y=0.99)
            )
            st.plotly_chart(fig2, width="stretch")

# ----------------- TAB 9: ORDINAL MEMORY -----------------
with tab9:
    st.header("Ordinal Memory (Information Flow) 🧠")
    
    if "analysis_results" not in st.session_state or st.session_state.analysis_results is None:
        st.warning("⚠️ **Action Required:** Please run Systemic Analysis first.")
    else:
        res = st.session_state.raw_results
        ordinal = res.ordinal_results
        
        if ordinal is None:
            st.warning("Ordinal Memory was not computed. Please enable 'Ordinal Memory (STE)' in the side panel and re-run the global analysis.")
        else:
            t_star = res.t_star
            x_idx = ordinal['computation_windows']
            y_flow = ordinal['total_flow'][x_idx]
            y_asym = ordinal['net_asymmetry'][x_idx]
            mode = ordinal['mode']
            
            st.markdown(f"**Mode:** {mode.upper()} | **m:** {ordinal['m']} | **delay:** {ordinal['delay']}")
            st.markdown("Higher **Total Flow** indicates strong non-linear coupling. Higher **Net Asymmetry** indicates that specific components are leading the systemic behavior.")
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=x_idx,
                y=y_flow,
                mode='lines+markers',
                name='Total Information Flow',
                line=dict(color='#00d2d3', width=2),
                marker=dict(size=4)
            ))
            
            if mode == 'full':
                fig.add_trace(go.Scatter(
                    x=x_idx,
                    y=y_asym,
                    mode='lines+markers',
                    name='Net Flow Asymmetry',
                    line=dict(color='#ff9f43', width=2),
                    marker=dict(size=4)
                ))
                
            if t_star is not None:
                fig.add_vline(x=t_star, line_width=2, line_dash="dash", line_color="red", annotation_text="t*")
                
            fig.update_layout(
                title=f"Ordinal Memory Dynamics ({mode.upper()})",
                xaxis_title="Time Step (t)",
                yaxis_title="Metric Value",
                height=500,
                template="plotly_white",
                legend=dict(x=0.01, y=0.99)
            )
            st.plotly_chart(fig, width="stretch")

# ----------------- TAB 10: NESTED RECD -----------------
with tab10:
    st.header("Nested RECD (Extramental Clock) 🕰️")
    
    if "analysis_results" not in st.session_state or st.session_state.analysis_results is None:
        st.warning("⚠️ **Action Required:** Please run Systemic Analysis first.")
    else:
        res = st.session_state.raw_results
        recd_res = res.nested_recd_results
        
        if recd_res is None:
            st.warning("Nested RECD was not computed. Please enable 'Nested RECD (Extramental Clock)' in the side panel and re-run the global analysis.")
        else:
            st.markdown("### Extramental Clock & Synergistic Emergence")
            st.markdown("The **Extramental Clock ($T_{recd}$)** is a modified, purely relational timescale driven by ordinal conjunctions. The **Excess3** variable tracks the surplus of Level 3 conjunctions compared to uniform noise, acting as a proxy for complex emergence.")
            
            # Display Dashboard
            from systemictau.visualization.monumental_viz import plot_nested_recd_dashboard
            fig = plot_nested_recd_dashboard(st.session_state.analysis_results)
            if fig is not None:
                st.plotly_chart(fig, width="stretch")

# ----------------- TAB 4: DIAGNOSTICS & EWS -----------------
with tab4:
    st.header("Diagnostics & Early Warning Signals (EWS)")
    
    if st.session_state.analysis_results is not None:
        res = st.session_state.analysis_results
        
        st.subheader("Phase Space (The Strange Attractor)")
        st.markdown("Rotate this 3D projection to observe how the system collapses into the Feigenbaum chaotic boundary.")
        fig_3d = plot_3d_strange_attractor(res)
        st.plotly_chart(fig_3d, width="stretch")
        
        st.markdown("---")
        st.subheader("Early Warning Signals (Critical Slowing Down)")
        with st.status("Calculating AC1, Variance and Skewness...", expanded=True) as status:
            taus_global = res["taus_global"]
            ews = get_ews(taus_global, window_size)
            st.session_state.ews_results = ews
            status.update(label="EWS Calculation Complete", state="complete", expanded=False)
            
        fig_ews = plot_ews_dashboard(res, ews)
        st.plotly_chart(fig_ews, width="stretch")
            
    else:
        st.warning("⚠️ **Action Required:** Please run Systemic Analysis in Step 3 first.")

# ----------------- TAB 5: EXPORT CENTER -----------------
with tab5:
    st.header("Export & Reporting")
    
    if "analysis_results" in st.session_state and st.session_state.analysis_results is not None:
        st.subheader("Workspace Management")
        st.download_button(
            label="💾 Save Workspace Session (.tau)",
            data=export_workspace(),
            file_name="my_analysis.tau",
            mime="application/octet-stream",
            type="primary",
            help="Saves your exact current state (data, clusters, mathematical results) so you can resume later without recalculating."
        )
        
        st.markdown("---")
        
        st.subheader("Data Export")
        export_df = pd.DataFrame({
            "Time_Step": np.arange(len(st.session_state.analysis_results["taus_global"])),
            "Systemic_Tau": st.session_state.analysis_results["taus_global"],
            "Delta_tk": st.session_state.analysis_results["dtk_series"],
            "Accumulated_RECD": st.session_state.analysis_results["T_series"]
        })
        
        if st.session_state.ews_results is not None:
            ews = st.session_state.ews_results
            export_df["EWS_Variance"] = ews["variance"]
            export_df["EWS_AC1"] = ews["ac1"]
            export_df["EWS_Skewness"] = ews["skewness"]
            
        csv = export_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="⬇️ Download Analytical Dataset (CSV)",
            data=csv,
            file_name="systemic_tau_results.csv",
            mime="text/csv",
        )
        
        st.info("The Dual-Plot from Tab 3 can be downloaded by right-clicking it. The interactive 3D/Plotly charts can be saved directly via the camera icon in the top right corner of each plot.")
        
        st.markdown("---")
        st.subheader("Statistical Validation")
        st.markdown(
            "Test the origin of the topological collapse using the **Linear Baseline Test** (optimized IAAFT Surrogate Data). "
            "Instead of merely classifying a result as 'true' or 'false', this tool calculates **how much of your detected transition can be explained purely by individual linear properties** (like autocorrelation and seasonality). "
            "It generates 500 parallel 'null' universes that destroy non-linear topologies while preserving frequency spectra, allowing you to quantify the genuine non-linear systemic component."
        )
        
        if st.button("Run Surrogate Validation (500 universes)", type="primary"):
            # Nivel 1: Detección temprana
            if "analysis_results" not in st.session_state or st.session_state.analysis_results is None:
                st.error("No analysis results found. Run the global pipeline first.")
            else:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    real_max_tau = float(np.nanmax(st.session_state.analysis_results["taus_global"]))
                if np.isnan(real_max_tau):
                    st.error("❌ Cannot run surrogate validation: 'Max Systemic Tau' in real analysis is NaN.")
                    st.info("Possible causes and solutions:")
                    st.markdown("""
                    - Data with NaNs or constant values → go to **Data Hub** and review/impute.
                    - Window size too small for the series length.
                    - Very weak detected transition (t* in an area with almost no variability).
                    - Calculation error in `dtk_series` or `taus_global`.
                    """)
                else:
                    with st.spinner("Generating 500 parallel universes (IAAFT) and running Systemic Tau on each. This is computationally intense..."):
                        try:
                            import systemictau as st_engine
                            if "clustered_df" in st.session_state and st.session_state.clustered_df is not None:
                                X_real = st.session_state.clustered_df.select_dtypes(include=[np.number]).values
                            elif "clean_df" in st.session_state and st.session_state.clean_df is not None:
                                X_real = st.session_state.clean_df.select_dtypes(include=[np.number]).values
                            else:
                                st.error("No valid data loaded. Please upload and prepare data first.")
                                st.stop()
                                
                            surr_res = st_engine.run_surrogate_validation(
                                X_real, 
                                n_surrogates=500, 
                                mode="fast", 
                                window_size=window_size
                            )
                            st.session_state["surrogate_result"] = surr_res
                            st.success("Surrogate validation complete! Results are ready for the Academic Report.")
                        except Exception as e:
                            st.error(f"Error running surrogates: {e}")
                    
        if "surrogate_result" in st.session_state:
            surr = st.session_state["surrogate_result"]
            st.info("### Linear Baseline Test (via IAAFT Surrogates)")
            
            # Nivel 2: Visualización estructurada en Tab 5
            real_max_tau = surr.metadata.get("real_max_tau", float('nan'))
            
            # Robust extraction of surrogate statistics
            surr_mean = surr.metadata.get("surrogate_max_tau_mean", float('nan'))
            surr_std = surr.metadata.get("surrogate_max_tau_std", float('nan'))
            
            # Fallback to direct calculation if metadata is missing or NaN
            if np.isnan(surr_mean) and hasattr(surr, "surrogate_t_stars") and surr.surrogate_t_stars:
                surr_mean = float(np.mean(surr.surrogate_t_stars))
            if np.isnan(surr_std) and hasattr(surr, "surrogate_t_stars") and surr.surrogate_t_stars:
                surr_std = float(np.std(surr.surrogate_t_stars))
            
            if np.isnan(real_max_tau):
                st.error("**Status:** Invalid (NaN detected)\n\nValidation is unreliable because the real Max Systemic Tau is NaN. Review data quality and re-run the main analysis before interpreting surrogates.")
            else:
                linear_contribution = min(100.0, (surr_mean / real_max_tau * 100.0)) if real_max_tau > 0 and not np.isnan(surr_mean) else 100.0
                if surr.percentile_rank < 5:
                    status_color = "red"
                    status_text = "Linear Dominated"
                elif surr.is_significant:
                    status_color = "green"
                    status_text = "Non-Linear Transition Confirmed"
                else:
                    status_color = "orange"
                    status_text = "Mixed/Inconclusive"
                    
                st.markdown(f"#### **Status:** <span style='color:{status_color}'>{status_text}</span>", unsafe_allow_html=True)
                
                has_surr_stats = not np.isnan(surr_mean)
                if has_surr_stats:
                    col1, col2, col3, col4 = st.columns(4)
                else:
                    col1, col2, col3 = st.columns(3)
                    
                with col1:
                    st.metric("Linear Baseline Contribution", f"{linear_contribution:.1f}%", help="Percentage of the observed systemic correlation that can be explained purely by the linear autocorrelation of the variables.")
                with col2:
                    st.metric("Real Max Tau", f"{real_max_tau:.3f}", help="The maximum Systemic Tau observed in the real (non-surrogate) dataset.")
                
                if has_surr_stats:
                    with col3:
                        st.metric("Surrogate Mean (Linear)", f"{surr_mean:.3f}", help="The average maximum Systemic Tau produced by phase-randomized linear surrogates.")
                    with col4:
                        if not np.isnan(surr_std) and surr_std > 0:
                            excess = (real_max_tau - surr_mean) / surr_std
                            st.metric("Excess over Linear (Z)", f"{excess:.2f}", help="Z-score representing how much the real correlation exceeds the linear expectation. Z > 2 indicates significant non-linear/topological coupling.")
                        else:
                            st.metric("Excess over Linear (Z)", "N/A", help="Standard deviation is 0 or missing.")
                else:
                    with col3:
                        st.metric("Percentile", f"{surr.percentile_rank:.2f}%", help="The rank of the real observation among the surrogates. Lower is more significant (e.g., < 5%).")
                
                # Interpretación por casos
                if surr.percentile_rank < 5.0:
                    st.warning("**Interpretation:** ~100% of the magnitude of the detected topological change can be explained by the individual linear properties of the series (autocorrelation and amplitude). This does not invalidate the transition, but indicates that the systemic non-linear coupling component is very weak or not detectable with the current configuration.\n\n**Recommendation:** Activate the 🫁 **Adaptive Breathing Window** (if not already on) and check if one or two highly autocorrelated variables are dominating the signal.")
                elif not surr.is_significant:
                    excess_str = f"{excess:.2f}" if 'excess' in locals() else "N/A"
                    st.info(f"**Interpretation:** The empirical coupling failed to achieve the strict 95% statistical significance threshold against linear models (Percentile: {surr.percentile_rank:.2f}%). While some non-linear structure exists (Excess Z = {excess_str}), it cannot be conclusively differentiated from random phase-shifted noise.")
                else:
                    excess_str = f"{excess:.2f}" if 'excess' in locals() else "N/A"
                    st.success(f"**Interpretation:** The detected transition clearly exceeds what can be explained by linear effects alone (Percentile: {surr.percentile_rank:.2f}%, Excess Z = {excess_str}). There is strong empirical evidence of genuine systemic non-linear topological reorganization.")
            
        st.markdown("---")
        st.subheader("Academic Report Hub")
        if HAS_REPORTS:
            # Add toggle for extended theoretical captions
            include_theory = st.checkbox("Include extended theoretical graph interpretations", value=False, help="Appends rigorous academic interpretations below each figure in the generated PDF/Markdown report based on the Systemic Tau framework.")
            # Add toggle for statistical significance appendix
            include_significance = st.checkbox("Include Statistical Significance Appendix (Appendix B)", value=False, help="Appends an appendix explaining how to interpret the results mathematically without relying on classical p-values.")
            
            @st.cache_data(show_spinner="Compiling Academic Report and PDF (this may take 1-2 minutes due to high-res Plotly rendering)...")
            def compile_report(res_obj_id, inc_theory, inc_sig, title, author, org, has_ews, has_adaptive):
                # We use a primitive signature so caching is lightning fast and bulletproof.
                report_md = generate_academic_report(
                    st.session_state.raw_results, 
                    lang='en', 
                    include_figures=True,
                    include_theoretical_captions=inc_theory,
                    include_significance_appendix=inc_sig,
                    title=title,
                    author=author,
                    organization=org,
                    ews_results=st.session_state.get("ews_results"),
                    surrogate_result=st.session_state.get("surrogate_result"),
                    adaptive_breathing=has_adaptive
                )
                pdf_bytes = convert_markdown_to_pdf(report_md)
                return report_md, pdf_bytes

            # We use the unique memory id of raw_results to know if it changed
            res_id = id(st.session_state.raw_results)
            has_ews = st.session_state.get("ews_results") is not None
            
            try:
                report_md, pdf_bytes = compile_report(
                    res_id, 
                    include_theory, 
                    include_significance, 
                    st.session_state.get("report_title", "Systemic Tau Paradigm Analytical Report"),
                    st.session_state.get("report_author", ""),
                    st.session_state.get("report_org", ""),
                    has_ews,
                    st.session_state.get("adaptive_breathing", True)
                )
                
                st.download_button(
                    label="⬇️ Download Academic Report (.md)",
                    data=report_md,
                    file_name="systemic_tau_academic_report.md",
                    mime="text/markdown",
                    type="primary"
                )
                
                st.download_button(
                    label="📄 Download Academic Report (.pdf)",
                    data=pdf_bytes,
                    file_name="systemic_tau_academic_report.pdf",
                    mime="application/pdf",
                    type="secondary"
                )
                
                with st.expander("Preview Academic Report", expanded=False):
                    st.markdown(report_md)
                    
            except Exception as e:
                st.warning(f"Could not generate Report/PDF: {e}")
        else:
            st.warning(f"The systemictau[reports] module is not installed or available. Error: {REPORT_ERROR}")
            
    else:
        st.warning("⚠️ **Action Required:** No data to export. Please run Systemic Analysis in Step 3 first.")

# ----------------- TAB 6: META-ANALYSIS -----------------
with tab6:
    st.header("Comparative Meta-Analysis")
    st.markdown("Upload multiple `.tau` sessions to compare their Systemic Transition (RECD) curves on a single unified chart.")
    
    meta_files = st.file_uploader("Upload multiple .tau files", type=["tau"], accept_multiple_files=True, key="meta_uploader")
    
    if meta_files and len(meta_files) > 0:
        if st.button("🔬 Generate Meta-Analysis"):
            with st.spinner("Analyzing and rendering multiple sessions..."):
                fig = go.Figure()
                valid_files = 0
                
                for m_file in meta_files:
                    try:
                        # Load the pickle
                        state_dict = pickle.loads(m_file.getvalue())
                        if "analysis_results" in state_dict and state_dict["analysis_results"] is not None:
                            res = state_dict["analysis_results"]
                            T_series = res.get("T_series", [])
                            t_star = res.get("t_star", None)
                            
                            if len(T_series) > 0:
                                # Add trace
                                fig.add_trace(go.Scatter(
                                    x=np.arange(len(T_series)),
                                    y=T_series,
                                    mode='lines',
                                    name=f"{m_file.name}",
                                    opacity=0.8
                                ))
                                # Add t_star vertical line if present and valid
                                if t_star is not None and isinstance(t_star, (int, float)) and not np.isnan(t_star):
                                    fig.add_vline(x=t_star, line_dash="dot", 
                                                  annotation_text=f"t* ({m_file.name})", 
                                                  annotation_position="top right")
                                valid_files += 1
                    except Exception as e:
                        st.error(f"Error parsing {m_file.name}: {e}")
                        
                if valid_files > 0:
                    fig.update_layout(
                        title=f"Comparative RECD Manifold ({valid_files} Sessions)",
                        xaxis_title="Time Step (k)",
                        yaxis_title="Accumulated RECD ($T_k$)",
                        template="plotly_white",
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.warning("No valid RECD data found in the uploaded .tau files.")

# ----------------- TAB 7: ROBUSTNESS & NARRATIVE (v4.6.0) -----------------
with tab7:
    st.header("🛡️ Robustness & Tiered Narrative")
    st.write("Evaluate the mathematical stability of the topological transition and explore the ontological narrative.")
    
    if st.session_state.ews_results is None:
        st.warning("Please run the analysis in the Systemic Analysis tab first.")
    else:
        st.subheader("Tier 1: Ontological Overview")
        st.info("A unified, publication-ready overview of the ontological ascent.")
        from systemictau.visualization.tier_viz import plot_ontological_overview, plot_layer_details
        
        # We already cached res_obj as `raw_results` during the Systemic Analysis run.
        if "raw_results" in st.session_state and st.session_state.raw_results is not None:
            res_obj = st.session_state.raw_results
            
            fig_t1 = plot_ontological_overview(res_obj)
            st.pyplot(fig_t1)
            
            st.divider()
            st.subheader("Tier 2: Layer Detail")
            fig_t2 = plot_layer_details(res_obj)
            st.pyplot(fig_t2)
            
            st.divider()
            st.subheader("Statistical Robustness (Noise Sweeps)")
            st.write("Test if $t^*$ remains mathematically bounded under varying levels of Gaussian noise.")
            
            col1, col2 = st.columns(2)
            with col1:
                noise_levels_input = st.multiselect("Noise Levels", [0.05, 0.10, 0.15, 0.20], default=[0.05, 0.10])
            with col2:
                iterations = st.number_input("Iterations per level", min_value=1, max_value=10, value=3)
                
            if st.button("Run Sensitivity Analysis"):
                if "clustered_df" in st.session_state and st.session_state.clustered_df is not None:
                    with st.spinner("Injecting noise and perturbing manifold..."):
                        from systemictau.robustness import run_sensitivity_analysis
                        
                        rob_report = run_sensitivity_analysis(
                            X=st.session_state.clustered_df.values,
                            window_size=window_size,
                            theta_A=theta_A,
                            D_min=D_min,
                            noise_levels=noise_levels_input,
                            iterations_per_level=iterations,
                            component_names=st.session_state.clustered_df.columns.tolist()
                        )
                        
                        st.success("Sensitivity Analysis Complete!")
                        st.json(rob_report)
                else:
                    st.error("Data not found. Please upload a dataset.")
