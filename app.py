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

st.set_page_config(page_title="Systemic Tau Studio", page_icon="🧬", layout="wide")

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
def run_core_math(X, window_size, component_names=None):
    res_obj = run_full_analysis(X, window_size=window_size, component_names=component_names)
    
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
    st.markdown("<h3 style='text-align: center;'>Systemic Tau 🧬 v4.4.x</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888;'>True Mathematical Engine</p>", unsafe_allow_html=True)
    
    window_size = st.slider(
        "Window Size (n)", 
        min_value=5, max_value=150, value=13, step=1, 
        help="**¿Qué es la Ventana Deslizante?**\nEs el número de observaciones temporales consecutivas (t) evaluadas simultáneamente para calcular el nivel de entrelazamiento sistémico.\n\n*   **Ventanas Cortas (Ej. 5-10):** Capturan dinámicas ultrarrápidas y cambios abruptos, pero son altamente susceptibles al 'ruido' estadístico.\n*   **Ventanas Largas (Ej. 20+):** Suavizan la señal revelando la tendencia macroscópica, pero *retrasan* matemáticamente la detección de la Alerta Temprana (EWS).\n*Recomendación*: 13 es un óptimo de Pareto empírico probado en ciclos biológicos y financieros."
    )
    
    with st.expander("⚙️ Advanced Parameters"):
        st.markdown("<small>Filtros Probabilísticos</small>", unsafe_allow_html=True)
        theta_A = st.number_input(
            "Umbral de Magnitud (θ_A)", 
            value=0.05, step=0.01, 
            help="**Umbral de Magnitud Topológica (θ_A)**\nFiltra fluctuaciones menores. Define la agresividad mínima requerida para que un cambio relacional sea catalogado como un 'Episodio Sistémico' (Critical Lock-in). Valores históricos como 0.41 representan barreras termodinámicas extremas, mientras que 0.05 es ideal para alta sensibilidad."
        )
        D_min = st.number_input(
            "Duración Mínima (D_min)", 
            value=10, step=1, 
            help="**Duración Mínima (D_min)**\nCondición de contorno temporal. Un bloqueo topológico solo adquiere validación matemática si el sistema permanece anclado en hiper-sincronización durante al menos *D_min* pasos temporales. Actúa como un filtro pasa-bajos contra falsos positivos y micro-shocks transitorios."
        )
        
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
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Data Hub & Health", 
    "🌐 Ontological Scales", 
    "📈 Systemic Analysis", 
    "🌀 EWS & Diagnostics", 
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
        st.markdown("### 🧪 Datasets de Prueba Integrados")
        st.info("**¿No tienes datos a la mano?** Explora el potencial del motor Systemic Tau cargando nuestro dataset validado metodológicamente.")
        
        if st.button("🦠 Cargar Dataset: Brote de Dengue (DengAI)"):
            if os.path.exists("data/datos_dengai_completo.csv"):
                st.session_state.raw_df = pd.read_csv("data/datos_dengai_completo.csv")
                st.success("✅ **Dataset de Epidemia de Dengue cargado.** Este dataset contiene variables climáticas y de vegetación. Ve a 'Data Health' para iniciar el preprocesamiento y luego a 'Ontological Scales'.")
                st.rerun()
            else:
                st.error("Archivo de dataset de ejemplo no encontrado en la ruta 'data/'.")
    
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
            res_dict, res_obj = run_core_math(X, window_size, component_names=c_names)
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
        st.subheader("Academic Report Hub")
        if HAS_REPORTS:
            # Add toggle for extended theoretical captions
            include_theory = st.checkbox("Include extended theoretical graph interpretations", value=False, help="Appends rigorous academic interpretations below each figure in the generated PDF/Markdown report based on the Systemic Tau framework.")
            # Add toggle for statistical significance appendix
            include_significance = st.checkbox("Include Statistical Significance Appendix (Appendix B)", value=False, help="Appends an appendix explaining how to interpret the results mathematically without relying on classical p-values.")
            
            with st.status("Compiling Academic Report...", expanded=True) as status:
                st.write("Generating mathematical logic...")
                st.write("Embedding high-res Base64 figures...")
                
                # The systemictau package now handles generating and safely embedding 
                # all base64 figures automatically when include_figures=True.
                report_md = generate_academic_report(
                    st.session_state.raw_results, 
                    lang='en', 
                    include_figures=True,
                    include_theoretical_captions=include_theory,
                    include_significance_appendix=include_significance,
                    title=st.session_state.get("report_title", "Systemic Tau Paradigm Analytical Report"),
                    author=st.session_state.get("report_author", ""),
                    organization=st.session_state.get("report_org", ""),
                    ews_results=st.session_state.get("ews_results")
                )
                
                status.update(label="Report Compilation Complete", state="complete", expanded=False)
                
            st.download_button(
                label="⬇️ Download Academic Report (.md)",
                data=report_md,
                file_name="systemic_tau_academic_report.md",
                mime="text/markdown",
                type="primary"
            )
            
            # Create PDF equivalent
            try:
                pdf_bytes = convert_markdown_to_pdf(report_md)
                st.download_button(
                    label="📄 Download Academic Report (.pdf)",
                    data=pdf_bytes,
                    file_name="systemic_tau_academic_report.pdf",
                    mime="application/pdf",
                    type="secondary"
                )
            except Exception as e:
                st.warning(f"Could not generate PDF: {e}")
            
            with st.expander("Preview Academic Report", expanded=False):
                st.markdown(report_md)
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
                        template="plotly_dark",
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.warning("No valid RECD data found in the uploaded .tau files.")
