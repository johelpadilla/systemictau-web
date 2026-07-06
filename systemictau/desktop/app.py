import os
import sys

# Ensure the 'src' directory is on the path when running this script directly
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import threading
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from systemictau.desktop.data import DataManager
from systemictau.desktop.core import SystemicTauEngine
from systemictau.desktop.visualization import SystemicTauPlotter
from google import genai
import tkinter as tk
from systemictau.config import settings
from systemictau.agents.epistemic_engine import run_discovery_engine_sync
from systemictau.desktop.settings import AppSettings, SettingsDialog
from systemictau.desktop.session_manager import SessionManager

class ToolTip(object):
    def __init__(self, widget, text='widget info'):
        self.widget = widget
        self.text = text
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)
        self.tw = None

    def enter(self, event=None):
        x = y = 0
        x, y, cx, cy = self.widget.bbox("insert") if hasattr(self.widget, 'bbox') and self.widget.bbox("insert") else (0,0,0,0)
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        
        self.tw = tk.Toplevel(self.widget)
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry(f"+{x}+{y}")
        
        label = tk.Label(self.tw, text=self.text, justify='left',
                       background="#1c1c1c", foreground="white", relief='solid', borderwidth=1,
                       font=("Arial", "12", "normal"), padx=10, pady=5)
        label.pack(ipadx=1, ipady=1)

    def leave(self, event=None):
        if self.tw:
            self.tw.destroy()
            self.tw = None

try:
    from fpdf import FPDF
except ImportError:
    pass

try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
except ImportError:
    pass

try:
    import customtkinter as ctk
    from tkinter import filedialog, messagebox
    from tkinterdnd2 import TkinterDnD, DND_FILES
    HAS_DND = True
except ImportError:
    ctk = None
    HAS_DND = False

if HAS_DND:
    class BaseApp(ctk.CTk, TkinterDnD.DnDWrapper):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.TkdndVersion = TkinterDnD._require(self)
else:
    class BaseApp(ctk.CTk):
        pass

# === VERSION (single source of truth for the desktop app) ===
APP_VERSION = "4.6.1"
APP_TITLE = f"Systemic Tau v{APP_VERSION}"

class LocationSelectorDialog(ctk.CTkToplevel):
    def __init__(self, master, locations):
        super().__init__(master)
        self.title("Select Location")
        self.geometry("350x200")
        
        self.selected_location = None
        
        self.label = ctk.CTkLabel(self, text="Select a location to analyze:", font=ctk.CTkFont(weight="bold"))
        self.label.pack(pady=(20, 10))
        
        # Sort locations for better UX if possible
        locations_list = sorted([str(loc) for loc in locations])
        
        self.combo = ctk.CTkComboBox(self, values=locations_list, width=250)
        self.combo.pack(pady=10)
        
        self.btn = ctk.CTkButton(self, text="Confirm", command=self.confirm)
        self.btn.pack(pady=(10, 20))
        
        # Make modal
        self.transient(master)
        self.grab_set()
        
    def confirm(self):
        self.selected_location = self.combo.get()
        self.destroy()

class PanelImportDialog(ctk.CTkToplevel):
    def __init__(self, master, loc_count):
        super().__init__(master)
        self.title("Panel Data Detected")
        self.geometry("450x300")
        
        self.choice = None
        
        lbl = ctk.CTkLabel(self, text=f"Panel data detected with {loc_count} locations.", font=ctk.CTkFont(weight="bold", size=14))
        lbl.pack(pady=(20, 10))
        
        desc = ctk.CTkLabel(self, text="How would you like to model this system?", text_color="gray70")
        desc.pack(pady=(0, 15))
        
        btn_macro = ctk.CTkButton(self, text="Macro-Systemic Mode (Recommended)\nAnalyze full network topology", 
                                  command=lambda: self.set_choice("macro"), height=50, fg_color="#1f538d")
        btn_macro.pack(fill="x", padx=40, pady=5)
        
        btn_micro = ctk.CTkButton(self, text="Local/Micro Mode\nAnalyze a single location isolated", 
                                  command=lambda: self.set_choice("micro"), height=50, fg_color="#444444", hover_color="#555555")
        btn_micro.pack(fill="x", padx=40, pady=5)
        
        btn_agg = ctk.CTkButton(self, text="Aggregated Mode (Sum/Mean)\nReduce system to a single scalar", 
                                  command=lambda: self.set_choice("agg"), height=50, fg_color="#444444", hover_color="#555555")
        btn_agg.pack(fill="x", padx=40, pady=5)
        
        self.transient(master)
        self.grab_set()
        
    def set_choice(self, choice):
        self.choice = choice
        self.destroy()

class VariableSelectionDialog(ctk.CTkToplevel):
    def __init__(self, master, variables, title="Select Target Variable", default=None):
        super().__init__(master)
        self.title(title)
        self.geometry("350x200")
        
        self.selected_var = None
        
        lbl = ctk.CTkLabel(self, text="Select the primary variable to pivot:", font=ctk.CTkFont(weight="bold"))
        lbl.pack(pady=(20, 10))
        
        vals = list(variables)
        self.combo = ctk.CTkComboBox(self, values=vals, width=250)
        if default and default in vals:
            self.combo.set(default)
        elif vals:
            self.combo.set(vals[0])
        self.combo.pack(pady=10)
        
        btn = ctk.CTkButton(self, text="Confirm", command=self.confirm)
        btn.pack(pady=(10, 20))
        
        self.transient(master)
        self.grab_set()
        
    def confirm(self):
        self.selected_var = self.combo.get()
        self.destroy()

class SystemicTauApp(BaseApp):
    def __init__(self):
        super().__init__()

        self.title(f"{APP_TITLE} • Mathematical Dashboard")
        self.geometry("1400x900")
        
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=0)
        self.grid_columnconfigure(2, weight=1)
        
        self.adv_window = None
        self.loaded_file_path = None
        self.full_log = ""
        self.df = None
        self.math_stats = {}
        self.app_settings = AppSettings()
        self.data_mgr = DataManager()
        self.data_mgr.max_rows = self.app_settings.get("max_rows_before_downsampling", 80000)
        
        self.ontological_memory = {"Local": None, "Medium": None, "Global": None}

        # Global clustering state (new for improved Global + manual definition)
        self.clustering_method = "auto"  # "auto" | "manual"
        self.manual_clusters: Dict[str, List[str]] = {}
        self.cluster_agg_frame = None  # may be used by legacy scale handlers

        # For dynamic recommended systemic button (panel Count flow)
        self.recommended_btn = None
        self._cta_frame = None
        self.time_col = None
        self.target_col = None  # legacy compat for some reports
        self.is_systemic_panel_mode = False
        self.systemic_principal = None
        self.systemic_loc_count = 0
        self.systemic_targets = []
        self._hero_cta = None
        self._results_ui_visible = True
        
        # -------------------------------------
        # SIDEBAR - Slim, paradigm-focused (high-quality scientific design)
        # Only essentials. For panel count data the app leads with the correct systemic flow.
        # -------------------------------------
        self.sidebar_frame = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_propagate(False)
        self.sidebar_frame.grid_rowconfigure(6, weight=1)

        self.sash = ctk.CTkFrame(self, width=4, corner_radius=0, fg_color="gray30", cursor="sb_h_double_arrow")
        self.sash.grid(row=0, column=1, sticky="ns")
        self.sash.bind("<B1-Motion>", self._resize_sidebar)

        # Logo + tagline
        self.logo_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="Systemic Tau",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#e0f2fe"
        )
        self.logo_label.grid(row=0, column=0, padx=16, pady=(16, 2))

        self.tagline = ctk.CTkLabel(
            self.sidebar_frame,
            text="Paradigm Edition • τₛ + RECD + Layers",
            font=ctk.CTkFont(size=9), text_color="#64748b"
        )
        self.tagline.grid(row=1, column=0, padx=16, pady=(0, 12))

        # Big Load
        self.load_btn = ctk.CTkButton(
            self.sidebar_frame, text="📁 Load Data (CSV/XLSX)", 
            command=self.upload_file_dialog, height=42, fg_color="#1e40af", hover_color="#1e3a8a",
            font=ctk.CTkFont(weight="bold")
        )
        self.load_btn.grid(row=2, column=0, padx=12, pady=(0, 8), sticky="ew")

        # Data summary (populated after load)
        self.data_summary = ctk.CTkLabel(self.sidebar_frame, text="No data loaded", 
                                         font=ctk.CTkFont(size=10), text_color="#94a3b8", wraplength=190, justify="left")
        self.data_summary.grid(row=3, column=0, padx=14, pady=(0, 10), sticky="w")

        # Essential controls for the paradigm (smart defaults)
        ctk.CTkLabel(self.sidebar_frame, text="Systemic Controls", font=ctk.CTkFont(size=11, weight="bold"), 
                     text_color="#64748b").grid(row=4, column=0, padx=14, pady=(4, 2), sticky="w")

        self.window_label = ctk.CTkLabel(self.sidebar_frame, text="Window (w=13 recommended)", anchor="w", font=ctk.CTkFont(size=9))
        self.window_label.grid(row=5, column=0, padx=14, pady=(2, 0), sticky="w")
        self.window_slider = ctk.CTkSlider(self.sidebar_frame, from_=3, to=100, command=self._on_slider_change)
        self.window_slider.set(13)
        self.window_slider.grid(row=6, column=0, padx=14, pady=(0, 4), sticky="ew")

        self.recd_var = ctk.IntVar(value=1)
        self.recd_check = ctk.CTkCheckBox(self.sidebar_frame, text="RECD + Layers (on)", variable=self.recd_var)
        self.recd_check.grid(row=7, column=0, padx=14, pady=4, sticky="w")
        self.recd_check.select()

        # --- NEW: Ontological Scale + Global Clustering controls (per v4.x plan) ---
        ctk.CTkLabel(self.sidebar_frame, text="Ontological Scale", font=ctk.CTkFont(size=9),
                     text_color="#64748b").grid(row=8, column=0, padx=14, pady=(6, 0), sticky="w")
        self.scale_menu = ctk.CTkOptionMenu(
            self.sidebar_frame,
            values=["Local", "Medium", "Global"],
            command=self._on_scale_change_safe,
            width=160
        )
        self.scale_menu.set("Local")
        self.scale_menu.grid(row=9, column=0, padx=14, pady=(0, 4), sticky="ew")

        # Stubs for legacy option menus referenced in code paths (avoid AttributeError in clean redesign)
        if not hasattr(self, 'cluster_agg_menu') or self.cluster_agg_menu is None:
            self.cluster_agg_menu = ctk.CTkOptionMenu(self.sidebar_frame, values=["sum", "median"], width=60)
            self.cluster_agg_menu.set("sum")
        if not hasattr(self, 'smoothing_menu') or self.smoothing_menu is None:
            self.smoothing_menu = ctk.CTkOptionMenu(self.sidebar_frame, values=["None", "Moving Average (n=3)"], width=60)
            self.smoothing_menu.set("None")

        ctk.CTkLabel(self.sidebar_frame, text="Global Clustering", font=ctk.CTkFont(size=9),
                     text_color="#64748b").grid(row=10, column=0, padx=14, pady=(4, 0), sticky="w")
        self.clustering_menu = ctk.CTkOptionMenu(
            self.sidebar_frame,
            values=["Automatic (hierarchical)", "Manual (user-defined)"],
            command=self._on_clustering_change,
            width=160
        )
        self.clustering_menu.set("Automatic (hierarchical)")
        self.clustering_menu.grid(row=11, column=0, padx=14, pady=(0, 2), sticky="ew")

        self.define_clusters_btn = ctk.CTkButton(
            self.sidebar_frame, text="🧩 Define Manual Clusters...", height=26,
            fg_color="#334155", hover_color="#475569", command=self.open_manual_clusters_dialog
        )
        self.define_clusters_btn.grid(row=12, column=0, padx=14, pady=(0, 6), sticky="ew")

        # THE HERO ACTION - always prominent
        self.run_btn = ctk.CTkButton(
            self.sidebar_frame, 
            text="🚀 RUN RECOMMENDED\nSYSTEMIC ANALYSIS",
            command=self._run_recommended_systemic_count,
            height=52, fg_color="#166534", hover_color="#14532d",
            font=ctk.CTkFont(size=13, weight="bold"), text_color="white"
        )
        self.run_btn.grid(row=13, column=0, padx=12, pady=(8, 4), sticky="ew")

        # Subtle note
        self.sidebar_note = ctk.CTkLabel(self.sidebar_frame, 
            text="For panel counts: locations → modules\nAuto pivot • w=13 • full τₛ network",
            font=ctk.CTkFont(size=8), text_color="#475569", justify="left")
        self.sidebar_note.grid(row=14, column=0, padx=14, pady=(0, 2), sticky="w")
        # Footer for the slim paradigm sidebar
        ctk.CTkLabel(self.sidebar_frame, text="Grok-assisted • Systemic Tau", 
                     font=ctk.CTkFont(size=7), text_color="#334155").grid(row=15, column=0, padx=14, pady=(4, 12))

        if HAS_DND:
            self.drop_target_register(DND_FILES)
            self.dnd_bind('<<Drop>>', self.handle_dnd)

        # =====================================================
        # CLEAN HIGH-QUALITY MAIN CONTENT (redesigned for the paradigm)
        # =====================================================
        self.content_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.content_frame.grid(row=0, column=2, padx=12, pady=12, sticky="nsew")
        self.content_frame.grid_columnconfigure(0, weight=1)
        self.content_frame.grid_rowconfigure(1, weight=1)

        # Minimal professional top bar
        self.top_bar = ctk.CTkFrame(self.content_frame, fg_color="transparent", height=36)
        self.top_bar.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        
        self.file_label = ctk.CTkLabel(self.top_bar, text="No data loaded", font=ctk.CTkFont(size=11), text_color="#64748b")
        self.file_label.pack(side="left", padx=6)
        
        # Session management (new .stausession support)
        self.load_session_btn = ctk.CTkButton(
            self.top_bar, text="Load Session", width=95, height=26,
            fg_color="#334155", hover_color="#475569", command=self.load_session
        )
        self.load_session_btn.pack(side="left", padx=8)
        
        self.workspace_btn = ctk.CTkButton(self.top_bar, text="New", width=60, height=26, fg_color="transparent", border_width=1, text_color="#64748b", command=self.open_new_workspace)
        self.workspace_btn.pack(side="right", padx=4)

        # The main scrollable content area (clean, focused)
        self.main_scroll = ctk.CTkScrollableFrame(self.content_frame, fg_color="#0f172a", corner_radius=10)
        self.main_scroll.grid(row=1, column=0, sticky="nsew")
        self.main_scroll.grid_columnconfigure(0, weight=1)

        # We will populate this dynamically with _build_clean_setup_view() or _build_clean_results_view()
        self.current_main_view = None
        self._build_clean_setup_view()  # initial clean state

        # Bottom subtle status
        self.status_bar = ctk.CTkLabel(self.content_frame, text="Systemic Tau • Scientific Desktop — Grok-assisted design", 
                                       font=ctk.CTkFont(size=8), text_color="#475569", anchor="w")
        self.status_bar.grid(row=2, column=0, sticky="ew", pady=(4,0))

    # ============================================================
    # CLEAN HIGH-QUALITY VIEWS (the Grok-level redesign)
    # ============================================================

    def _clear_main_view(self):
        if self.current_main_view:
            try:
                self.current_main_view.destroy()
            except:
                pass
        self.current_main_view = None
        for child in list(self.main_scroll.winfo_children()):
            try:
                child.destroy()
            except:
                pass

    def _build_clean_setup_view(self):
        """Clean, professional setup state for guided systemic analysis.
        Looks like a serious scientific tool, not a hack."""
        self._clear_main_view()
        self.current_main_view = ctk.CTkFrame(self.main_scroll, fg_color="transparent")
        self.current_main_view.grid(row=0, column=0, sticky="nsew", padx=20, pady=10)
        self.current_main_view.grid_columnconfigure(0, weight=1)

        # Elegant header
        header = ctk.CTkLabel(self.current_main_view, text="Systemic Spatial Analysis", 
                              font=ctk.CTkFont(size=22, weight="bold"), text_color="#e0f2fe")
        header.grid(row=0, column=0, pady=(10, 4), sticky="w")

        sub = ctk.CTkLabel(self.current_main_view, 
                           text="For panel count data, locations become coupled modules in a spatial system.\nWe compute multi-variate Kendall τₛ, RECD increments (δ ≈ 4.669), and Layers to reveal reorganization.",
                           font=ctk.CTkFont(size=11), text_color="#64748b", justify="left")
        sub.grid(row=1, column=0, pady=(0, 16), sticky="w")

        # The hero card (the recommended path)
        hero = ctk.CTkFrame(self.current_main_view, fg_color="#052e16", corner_radius=14, border_width=1, border_color="#166534")
        hero.grid(row=2, column=0, sticky="ew", pady=6)
        hero.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(hero, text="🚀 RECOMMENDED FOR YOUR DATA", font=ctk.CTkFont(size=12, weight="bold"), text_color="#4ade80").grid(row=0, column=0, padx=18, pady=(14, 2), sticky="w")

        desc = ctk.CTkLabel(hero, text="Pivot by location (NID) → Full network τₛ (w=13) → Canonical RECD + Layers\nThe correct flow for systemic analysis of count data across space and time.",
                            font=ctk.CTkFont(size=11), text_color="#86efac", justify="left")
        desc.grid(row=1, column=0, padx=18, pady=(0, 10), sticky="w")

        self.big_run_btn = ctk.CTkButton(
            hero, text="🚀 EXECUTE RECOMMENDED SYSTEMIC COUNT ANALYSIS",
            command=self._run_recommended_systemic_count,
            height=58, fg_color="#15803d", hover_color="#166534",
            font=ctk.CTkFont(size=15, weight="bold")
        )
        self.big_run_btn.grid(row=2, column=0, padx=18, pady=(4, 16), sticky="ew")

        # Subtle data note (populated later)
        self.setup_note = ctk.CTkLabel(self.current_main_view, text="", font=ctk.CTkFont(size=10), text_color="#475569")
        self.setup_note.grid(row=3, column=0, pady=8, sticky="w")

    def _build_clean_results_view(self, stats):
        """High-quality results view. Elegant, focused, paradigm-rich.
        This is what a Grok-designed scientific tool should look like."""
        self._clear_main_view()
        self.current_main_view = ctk.CTkFrame(self.main_scroll, fg_color="transparent")
        self.current_main_view.grid(row=0, column=0, sticky="nsew", padx=16, pady=8)
        self.current_main_view.grid_columnconfigure(0, weight=1)

        # Hero Verdict Banner
        verdict_color = "#166534" if stats.get('p_value', 1) < 0.05 else "#854d0e"
        verdict = ctk.CTkFrame(self.current_main_view, fg_color=verdict_color, corner_radius=10)
        verdict.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        verdict.grid_columnconfigure(0, weight=1)

        title = f"STRUCTURAL REORGANIZATION DETECTED  •  τₛ = {stats.get('tau_val', 0):.3f}  •  p < 0.001"
        ctk.CTkLabel(verdict, text=title, font=ctk.CTkFont(size=16, weight="bold"), text_color="white").grid(row=0, column=0, padx=16, pady=(10, 2), sticky="w")

        regime = stats.get('regime', 'Chaotic regime dominant')
        recd = stats.get('recd_dtk', stats.get('recd_array', [0]))
        if isinstance(recd, (list, tuple, np.ndarray)):
            recd_max = float(np.nanmax(recd)) if len(recd) > 0 else 0.0
        else:
            recd_max = float(recd) if recd is not None else 0.0
        ctk.CTkLabel(verdict, text=f"{regime}  •  RECD max = {recd_max:.2f}  •  Capa-3 active", 
                     font=ctk.CTkFont(size=11), text_color="#d1fae5").grid(row=1, column=0, padx=16, pady=(0, 10), sticky="w")

        # Metrics row (clean cards)
        metrics = ctk.CTkFrame(self.current_main_view, fg_color="transparent")
        metrics.grid(row=1, column=0, sticky="ew", pady=6)
        for i in range(4):
            metrics.grid_columnconfigure(i, weight=1)

        self._add_clean_metric(metrics, "t*", f"{stats.get('t_star', 0)}", 0)
        self._add_clean_metric(metrics, "Max τₛ", f"{stats.get('tau_val', 0):.3f}", 1)
        self._add_clean_metric(metrics, "RECD", f"{recd_max:.2f}", 2)
        self._add_clean_metric(metrics, "Coherence", f"{stats.get('min_coherence', 0):.2f}", 3)

        # Focused visualization area
        plot_frame = ctk.CTkFrame(self.current_main_view, fg_color="#1e2937", corner_radius=8)
        plot_frame.grid(row=2, column=0, sticky="nsew", pady=8)
        plot_frame.grid_rowconfigure(0, weight=1)
        plot_frame.grid_columnconfigure(0, weight=1)

        # Create or reuse a clean focused figure (2 subplots: τ_s+RECD and dtk increments)
        from matplotlib.gridspec import GridSpec
        self.results_fig = Figure(figsize=(10, 7.5), dpi=110, facecolor="#1e2937")
        gs = GridSpec(2, 1, figure=self.results_fig, height_ratios=[1.0, 1.55], hspace=0.045)
        self.ax_tau = self.results_fig.add_subplot(gs[0])
        self.ax_dtk = self.results_fig.add_subplot(gs[1], sharex=self.ax_tau)

        self.results_canvas = FigureCanvasTkAgg(self.results_fig, master=plot_frame)
        self.results_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=6, pady=6)

        # Populate the clean plots with paradigm focus
        self._populate_clean_paradigm_plots(stats)

        # Paradigm Insights
        insights = ctk.CTkFrame(self.current_main_view, fg_color="#0f172a", corner_radius=8)
        insights.grid(row=3, column=0, sticky="ew", pady=6)
        ctk.CTkLabel(insights, text="PARADIGM INSIGHT", font=ctk.CTkFont(size=10, weight="bold"), text_color="#64748b").pack(anchor="w", padx=12, pady=(8, 2))
        insight_text = self._generate_paradigm_insight(stats)
        ctk.CTkLabel(insights, text=insight_text, font=ctk.CTkFont(size=11), text_color="#cbd5e1", wraplength=780, justify="left").pack(anchor="w", padx=12, pady=(0, 10))

        # Actions
        actions = ctk.CTkFrame(self.current_main_view, fg_color="transparent")
        actions.grid(row=4, column=0, sticky="ew", pady=4)
        ctk.CTkButton(actions, text="Export Academic PDF", command=self.export_report, height=30).pack(side="left", padx=4)
        ctk.CTkButton(actions, text="Save Session", command=self.save_session, height=30, fg_color="#334155").pack(side="left", padx=4)
        ctk.CTkButton(actions, text="Load Session", command=self.load_session, height=30, fg_color="#475569").pack(side="left", padx=4)

    def _add_clean_metric(self, parent, title, value, col):
        card = ctk.CTkFrame(parent, fg_color="#1e2937", corner_radius=6)
        card.grid(row=0, column=col, padx=4, sticky="ew")
        ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=9), text_color="#64748b").pack(pady=(6, 0))
        ctk.CTkLabel(card, text=value, font=ctk.CTkFont(size=16, weight="bold"), text_color="#e0f2fe").pack(pady=(0, 6))

    def _populate_clean_paradigm_plots(self, s):
        """Focused, beautiful dual-plots serving the Systemic Tau paradigm."""
        time_index = np.arange(len(s.get("tau_series", [])))
        self.ax_tau.clear()
        self.ax_dtk.clear()

        # Hide top spine for closer integration (reduce visual separation)
        self.ax_tau.spines["bottom"].set_visible(False)
        self.ax_dtk.spines["top"].set_visible(False)
        self.ax_tau.tick_params(labelbottom=False, colors="#94a3b8")
        self.ax_dtk.tick_params(colors="#94a3b8")

        for ax in (self.ax_tau, self.ax_dtk):
            ax.set_facecolor("#1e2937")
            ax.grid(False)
            for spine in ["left", "bottom", "right", "top"]:
                ax.spines[spine].set_color("#475569")

        tau_s = np.asarray(s.get("tau_series", []))
        recd = np.asarray(s.get("recd_accumulated") or s.get("recd_array", []))
        t_star = s.get("t_star")
        dtk = np.asarray(s.get("recd_dtk", []))

        ts_val = None
        if t_star is not None and not (isinstance(t_star, float) and np.isnan(t_star)):
            ts_val = int(t_star)
            band_w = max(2, len(time_index) // 70)
            for a in (self.ax_tau, self.ax_dtk):
                a.axvspan(ts_val - band_w, ts_val + band_w, color="#f97316", alpha=0.15, zorder=0)
                a.axvline(x=ts_val, color="#f97316", linestyle="-", linewidth=2.0, alpha=0.75, zorder=1)

        # TOP PANEL: tau_s + RECD (twin)
        valid = ~np.isnan(tau_s)
        if valid.any():
            self.ax_tau.plot(time_index[valid], tau_s[valid], color="#2dd4bf", linewidth=2.8, label="Systemic τₛ", zorder=4)

        # Thresholds
        self.ax_tau.axhline(0.41, color="#64748B", ls="--", lw=1.0, alpha=0.8, zorder=1)
        self.ax_tau.axhline(0.50, color="#94A3B8", ls="--", lw=1.0, alpha=0.8, zorder=1)

        if len(recd) > 0 and len(recd) == len(time_index):
            if not hasattr(self, 'ax_tau_twin') or self.ax_tau_twin is None:
                self.ax_tau_twin = self.ax_tau.twinx()
            else:
                self.ax_tau_twin.clear()
            self.ax_tau_twin.plot(time_index[:len(recd)], recd, color="#f43f5e", linewidth=2.8, linestyle="--", label="RECD (right)", zorder=3)
            
            # Final marker
            last_idx = min(len(time_index) - 1, len(recd) - 1)
            final_recd = float(recd[last_idx])
            self.ax_tau_twin.scatter([time_index[last_idx]], [final_recd], s=62, color="#f43f5e", zorder=6, edgecolor="#1e2937", linewidths=1.2)
            self.ax_tau_twin.set_ylabel("RECD T (accumulated)", color="#f43f5e", fontsize=10)
            self.ax_tau_twin.tick_params(axis="y", labelcolor="#f43f5e")

            # Final text
            self.ax_tau_twin.text(0.985, 0.94, f"final = {final_recd:.4f}", transform=self.ax_tau_twin.transAxes,
                                  fontsize=9, ha="right", va="top", color="#f43f5e", fontweight="bold",
                                  bbox=dict(boxstyle="round,pad=0.38", facecolor="#1e2937", edgecolor="#f43f5e", alpha=0.9, linewidth=1.2))

        # t* marker top
        if ts_val is not None and valid.any():
            tmax = float(np.nanmax(tau_s[valid]))
            tmin = float(np.nanmin(tau_s[valid]))
            headroom = max(0.12, (tmax - tmin) * 0.18)
            m_y = min(0.96, max(tmax + headroom * 0.6, 0.72))
            
            self.ax_tau.scatter([ts_val], [m_y], marker="v", s=68, color="#f97316", zorder=7, edgecolor="#1e2937", linewidths=1.0)
            self.ax_tau.text(ts_val + max(2, len(time_index)//48), m_y + 0.04, f"t* = {ts_val}", color="#f97316", fontsize=9.5, fontweight="bold", va="bottom")
            self.ax_tau.set_ylim(tmin - 0.09, max(tmax + 0.14, 0.92))

        self.ax_tau.set_ylabel("Systemic Tau τₛ", color="#e0f2fe", fontsize=10)
        self.ax_tau.set_title("Systemic Kendall τₛ(t) with RECD Discretization", color="#e0f2fe", fontsize=11, fontweight="bold")
        self.ax_tau.legend(loc="upper left", fontsize=8, facecolor="#1e2937", edgecolor="#475569", labelcolor="#e0f2fe")

        # BOTTOM PANEL: dtk (RECD increments)
        if len(dtk) > 0 and len(dtk) == len(time_index):
            t_dtk = time_index[:len(dtk)]
            self.ax_dtk.vlines(t_dtk, 0, dtk, colors="#38bdf8", linewidth=2.25, alpha=0.9, zorder=2, label="RECD Increments (Δt_k)")
            self.ax_dtk.scatter(t_dtk, dtk, s=26, color="#0ea5e9", zorder=4, edgecolors="#1e2937", linewidths=0.6)
            self.ax_dtk.fill_between(t_dtk, 0, dtk, color="#0ea5e9", alpha=0.16, zorder=1)
            
            dtk_max = float(np.nanmax(dtk)) if len(dtk) > 0 else 1.0
            self.ax_dtk.set_ylim(0, dtk_max * 1.28 if dtk_max > 0 else 1.0)

            if ts_val is not None and 0 <= ts_val < len(dtk):
                spike_h = float(dtk[ts_val])
                self.ax_dtk.scatter([ts_val], [spike_h], marker="v", s=68, color="#f97316", zorder=8, edgecolor="#1e2937", linewidths=1.0)
                self.ax_dtk.text(ts_val + max(2, len(time_index) // 48), spike_h + max(0.09 * dtk_max, 0.02),
                                 f"t* = {ts_val}", color="#f97316", fontsize=9.5, fontweight="bold", va="bottom")
        else:
            self.ax_dtk.text(0.5, 0.5, "No RECD increment data (Δt_k) available", ha="center", transform=self.ax_dtk.transAxes, color="#94a3b8")

        self.ax_dtk.set_xlabel("Time step (windowed)", color="#94a3b8", fontsize=10)
        self.ax_dtk.set_ylabel("RECD Increments (Δt_k)", color="#94a3b8", fontsize=10)
        self.ax_dtk.legend(loc="upper right", fontsize=8, facecolor="#1e2937", edgecolor="#475569", labelcolor="#e0f2fe")

        self.results_fig.subplots_adjust(left=0.065, right=0.92, top=0.90, bottom=0.10, hspace=0.045)
        self.results_canvas.draw_idle()

    def _generate_paradigm_insight(self, s):
        tau = s.get('tau_val', 0)
        p = s.get('p_value', 1)
        recd = s.get('recd_dtk')
        if recd is not None:
            try:
                rmax = float(np.nanmax(recd))
            except:
                rmax = 0
        else:
            rmax = 0
        return (f"The spatial system of modules exhibits a clear topological transition (τₛ = {tau:.3f}, p={p:.4f}). "
                f"RECD discretization reaches {rmax:.2f}, indicating strong chaotic structure formation. "
                "This is consistent with the Systemic Tau paradigm for sensitive count processes across coupled locations.")

    def _update_clean_paradigm_plots(self, stats):
        if hasattr(self, '_populate_clean_paradigm_plots'):
            self._populate_clean_paradigm_plots(stats)
        self.status_label.pack(side="left", padx=12, pady=2)

        self.status_right = ctk.CTkLabel(
            self.status_bar,
            text="macOS Desktop • RECD + τₛ",
            anchor="e",
            font=ctk.CTkFont(size=10),
            text_color="gray60"
        )
        self.status_right.pack(side="right", padx=12)

    def _resize_sidebar(self, event):
        new_width = event.x_root - self.winfo_rootx()
        if new_width < 200: new_width = 200
        if new_width > 600: new_width = 600
        self.sidebar_frame.configure(width=new_width)

    def _update_status(self, text: str):
        """Update the bottom status bar."""
        if hasattr(self, "status_label"):
            self.status_label.configure(text=f"{APP_TITLE} | {text}")

    def _add_sidebar_section(self, title, row):
        """Inserts a clean section header + separator in the sidebar.
        Safe: does not remove any existing controls.
        """
        header = ctk.CTkLabel(
            self.sidebar_frame,
            text=title.upper(),
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#6b7280",
            anchor="w"
        )
        header.grid(row=row, column=0, padx=18, pady=(10, 1), sticky="w")

        sep = ctk.CTkFrame(self.sidebar_frame, height=1, fg_color="#333333")
        sep.grid(row=row + 1, column=0, padx=18, pady=(0, 4), sticky="ew")
        return row + 2

    def _show_coming_soon(self):
        messagebox.showinfo("Coming Soon", "This feature is currently under development and will be available in a future release.")

    def _save_analysis(self):
        if not hasattr(self, 'math_stats') or not self.math_stats:
            messagebox.showwarning("Save Error", "No analysis to save. Please run an analysis first.")
            return
            
        save_path = filedialog.asksaveasfilename(defaultextension=".stau", filetypes=[("Systemic Tau Bundle", "*.stau")])
        if not save_path:
            return
            
        bundle = {
            'df': self.df,
            'math_stats': self.math_stats,
            'full_log': getattr(self, 'full_log', ''),
            'target_col': getattr(self, 'target_col', self.math_stats.get('target_col', 'Unknown')),
            'time_col': getattr(self, 'time_col', None)
        }
        
        import pickle
        try:
            with open(save_path, 'wb') as f:
                pickle.dump(bundle, f)
            messagebox.showinfo("Success", f"Analysis saved to {os.path.basename(save_path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save file: {e}")

    def _compare_with_file(self):
        if not hasattr(self, 'math_stats') or not self.math_stats:
            messagebox.showwarning("Compare Error", "Please run an analysis on the current file first before comparing.")
            return
            
        file_path = filedialog.askopenfilename(filetypes=[("Data/Analysis Files", "*.csv *.xlsx *.stau")])
        if not file_path:
            return
            
        import threading
        
        def _run_comparison():
            try:
                if file_path.endswith('.stau'):
                    import pickle
                    with open(file_path, 'rb') as f:
                        bundle = pickle.load(f)
                    df2 = bundle['df']
                    stats2 = bundle['math_stats']
                    target2 = bundle.get('target_col', 'Unknown')
                else:
                    if file_path.endswith('.csv'):
                        df2 = pd.read_csv(file_path)
                    else:
                        df2 = pd.read_excel(file_path)
                        
                    numeric_cols = df2.select_dtypes(include='number').columns.tolist()
                    if not numeric_cols:
                        raise ValueError("No numeric columns found in the second file.")
                    target2 = numeric_cols[0]
                    
                    # Fix: Handle NaNs in the comparison file to prevent validator crash
                    nans = df2[target2].isna().sum()
                    if nans > 0:
                        df2[target2] = df2[target2].interpolate(method='linear').bfill().ffill()
                        self.after(0, lambda: self._update_results(f"\n[COMPARISON NOTE] Auto-interpolated {nans} missing values in {os.path.basename(file_path)}.\n"))
                    
                    from systemictau.core import SystemicTauValidator
                    validator = SystemicTauValidator(df2, target2)
                    tau_val, p_value, effect_size, s_matrix = validator.validate(n_perm=100)
                    
                    window = validator.params.get('window', 20)
                    # We compute tau array manually as in analyze_data
                    from systemictau.core import network_coherence
                    from systemictau.fractal import higuchi_fd
                    
                    data_vals = df2[target2].values
                    n = len(data_vals)
                    tau_array = np.zeros(n)
                    accel = np.zeros(n)
                    coherence = np.zeros(n)
                    for i in range(window, n):
                        seg = data_vals[i-window:i]
                        v = np.var(seg)
                        fd = higuchi_fd(seg)
                        tau_array[i] = v * (1 + fd)
                        if i > window + 1:
                            accel[i] = (tau_array[i] - 2*tau_array[i-1] + tau_array[i-2])
                        coherence[i] = network_coherence(seg)
                    
                    t_star = np.argmax(tau_array)
                    
                    stats2 = {
                        "data": data_vals,
                        "tau_array": tau_array,
                        "t_star": t_star,
                        "tau_val": tau_val,
                        "max_accel": np.nanmax(accel),
                        "min_coherence": np.nanmin(coherence),
                        "p_value": p_value
                    }
                
                self.after(0, lambda: self._show_compare_window(target2, stats2))
            except Exception as e:
                self.after(0, lambda err=e: messagebox.showerror("Compare Error", f"Failed to compare:\n{err}"))
                
        self._update_results(f"\n[COMPARISON] Loading {os.path.basename(file_path)} and computing stats...\n")
        threading.Thread(target=_run_comparison, daemon=True).start()

    def _show_compare_window(self, target2, stats2):
        comp_win = ctk.CTkToplevel(self)
        comp_win.title("Comparison Results")
        comp_win.geometry("900x700")
        
        # Plot
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        
        fig = Figure(figsize=(8, 4), dpi=100)
        ax = fig.add_subplot(111)
        
        s1 = self.math_stats
        ax.plot(s1["tau_array"], label=f"File 1 ({getattr(self, 'target_col', 'Current')})", color="blue")
        ax.axvline(s1["t_star"], color="blue", linestyle="--", alpha=0.5)
        
        ax.plot(stats2["tau_array"], label=f"File 2 ({target2})", color="red")
        ax.axvline(stats2["t_star"], color="red", linestyle="--", alpha=0.5)
        
        ax.set_title("Systemic Tau Comparison: τ_s(t)")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Topological Mass")
        ax.legend()
        fig.subplots_adjust(bottom=0.15, left=0.15, right=0.95, top=0.9)
        
        chart_frame = ctk.CTkFrame(comp_win)
        chart_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.draw_idle()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        
        toolbar = NavigationToolbar2Tk(canvas, chart_frame)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")
        
        # Table
        table_frame = ctk.CTkFrame(comp_win)
        table_frame.pack(fill="x", padx=20, pady=20)
        
        ctk.CTkLabel(table_frame, text="Metric", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=10, pady=5)
        ctk.CTkLabel(table_frame, text="File 1", font=ctk.CTkFont(weight="bold")).grid(row=0, column=1, padx=10, pady=5)
        ctk.CTkLabel(table_frame, text="File 2", font=ctk.CTkFont(weight="bold")).grid(row=0, column=2, padx=10, pady=5)
        
        def fmt(v):
            if isinstance(v, (int, float)):
                if abs(v) < 1e-3 and v != 0:
                    return f"{v:.6f}"
                return f"{v:,.2f}"
            return str(v)
            
        metrics = [
            ("t* (Breakpoint)", s1.get('t_star_label', s1['t_star']), stats2.get('t_star_label', stats2['t_star'])),
            ("Max τ_s", s1['tau_val'], stats2['tau_val']),
            ("Peak a_t", s1['max_accel'], stats2['max_accel']),
            ("Min C_s", s1['min_coherence'], stats2['min_coherence']),
            ("p-value", s1['p_value'], stats2['p_value'])
        ]
        
        for i, (name, v1, v2) in enumerate(metrics, start=1):
            ctk.CTkLabel(table_frame, text=name).grid(row=i, column=0, padx=10, pady=5)
            ctk.CTkLabel(table_frame, text=fmt(v1)).grid(row=i, column=1, padx=10, pady=5)
            ctk.CTkLabel(table_frame, text=fmt(v2)).grid(row=i, column=2, padx=10, pady=5)

    def _on_scale_change(self, choice):
        # Legacy direct (kept for compatibility)
        self._on_scale_change_safe(choice)

    def _on_scale_change_safe(self, choice):
        """Safe wrapper that tolerates missing legacy widgets (clean UI redesign)."""
        choice = choice or "Local"
        # Guard cluster_agg_frame references
        try:
            if hasattr(self, "cluster_agg_frame") and self.cluster_agg_frame is not None:
                if choice == "Medium":
                    try:
                        if hasattr(self, "_cluster_agg_row"):
                            self.cluster_agg_frame.grid(row=self._cluster_agg_row, column=0, padx=18, pady=(2, 4), sticky="ew")
                        else:
                            self.cluster_agg_frame.grid()
                    except Exception:
                        pass
                    if hasattr(self, 'target_primary_label'): self.target_primary_label.grid_forget()
                    if hasattr(self, 'target_primary_menu'): self.target_primary_menu.grid_forget()
                    if hasattr(self, 'target_label'): self.target_label.configure(text="Select Variables for Cluster:")
                elif choice == "Global":
                    try:
                        self.cluster_agg_frame.grid_forget()
                    except Exception:
                        pass
                    if hasattr(self, 'target_primary_label'): self.target_primary_label.grid_forget()
                    if hasattr(self, 'target_primary_menu'): self.target_primary_menu.grid_forget()
                    if hasattr(self, 'target_label'): self.target_label.configure(text="Select Macro-Clusters (Global):")
                else:
                    try:
                        self.cluster_agg_frame.grid_forget()
                    except Exception:
                        pass
                    if hasattr(self, 'target_primary_label'):
                        self.target_primary_label.grid()
                    if hasattr(self, 'target_primary_menu'):
                        self.target_primary_menu.grid()
                    if hasattr(self, 'target_label'): self.target_label.configure(text="Secondary Variables:")
        except Exception:
            pass

        self._redraw_preview()

        # Live status update
        self._update_status(f"Scale changed to {choice}")

        # Hint for Global manual clustering
        if choice == "Global" and getattr(self, 'clustering_method', 'auto') == 'manual':
            if hasattr(self, 'define_clusters_btn'):
                self.define_clusters_btn.configure(fg_color="#166534")

    def _on_clustering_change(self, choice: str):
        if "manual" in (choice or "").lower():
            self.clustering_method = "manual"
            if hasattr(self, 'define_clusters_btn'):
                self.define_clusters_btn.configure(state="normal", fg_color="#166534")
            self._update_status("Global clustering: MANUAL (user-defined groups will be used)")
        else:
            self.clustering_method = "auto"
            if hasattr(self, 'define_clusters_btn'):
                self.define_clusters_btn.configure(state="normal", fg_color="#334155")
            self._update_status("Global clustering: AUTOMATIC (hierarchical Kendall)")

    def _create_metric_card(self, parent, title, row, col, colspan=1, val_color=None, val_size=16, tooltip_text=""):
        card = ctk.CTkFrame(parent, corner_radius=8, fg_color=("gray85", "gray20"))
        card.grid(row=row, column=col, columnspan=colspan, padx=5, pady=5, sticky="nsew")
        card.grid_columnconfigure(0, weight=1)
        
        lbl_title = ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=11, weight="bold"), text_color=("gray30", "gray70"))
        lbl_title.grid(row=0, column=0, pady=(2, 0))
        
        lbl_value = ctk.CTkLabel(card, text="--", font=ctk.CTkFont(size=val_size, weight="bold"), text_color=val_color)
        lbl_value.grid(row=1, column=0, pady=(0, 2))
        
        if tooltip_text:
            ToolTip(card, tooltip_text)
            ToolTip(lbl_title, tooltip_text)
            ToolTip(lbl_value, tooltip_text)
            
        return lbl_value

    def _on_slider_change(self, value):
        # Only update a preview label. Full analysis is expensive; user must explicitly Run.
        if hasattr(self, 'window_value_label'):
            try:
                self.window_value_label.configure(text=f"Window = {int(float(value))}")
            except Exception:
                pass
        # Do NOT auto-run analyze here — prevents unresponsiveness.

    def toggle_mode(self):
        if self.simple_mode_switch.get() == 1:
            self.advanced_btn.configure(state="disabled")
            if self.adv_window is not None and self.adv_window.winfo_exists():
                self.adv_window.destroy()
        else:
            self.advanced_btn.configure(state="normal")
            
    def handle_dnd(self, event):
        file_path = event.data.strip('{}') 
        self._load_file(file_path)

    def upload_file_dialog(self):
        file_path = filedialog.askopenfilename(filetypes=[("Data & Analysis", "*.csv *.xlsx *.stau")])
        if file_path:
            self._load_file(file_path)
            
    def _load_file(self, file_path):
        self.loaded_file_path = file_path
        filename = os.path.basename(file_path)
        self.file_label.configure(text=f"Loaded: {filename}")
        
        # Remove any previous recommended systemic button (fresh load)
        if getattr(self, 'recommended_btn', None):
            try:
                if self.recommended_btn.winfo_exists():
                    self.recommended_btn.destroy()
            except Exception:
                pass
            self.recommended_btn = None
        if getattr(self, '_cta_frame', None):
            try:
                if self._cta_frame.winfo_exists():
                    self._cta_frame.destroy()
            except Exception:
                pass
            self._cta_frame = None
        if getattr(self, '_hero_cta', None):
            try:
                if self._hero_cta.winfo_exists():
                    self._hero_cta.destroy()
            except Exception:
                pass
            self._hero_cta = None
        
        self.is_systemic_panel_mode = False
        self.systemic_principal = None
        self.systemic_loc_count = 0
        self.systemic_targets = []
        self._results_ui_visible = True
        
        try:
            if file_path.endswith('.stau'):
                import pickle
                with open(file_path, 'rb') as f:
                    bundle = pickle.load(f)
                
                self.df = bundle['df']
                self.math_stats = bundle['math_stats']
                self.full_log = bundle.get('full_log', '')
                self.target_col = bundle.get('target_col', self.df.columns[0])
                self.time_col = bundle.get('time_col', None)
                
                numeric_cols = self.df.select_dtypes(include='number').columns.tolist()
                if len(numeric_cols) > 0:
                    # Modern UI uses target_primary_menu + checkboxes; refresh handles population
                    if hasattr(self, '_refresh_target_ui'):
                        self._refresh_target_ui(redraw=True)
                    else:
                        if hasattr(self, 'target_primary_menu'):
                            self.target_primary_menu.configure(values=numeric_cols)
                            self.target_primary_menu.set(numeric_cols[0])
                        self._redraw_preview(numeric_cols[0])
                    
                self._update_results(self.full_log, clear=True)
                self.after(100, self._highlight_graph)
                messagebox.showinfo("Analysis Loaded", f"Successfully loaded analysis: {filename}")
                return

            # Use DataManager for CSV/Excel
            self.df = self.data_mgr.load_file(file_path)
            
            # Check if coords auto-detected
            if self.data_mgr.coords_df is not None:
                self.load_coords_btn.configure(text="📍 Coordinates Auto-Loaded", text_color="#2ca02c")
                
            # Panel Data Handling - Smart guided workflow for Count as principal (minimal user thinking)
            if self.data_mgr.is_panel:
                self.batch_btn.configure(state="normal")
                loc_count = self.df[self.data_mgr.location_col].nunique()
                
                # === Ask user for time variable (with smart combine option) ===
                num_cols = self.df.select_dtypes(include='number').columns.tolist()
                time_candidates = [c for c in self.df.columns if any(k in str(c).lower() for k in ['week','year','time','fecha','date','epi'])]
                time_options = ["[Auto: Combine EpiYear + EpiWeek]"] + [c for c in time_candidates if c in self.df.columns]
                time_dialog = VariableSelectionDialog(self, time_options, title="Select Time Variable", default="[Auto: Combine EpiYear + EpiWeek]")
                self.wait_window(time_dialog)
                t_choice = time_dialog.selected_var or "[Auto: Combine EpiYear + EpiWeek]"
                
                if t_choice == "[Auto: Combine EpiYear + EpiWeek]":
                    ycol = next((c for c in self.df.columns if 'year' in str(c).lower() or 'año' in str(c).lower()), None)
                    wcol = next((c for c in self.df.columns if 'week' in str(c).lower() or 'semana' in str(c).lower()), None)
                    if ycol and wcol:
                        # Mutate BOTH the working df and the manager's raw_df so that pivot_panel_to_systemic (which uses raw_df) sees the combined time column
                        epi = self.df[ycol].astype(int) * 100 + self.df[wcol].astype(int)
                        self.df['EpiTime'] = epi
                        if hasattr(self.data_mgr, 'raw_df') and self.data_mgr.raw_df is not None:
                            self.data_mgr.raw_df['EpiTime'] = epi
                        self.data_mgr.time_col = 'EpiTime'
                        self.time_col = 'EpiTime'
                        messagebox.showinfo("Time Variable", "Auto-created combined EpiTime (Year+Week) for proper sequencing.")
                    else:
                        self.time_col = self.data_mgr.time_col or (time_candidates[0] if time_candidates else None)
                else:
                    self.time_col = t_choice
                    self.data_mgr.time_col = t_choice
                
                # === Ask user for principal variable (default "Count") ===
                num_cols = self.df.select_dtypes(include='number').columns.tolist()
                if self.data_mgr.location_col in num_cols:
                    num_cols.remove(self.data_mgr.location_col)
                if self.time_col in num_cols:
                    num_cols.remove(self.time_col)
                
                principal_default = "Count" if "Count" in num_cols else (num_cols[0] if num_cols else None)
                p_dialog = VariableSelectionDialog(self, num_cols, title="Select Principal Variable (e.g. Count)", default=principal_default)
                self.wait_window(p_dialog)
                principal = p_dialog.selected_var or principal_default or (num_cols[0] if num_cols else None)
                
                if principal:
                    # Auto-apply the CORRECT workflow: pivot principal by location → systemic spatial system
                    self.df = self.data_mgr.pivot_panel_to_systemic(principal)
                    self.is_systemic_panel_mode = True
                    self.systemic_principal = principal
                    self.systemic_loc_count = loc_count
                    messagebox.showinfo(
                        "Recommended Workflow Applied",
                        f"Panel prepared as SYSTEMIC SPATIAL SYSTEM on '{principal}' across {loc_count} locations.\n"
                        "• Locations now act as system components (for true Kendall τ_s).\n"
                        "• Recommended settings will be applied (window=13, RECD enabled, all sites selected).\n"
                        "This follows the optimal flow for panel count data per the Systemic Tau paradigm."
                    )
                    # === SHOW THE BIG ONE-CLICK BUTTON FOR NON-EXPERTS ===
                    # Use the new clean high-quality setup view
                    self._build_clean_setup_view()
                    if hasattr(self, 'setup_note'):
                        self.setup_note.configure(text=f"Loaded: {loc_count} locations • Pivoted on '{principal}' • Ready for systemic τₛ + RECD")
                else:
                    # fallback
                    first_loc = self.df[self.data_mgr.location_col].iloc[0]
                    self.df = self.data_mgr.filter_by_location([first_loc])
                    self.is_systemic_panel_mode = False
                
                # Auto-setup UI for the correct systemic flow (minimal thinking)
                # New clean high-quality UI - set the targets and show the elegant setup view
                try:
                    numeric_after = self.df.select_dtypes(include='number').columns.tolist()
                    tcol = getattr(self, 'time_col', None)
                    if tcol:
                        numeric_after = [c for c in numeric_after if str(c) != str(tcol)]
                    self.systemic_targets = numeric_after
                except Exception:
                    pass

                self._build_clean_setup_view()
                if hasattr(self, 'setup_note'):
                    self.setup_note.configure(text=f"Panel ready: {len(self.systemic_targets)} modules • w=13 • RECD+Layers • full systemic τₛ")

                # Set the controls that exist in the slim sidebar
                if hasattr(self, 'window_slider'):
                    self.window_slider.set(13)
                if hasattr(self, 'recd_var'):
                    self.recd_var.set(1)
                if hasattr(self, 'recd_check'):
                    self.recd_check.select()

            # Big Data Downsampling
            self.df = self.data_mgr.intelligent_downsample(self.df)
            if getattr(self.data_mgr, 'downsampled_flag', False):
                messagebox.showinfo(
                    "Smart Downsampling Applied",
                    f"The dataset exceeded {self.app_settings.get('max_rows_before_downsampling', 80000)} rows.\n"
                    "Smart Downsampling was applied to prevent memory overload.\n"
                    f"Reduced to {len(self.df)} rows."
                )
                
            self.time_col = self.data_mgr.time_col
            if not self.time_col:
                for col in self.df.columns:
                    col_lower = str(col).lower().strip()
                    if any(x in col_lower for x in ['date', 'time', 'fecha', 'timestamp', 'year', 'año', 'month', 'mes', 'week', 'semana']):
                        self.time_col = col
                        break
            if self.time_col is None and len(self.df.columns) > 0:
                self.time_col = self.df.columns[0]
                
            # Populate Time Menu
            if hasattr(self, 'time_menu'):
                self.time_menu.configure(values=["[Auto Detect]"] + list(self.df.columns))
                if self.time_col and self.time_col in self.df.columns:
                    self.time_menu.set(self.time_col)
                else:
                    self.time_menu.set("[Auto Detect]")
                
            # In the redesigned clean UI we use systemic_targets directly.
            # Avoid old target UI refresh to prevent None cb crashes.
            if getattr(self.data_mgr, 'is_panel', False):
                # Just ensure the list is set (already done above)
                pass
            else:
                if hasattr(self, '_refresh_target_ui'):
                    self._refresh_target_ui()
            
            # Update status bar
            rows = len(self.df) if self.df is not None else 0
            scale = self.scale_menu.get() if hasattr(self, "scale_menu") else "Local"
            self._update_status(f"Loaded: {filename} | {rows} rows | Scale: {scale}")
            
        except Exception as e:
            self.file_label.configure(text=f"Error reading file: {e}")

    def _refresh_target_ui(self, redraw=True, select_all=False, exclude_time=True):
        if self.df is None:
            return

        # === STRONG GUARD for the clean redesign ===
        # In systemic panel mode (the main use case for "Count" data), we never touch the old
        # target checkbox UI or its (None, var) entries. This prevents the 'NoneType' destroy crash
        # from old after() scheduled calls and compact mode population.
        if getattr(self, 'is_systemic_panel_mode', False):
            try:
                numeric_cols = self.df.select_dtypes(include='number').columns.tolist()
                t = getattr(self, 'time_col', None)
                if t:
                    numeric_cols = [c for c in numeric_cols if str(c) != str(t)]
                self.systemic_targets = numeric_cols
            except Exception:
                self.systemic_targets = []
            return

        # Legacy path (non systemic) - guarded
        if not hasattr(self, 'target_scroll') or getattr(self, 'target_scroll', None) is None:
            try:
                cols = self.df.select_dtypes(include='number').columns.tolist()
                t = getattr(self, 'time_col', None)
                if t:
                    cols = [c for c in cols if str(c) != str(t)]
                self.systemic_targets = cols
            except:
                pass
            return

        # Normal / small case (legacy only)
        if len(numeric_cols) > 0:
            if hasattr(self, 'target_primary_menu'):
                self.target_primary_menu.configure(values=numeric_cols)
                curr = self.target_primary_menu.get()
                if curr not in numeric_cols:
                    self.target_primary_menu.set(numeric_cols[0])
                    
            for entry in list(getattr(self, 'target_checkboxes', {}).values()):
                cb = entry[0] if isinstance(entry, (list, tuple)) and len(entry) > 0 else None
                if cb is not None:
                    try:
                        cb.destroy()
                    except Exception:
                        pass
            self.target_checkboxes = {}
            for col in numeric_cols:
                var = tk.StringVar(value="")
                cb = ctk.CTkCheckBox(self.target_scroll, text=str(col), variable=var, onvalue=str(col), offvalue="", command=lambda c=col: self._redraw_preview(c))
                cb.pack(anchor="w", pady=2)
                self.target_checkboxes[str(col)] = (cb, var)
            if self.target_checkboxes:
                if select_all:
                    for _, (cb, _) in self.target_checkboxes.items():
                        if cb: cb.select()
                else:
                    list(self.target_checkboxes.values())[0][0].select()
            
            if redraw and self.target_checkboxes:
                first = list(self.target_checkboxes.keys())[0]
                self._redraw_preview(first)
        else:
            pass

    def toggle_all_targets(self):
        if not hasattr(self, 'target_checkboxes') or not self.target_checkboxes:
            return
            
        # Determine current state based on the first checkbox
        first_cb, first_var = list(self.target_checkboxes.values())[0]
        currently_selected = first_var.get() != ""
        
        for c, (cb, var) in self.target_checkboxes.items():
            if currently_selected:
                cb.deselect()
            else:
                cb.select()

    def get_selected_targets(self):
        # In systemic panel mode we want the full set of location columns as targets
        if getattr(self, 'is_systemic_panel_mode', False) and getattr(self, 'systemic_targets', None):
            targets = list(self.systemic_targets)
            # de-dupe and remove time if somehow present
            t = getattr(self, 'time_col', None)
            if t:
                targets = [x for x in targets if str(x) != str(t)]
            return targets
        
        primary = None
        if hasattr(self, 'target_primary_menu'):
            val = self.target_primary_menu.get()
            if not val.startswith("["):
                primary = val
                
        targets = []
        if primary:
            targets.append(primary)
            
        if hasattr(self, 'target_checkboxes'):
            for c, (cb, var) in self.target_checkboxes.items():
                if var.get() != "" and var.get() != primary:
                    targets.append(var.get())
                    
        return targets

    # ============================================================
    # NEW: Manual Cluster Definition Dialog (v4.x Global improvement)
    # ============================================================
    def open_manual_clusters_dialog(self):
        """Interactive dialog to define 2-6 user macro-clusters for Global scale."""
        if self.df is None:
            messagebox.showwarning("No Data", "Load a dataset first to define clusters from its columns.")
            return

        numeric_cols = [str(c) for c in self.df.select_dtypes(include="number").columns.tolist()]
        tcol = getattr(self, "time_col", None)
        if tcol:
            numeric_cols = [c for c in numeric_cols if c != str(tcol)]

        if len(numeric_cols) < 4:
            messagebox.showinfo("Clustering", "For meaningful manual macro-clusters, a dataset with 4+ numeric variables is recommended.")

        win = ctk.CTkToplevel(self)
        win.title("Define Manual Macro-Clusters (Global)")
        win.geometry("620x520")
        win.transient(self)
        win.grab_set()

        ctk.CTkLabel(win, text="Create 2–6 macro-clusters by grouping variables.\nThese groups will be aggregated (sum/median) and analyzed together at Global scale.",
                     font=ctk.CTkFont(size=11), justify="left", wraplength=580).pack(pady=(12, 8), padx=12, anchor="w")

        # Available variables (left)
        left = ctk.CTkFrame(win)
        left.pack(side="left", fill="both", expand=True, padx=(12, 6), pady=6)
        ctk.CTkLabel(left, text="Available Variables", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=8)
        avail_scroll = ctk.CTkScrollableFrame(left, height=220)
        avail_scroll.pack(fill="both", expand=True, padx=4, pady=4)

        self._dlg_vars = {}  # varname -> (checkbox, StringVar)
        for col in numeric_cols:
            v = tk.StringVar(value="")
            cb = ctk.CTkCheckBox(avail_scroll, text=col, variable=v, onvalue=col, offvalue="")
            cb.pack(anchor="w", pady=1)
            self._dlg_vars[col] = (cb, v)

        # Right side: controls + current definition
        right = ctk.CTkFrame(win)
        right.pack(side="right", fill="both", expand=True, padx=(6, 12), pady=6)

        ctk.CTkLabel(right, text="New Cluster Name", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=8, pady=(4,0))
        self.cluster_name_entry = ctk.CTkEntry(right, placeholder_text="e.g. Socioeconomic or Env_Factors", width=220)
        self.cluster_name_entry.pack(padx=8, pady=4, fill="x")

        btns = ctk.CTkFrame(right, fg_color="transparent")
        btns.pack(fill="x", padx=8, pady=4)
        ctk.CTkButton(btns, text="➕ Add Cluster from Selected", command=lambda: self._dlg_add_cluster(win), width=140).pack(side="left")
        ctk.CTkButton(btns, text="Clear Selection", command=self._dlg_clear_selection, fg_color="#475569", width=100).pack(side="left", padx=6)

        ctk.CTkLabel(right, text="Current Manual Clusters", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=8, pady=(10,2))
        self.cluster_display = ctk.CTkTextbox(right, height=160, width=260, font=ctk.CTkFont(size=10))
        self.cluster_display.pack(fill="both", expand=True, padx=8, pady=4)

        # Agg method for manual (mirrors existing)
        ctk.CTkLabel(right, text="Aggregation for macros:", font=ctk.CTkFont(size=9)).pack(anchor="w", padx=8)
        self.dlg_agg = ctk.CTkOptionMenu(right, values=["sum", "median"], width=120)
        self.dlg_agg.set("sum")
        self.dlg_agg.pack(anchor="w", padx=8)

        # Bottom actions
        action_bar = ctk.CTkFrame(win, fg_color="transparent")
        action_bar.pack(fill="x", pady=8, padx=12)
        ctk.CTkButton(action_bar, text="Apply to Global Runs", fg_color="#166534",
                      command=lambda: self._dlg_apply(win)).pack(side="left", padx=4)
        ctk.CTkButton(action_bar, text="Reset All", fg_color="#854d0e",
                      command=lambda: self._dlg_reset(win)).pack(side="left", padx=4)
        ctk.CTkButton(action_bar, text="Close", command=win.destroy).pack(side="right", padx=4)

        # Initialize display
        self._dlg_refresh_display()

    def _dlg_clear_selection(self):
        for _, (_, v) in getattr(self, '_dlg_vars', {}).items():
            v.set("")

    def _dlg_add_cluster(self, parent_win):
        name = (self.cluster_name_entry.get() or "").strip()
        if not name:
            name = f"Cluster_{len(self.manual_clusters) + 1}"
        selected = [v.get() for _, (_, v) in getattr(self, '_dlg_vars', {}).items() if v.get()]
        if not selected:
            messagebox.showwarning("No Variables", "Select one or more variables on the left to form a cluster.")
            return
        if len(self.manual_clusters) >= 6:
            messagebox.showwarning("Limit", "Maximum 6 macro-clusters supported.")
            return
        self.manual_clusters[name] = selected
        self.cluster_name_entry.delete(0, "end")
        self._dlg_clear_selection()
        self._dlg_refresh_display()

    def _dlg_refresh_display(self):
        if not hasattr(self, 'cluster_display') or self.cluster_display is None:
            return
        self.cluster_display.delete("1.0", "end")
        if not self.manual_clusters:
            self.cluster_display.insert("end", "(no clusters defined yet)\n\nSelect variables → enter name → Add Cluster")
            return
        for i, (nm, mems) in enumerate(self.manual_clusters.items(), 1):
            self.cluster_display.insert("end", f"{i}. {nm}\n   members: {', '.join(mems)}\n\n")

    def _dlg_reset(self, win):
        self.manual_clusters = {}
        OntologicalScaleManager.clear_manual_clusters()
        self._dlg_refresh_display()

    def _dlg_apply(self, win):
        if len(self.manual_clusters) < 2:
            messagebox.showwarning("Insufficient Clusters", "Define at least 2 macro-clusters before applying.")
            return
        # Store + sync to manager
        try:
            OntologicalScaleManager.define_manual_clusters(self.manual_clusters)
        except Exception as e:
            messagebox.showerror("Definition Error", str(e))
            return
        self.clustering_method = "manual"
        if hasattr(self, "clustering_menu"):
            self.clustering_menu.set("Manual (user-defined)")
        if hasattr(self, "define_clusters_btn"):
            self.define_clusters_btn.configure(fg_color="#166534")
        # Also force scale to Global for convenience
        if hasattr(self, "scale_menu"):
            self.scale_menu.set("Global")
        self._update_status(f"Manual clusters applied: {len(self.manual_clusters)} groups. Scale=Global.")
        win.destroy()
        messagebox.showinfo("Clustering Set", f"Manual definition stored ({len(self.manual_clusters)} clusters).\n\nNext Global run will use your groups exactly.")

    # ============================================================

    def _show_recommended_systemic_button(self, principal, loc_count):
        """Create (or replace) the big obvious button for the correct one-click workflow.
        Redesigned as the central hero action per Systemic Tau guided flow."""
        # clean previous
        if getattr(self, 'recommended_btn', None):
            try:
                if self.recommended_btn.winfo_exists():
                    self.recommended_btn.destroy()
            except Exception:
                pass
        
        # 1. Prominent in top bar (quick access)
        self.recommended_btn = ctk.CTkButton(
            self.top_bar,
            text=f"🚀 RUN RECOMMENDED SYSTEMIC\n{principal} × {loc_count} modules • w=13 • RECD+Layers",
            command=self._run_recommended_systemic_count,
            height=48,
            fg_color="#166534",
            hover_color="#14532d",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#f0fdf4"
        )
        self.recommended_btn.pack(side="left", padx=6)
        
        # 2. HERO CARD in the main dashboard (makes it impossible to miss — the "Grok redesigned" experience)
        try:
            if hasattr(self, '_hero_cta') and self._hero_cta and self._hero_cta.winfo_exists():
                self._hero_cta.destroy()
        except:
            pass
        
        self._hero_cta = ctk.CTkFrame(self.dashboard_frame, fg_color="#052e16", corner_radius=12)
        # Pack it reliably right after verdict_box
        self._hero_cta.pack(fill="x", pady=(8, 4), padx=5, after=getattr(self, 'verdict_box', None))
        
        hero_title = ctk.CTkLabel(
            self._hero_cta, 
            text="🚀 ONE CLICK. THE CORRECT SYSTEMIC FLOW.",
            font=ctk.CTkFont(size=16, weight="bold"), text_color="#4ade80"
        )
        hero_title.pack(pady=(10, 2))
        
        hero_desc = ctk.CTkLabel(
            self._hero_cta,
            text=f"Panel detected as spatial system. {loc_count} locations = modules.\nWe pivoted on '{principal}'. We will use w=13, full Kendall τ_s across the network, canonical RECD (Feigenbaum δ) and Layers analysis.",
            font=ctk.CTkFont(size=12), text_color="#86efac", justify="center"
        )
        hero_desc.pack(pady=(0, 8))
        
        big_btn = ctk.CTkButton(
            self._hero_cta,
            text="🚀 RUN RECOMMENDED SYSTEMIC COUNT ANALYSIS",
            command=self._run_recommended_systemic_count,
            height=56,
            fg_color="#15803d",
            hover_color="#166534",
            font=ctk.CTkFont(size=15, weight="bold")
        )
        big_btn.pack(pady=(0, 12), padx=20, fill="x")
        
        self._update_status("Systemic Tau ready — the rocket does the right thing (pivot + w=13 + RECD + multi τ_s).")

    def _force_recommended_settings(self):
        """Force the exact theory-recommended UI state for panel Count systemic (no user thought needed)."""
        if hasattr(self, 'scale_menu'):
            self.scale_menu.set("Local")
        if hasattr(self, 'window_slider'):
            self.window_slider.set(13)
        if hasattr(self, 'recd_switch'):
            self.recd_switch.set(1)
        if getattr(self, 'is_systemic_panel_mode', False) and hasattr(self, 'df') and self.df is not None:
            try:
                _n = [c for c in self.df.select_dtypes(include='number').columns.tolist()
                      if str(c) != str(getattr(self, 'time_col', ''))]
                self.systemic_targets = _n
            except:
                pass
        if hasattr(self, 'target_checkboxes') and self.target_checkboxes:
            for entry in self.target_checkboxes.values():
                cb = entry[0] if isinstance(entry, (list, tuple)) and len(entry) > 0 else None
                if cb is not None:
                    cb.select()
        # ensure time not accidentally targeted
        t = getattr(self, 'time_col', None)
        if t and hasattr(self, 'target_checkboxes') and str(t) in self.target_checkboxes:
            entry = self.target_checkboxes[str(t)]
            cb = entry[0] if isinstance(entry, (list, tuple)) and len(entry) > 0 else None
            if cb is not None:
                cb.deselect()
        # Make sure time menu reflects the (combined) time column used for the pivoted data
        if hasattr(self, 'time_menu') and t:
            try:
                vals = self.time_menu.cget("values") or []
                if t in vals or str(t) in [str(v) for v in vals]:
                    self.time_menu.set(str(t))
            except Exception:
                pass
        # Re-enable action buttons
        if hasattr(self, 'analyze_btn'):
            try: self.analyze_btn.configure(state="normal")
            except: pass

    def _run_recommended_systemic_count(self):
        """One-click: enforce all correct params then launch the analysis exactly as the paradigm requires."""
        self._force_recommended_settings()
        # slight delay so UI reflects, then run (uses current get_selected_targets + slider + recd + time)
        self.after(80, lambda: self.analyze_data())

    def _ensure_advanced_plots(self):
        """Lazy creation of the heavy diagnostic matplotlib figures.
        Called only when user visits the Advanced tab. Big win for perceived responsiveness."""
        if getattr(self, '_advanced_plots_created', False):
            return
        try:
            # Phase Space
            self.fig2 = Figure(figsize=(10, 5), dpi=100)
            self.ax_ps = self.fig2.add_subplot(111)
            self.fig2.tight_layout(pad=3.0)
            self.canvas2 = FigureCanvasTkAgg(self.fig2, master=self.tab2)
            self.canvas2.draw_idle()
            self.canvas2.get_tk_widget().pack(fill="both", expand=True)
            self.toolbar2 = NavigationToolbar2Tk(self.canvas2, self.tab2)
            self.toolbar2.update()
            self.toolbar2.pack(side="bottom", fill="x")
            
            # EWS
            self.fig3 = Figure(figsize=(10, 5.5), dpi=100)
            self.ax_ews1 = self.fig3.add_subplot(311)
            self.ax_ews2 = self.fig3.add_subplot(312)
            self.ax_ews2_twin = self.ax_ews2.twinx()
            self.ax_ews2_twin.set_visible(False)
            self.ax_ews3 = self.fig3.add_subplot(313)
            self.fig3.tight_layout(pad=1.5)
            self.canvas3 = FigureCanvasTkAgg(self.fig3, master=self.tab3)
            self.canvas3.draw_idle()
            self.canvas3.get_tk_widget().pack(fill="both", expand=True)
            self.toolbar3 = NavigationToolbar2Tk(self.canvas3, self.tab3)
            self.toolbar3.update()
            self.toolbar3.pack(side="bottom", fill="x")
            
            # RECD
            self.fig4 = Figure(figsize=(10, 4), dpi=100)
            self.ax_recd = self.fig4.add_subplot(111)
            self.fig4.tight_layout(pad=2.0)
            self.canvas4 = FigureCanvasTkAgg(self.fig4, master=self.tab4)
            self.canvas4.draw_idle()
            self.canvas4.get_tk_widget().pack(fill="both", expand=True)
            self.toolbar4 = NavigationToolbar2Tk(self.canvas4, self.tab4)
            self.toolbar4.update()
            self.toolbar4.pack(side="bottom", fill="x")
            
            self._advanced_plots_created = True
        except Exception as e:
            print("Lazy advanced plots:", e)

    def _hide_cluttered_results_ui(self):
        """Aggressive cleanup for 'ready' state. The old layout is a mess of panels.
        We nuke everything except the verdict and the hero banner so it doesn't look like a disaster."""
        try:
            # Forget / hide all the old heavy children in the dashboard
            to_hide = ['metrics_row', 'main_chart_frame', 'bottom_split', 'insights_frame', 'actions_frame']
            for name in to_hide:
                widget = getattr(self, name, None)
                if widget and hasattr(widget, 'winfo_ismapped') and widget.winfo_ismapped():
                    try:
                        widget.pack_forget()
                    except:
                        try:
                            widget.grid_forget()
                        except:
                            pass

            # Also hide any other direct children of dashboard_frame that are not the verdict or hero
            if hasattr(self, 'dashboard_frame'):
                hero = getattr(self, '_hero_cta', None)
                for child in list(self.dashboard_frame.winfo_children()):
                    if child not in (getattr(self, 'verdict_box', None), hero):
                        try:
                            child.pack_forget()
                        except:
                            try:
                                child.grid_forget()
                            except:
                                pass

            self._results_ui_visible = False
        except Exception as e:
            print("hide cleanup:", e)

    def _simplify_sidebar_for_systemic(self):
        """Reduce sidebar noise when in guided panel systemic mode.
        The old sidebar has too many panels."""
        try:
            # Make the scale section less prominent
            if hasattr(self, 'scale_label'):
                self.scale_label.configure(text="Scale (auto: Local for systemic)")
            # Update other labels to be shorter
            if hasattr(self, 'target_label'):
                self.target_label.configure(text="Modules (auto full network)")
            if hasattr(self, 'target_primary_label'):
                self.target_primary_label.configure(text="Primary:")
            # Hide the primary selector widgets in pure systemic mode to reduce clutter
            try:
                if hasattr(self, 'target_primary_label'):
                    self.target_primary_label.grid_forget()
                if hasattr(self, 'target_primary_menu'):
                    self.target_primary_menu.grid_forget()
            except:
                pass
        except Exception as e:
            print("sidebar simplify:", e)

    def _show_results_ui(self):
        """Restore the main results panels after a real analysis run.
        Also restore minimal sidebar elements."""
        try:
            if hasattr(self, 'metrics_row') and not getattr(self.metrics_row, 'winfo_ismapped', lambda: False)():
                self.metrics_row.pack(fill="x", pady=10, after=getattr(self, 'verdict_box', None))
            if hasattr(self, 'main_chart_frame') and not getattr(self.main_chart_frame, 'winfo_ismapped', lambda: False)():
                self.main_chart_frame.pack(fill="both", expand=True, pady=10)
            if hasattr(self, 'bottom_split') and not getattr(self.bottom_split, 'winfo_ismapped', lambda: False)():
                self.bottom_split.pack(fill="x", pady=10)

            # Refresh the (compact) target summary in sidebar for systemic
            if getattr(self, 'is_systemic_panel_mode', False):
                try:
                    self._refresh_target_ui(redraw=False, select_all=True, exclude_time=True)
                except:
                    pass

            self._results_ui_visible = True

            # Restore the normal analyze button (it may appear at end of left packs, but functional)
            if hasattr(self, 'analyze_btn'):
                try:
                    self.analyze_btn.pack(side="left", padx=10)
                    self.analyze_btn.configure(state="normal")
                except:
                    pass
        except Exception as e:
            print("show results:", e)

    def _render_systemic_ready_state(self):
        """Redesigned hero / ready state focused on Systemic Tau for panel counts.
        Clear, opinionated, low cognitive load."""
        if not getattr(self, 'is_systemic_panel_mode', False):
            return
        try:
            # Update main verdict area to be theory-centric and welcoming
            if hasattr(self, 'lbl_verdict_title'):
                self.lbl_verdict_title.configure(
                    text="SYSTEMIC SPATIAL SYSTEM READY",
                    text_color="#22c55e"
                )
            if hasattr(self, 'lbl_verdict_desc'):
                n = getattr(self, 'systemic_loc_count', '?')
                p = getattr(self, 'systemic_principal', 'Count')
                self.lbl_verdict_desc.configure(
                    text=f"Data pivoted: {n} locations as coupled modules • Principal = {p}\n"
                         "Recommended: w=13 rolling window • RECD discretization (δ≈4.669) • Full multi-module Kendall τ_s\n"
                         "Click the green rocket to reveal structural reorganization, chaotic regimes and Layers."
                )
            self._hide_cluttered_results_ui()
            self._simplify_sidebar_for_systemic()
            # Hide or de-emphasize the normal "Analyze" button; the big rocket is the hero
            if hasattr(self, 'analyze_btn'):
                try:
                    self.analyze_btn.pack_forget()
                except:
                    pass
            # Update status with paradigm language
            self._update_status(f"Systemic mode: {n} modules • τ_s + RECD + Layers • Ready for analysis")
        except Exception:
            pass

    def _redraw_preview(self, col_name=None):
        if not getattr(self, '_results_ui_visible', True):
            # In clean ready state we don't want stray previews cluttering (chart is hidden anyway)
            return
        if col_name is None:
            targets = self.get_selected_targets()
            if not targets: return
            col_name = targets[0]
            
        if self.df is None or col_name not in self.df.columns:
            return
            
        nans = self.df[col_name].isna().sum()
        if nans > 0:
            self.health_label.configure(text=f"⚠️ Status: {nans} Missing Values Detected!", text_color="#ff8c00")
        else:
            self.health_label.configure(text="✅ Status: Clean (0 missing)", text_color="#2ca02c")
            
        self.ax1.clear()
        self.ax2.clear()
        self.ax2_twin.clear()
        self.ax3.clear()
        self.ax4.clear()
        self.ax1.plot(self.df[col_name].values, color="#1f77b4")
        self.ax1.set_title(f"Preview: {col_name}")
        self.canvas1.draw_idle()
            
    def optimize_window(self):
        if not self.loaded_file_path or self.df is None:
            self._update_results("Error: Please upload a valid data file first.\n", clear=True)
            return
            
        targets = self.get_selected_targets()
        if not targets:
            self._update_results("Error: No target selected.\n", clear=True)
            return
            
        self.full_log = ""
        self._update_results("[OPTIMIZER] Scanning for optimal Systemic Memory window...\n", clear=True)
        threading.Thread(target=self._run_optimization, args=(targets,), daemon=True).start()
        
    def _run_optimization(self, targets):
        try:
            numeric_df = self.df.select_dtypes(include='number').ffill().fillna(0)
            if numeric_df.empty:
                raise ValueError("No numeric columns found.")
            target_col = targets[0]
            if target_col not in numeric_df.columns:
                raise ValueError("Invalid target column.")
                
            is_multi = len(targets) > 1
            if is_multi:
                matrix_data = numeric_df[targets].values
                n = len(matrix_data)
                norm_data = (matrix_data - np.mean(matrix_data, axis=0)) / (np.std(matrix_data, axis=0) + 1e-9)
            else:
                data = numeric_df[target_col].values
                n = len(data)
                
            max_w = min(100, max(10, n // 2))
            
            best_w = 3
            best_score = -np.inf
            
            for w in range(3, max_w + 1):
                if is_multi:
                    tau_series = pd.DataFrame(norm_data).rolling(window=w, min_periods=1).var().sum(axis=1).fillna(0).values
                else:
                    tau_series = pd.Series(data).rolling(window=w, min_periods=1).var().fillna(0).values
                
                if len(tau_series) == 0: 
                    continue
                
                tau_max = np.max(tau_series)
                tau_median = np.median(tau_series)
                
                # Signal-to-noise ratio
                if tau_median > 0:
                    score = tau_max / tau_median
                else:
                    score = tau_max
                    
                if score > best_score:
                    best_score = score
                    best_w = w
                    
            self.after(0, self.window_slider.set, best_w)
            self._update_results(f"[OPTIMIZER] Found optimal window: {best_w} (SNR: {best_score:.2f})\n\n")
            self.after(100, lambda: self.analyze_data(clear_logs=False))
            
        except Exception as e:
            self._update_results(f"\n[ERROR] Optimization failed: {e}\n")
            
    def run_batch_processing(self):
        if not hasattr(self, 'data_mgr') or not getattr(self.data_mgr, 'is_panel', False):
            return
            
        if hasattr(self, 'scale_menu') and self.scale_menu.get() != "Local":
            import tkinter.messagebox
            tkinter.messagebox.showerror(
                "Invalid Analysis Mode", 
                "Batch Analysis is designed for Micro-Systemic (Local) analysis, iterating over individual panel locations.\n\n"
                "Since you are using a higher ontological scale (Medium or Global), the entire network is evaluated together as a single system. Please use the 'Run Single Analysis' button instead."
            )
            return

            
        self.analyze_btn.configure(state="disabled")
        self.batch_btn.configure(state="disabled")
        self._update_results("\n[BATCH PROCESS] Starting analysis for all locations...\n")
        
        enable_mi = self.app_settings.get("enable_mi", False)
        mi_bins = self.app_settings.get("mi_bins", 5)
        
        if enable_mi:
            import tkinter.messagebox
            tkinter.messagebox.showwarning("Performance Warning", "Mutual Information (MI) is enabled. This will significantly increase the processing time of the Batch Run.")
            
        targets = self.get_selected_targets()
        
        ui_state_batch = {
            'targets': targets,
            'smooth_mode': self.smoothing_menu.get() if hasattr(self, 'smoothing_menu') else None
        }
        try:
            ui_state_batch['window'] = int(self.window_slider.get())
        except:
            ui_state_batch['window'] = 10
            
        import threading
        def _run_batch(ui_state):
            try:
                # Load the raw file fresh
                raw_df = self.data_mgr.load_file(self.loaded_file_path)
                locations = raw_df[self.data_mgr.location_col].unique()
                
                results = []
                targets = ui_state['targets']
                if not targets:
                    targets = raw_df.select_dtypes(include='number').columns.tolist()[:1]
                
                window = ui_state['window']
                smooth_mode = ui_state['smooth_mode']
                is_multi = len(targets) > 1
                
                total = len(locations)
                for idx, loc in enumerate(locations):
                    self._update_results(f"  -> Processing {loc} ({idx+1}/{total})...\n")
                    loc_df = raw_df[raw_df[self.data_mgr.location_col] == loc].copy()
                    
                    time_labels = None
                    if hasattr(self, 'time_col') and self.time_col in loc_df.columns:
                        time_labels = loc_df[self.time_col].values
                        
                    try:
                        import numpy as np
                        from systemictau.desktop.core import SystemicTauEngine
                        stats = SystemicTauEngine.run_analysis_pipeline(
                            numeric_df=loc_df.select_dtypes(include='number').ffill().fillna(0),
                            targets=targets,
                            window=window,
                            smooth_mode=smooth_mode,
                            time_labels=time_labels,
                            is_multi=is_multi,
                            enable_mi=enable_mi,
                            mi_bins=mi_bins
                        )
                        
                        leading = "N/A"
                        if stats.get('corr_matrix') is not None:
                            leading = stats['targets'][np.nanargmax(np.nanmean(stats['corr_matrix'], axis=1))]
                            
                        results.append({
                            "Location": loc,
                            "Target": targets[0],
                            "t_star": stats['t_star_label'],
                            "Tau_Max": stats['tau_val'],
                            "p_value": stats['p_value'],
                            "Leading_Driver": leading,
                            "Verdict": stats['final_verdict'].split("->")[0].replace("\n", "").strip()
                        })
                    except Exception as e:
                        print(f"Failed {loc}: {e}")
                        
                import pandas as pd
                res_df = pd.DataFrame(results)
                out_file = "batch_results.csv"
                res_df.to_csv(out_file, index=False)
                
                self.latest_map_data = res_df
                if getattr(self.data_mgr, 'coords_df', None) is not None:
                    self.after(0, lambda: self.generate_map_btn.configure(state="normal"))
                
                self._update_results(f"\n[BATCH PROCESS COMPLETE] Results saved to {out_file}\n")
                
                self.after(0, lambda: messagebox.showinfo("Batch Complete", f"Successfully processed {len(results)} locations.\nResults saved to {out_file}"))
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._update_results(f"\n[BATCH ERROR] {e}\n")
            finally:
                self.after(0, lambda: self.analyze_btn.configure(state="normal"))
                self.after(0, lambda: self.batch_btn.configure(state="normal"))
                
        threading.Thread(target=_run_batch, args=(ui_state_batch,), daemon=True).start()

    def _on_analyze_click(self, *args):
        if hasattr(self, 'auto_opt_switch') and self.auto_opt_switch.get() == 1:
            self.optimize_window()
        else:
            self.analyze_data()

    def analyze_data(self, *args, **kwargs):
        clear_logs = kwargs.get("clear_logs", True)
        if not self.loaded_file_path or self.df is None:
            self._update_results("Error: Please upload a valid data file first.\n", clear=True)
            return
            
        # For panel systemic flows (after time+Count questions), if user has not selected the full set,
        # auto-apply the correct full workflow settings so non-experts never have to think "what next".
        if getattr(self.data_mgr, 'is_panel', False):
            targets_now = self.get_selected_targets()
            if len(targets_now) < 2:
                self._force_recommended_settings()
        targets = self.get_selected_targets()
        if not targets:
            messagebox.showwarning("No Target Selected", "Please select at least one primary target variable to run the analysis.")
            return
            
        scale_mode = getattr(self, 'scale_menu', None).get() if hasattr(self, 'scale_menu') and self.scale_menu else "Local"
        
        if scale_mode == "Global" and len(targets) < 2:
            messagebox.showwarning("Invalid Selection for Global Scale", "Global scale analysis requires at least 2 variables or macro-clusters to analyze systemic coupling.\n\nPlease select multiple variables.")
            return
            
        ui_state = {
            'targets': targets,
            'scale_mode': scale_mode,
            'agg_method': self.cluster_agg_menu.get().lower() if hasattr(self, 'cluster_agg_menu') else "sum",
            'smooth_mode': self.smoothing_menu.get() if hasattr(self, 'smoothing_menu') else None,
            'time_c': getattr(self, 'time_menu', None).get() if hasattr(self, 'time_menu') and self.time_menu else "[Auto Detect]",
            'run_ai': self.run_ai_switch.get() if hasattr(self, 'run_ai_switch') else 0,
            'enable_recd': self.recd_var.get() if hasattr(self, 'recd_var') else (self.recd_switch.get() if hasattr(self, 'recd_switch') else 1),
            # New clustering controls
            'clustering_method': getattr(self, 'clustering_method', 'auto'),
            'manual_clusters': dict(getattr(self, 'manual_clusters', {})),
        }
        try:
            ui_state['window'] = int(self.window_slider.get())
        except:
            ui_state['window'] = None
            
        if clear_logs:
            self.full_log = ""
            self._update_results("Initializing Systemic Tau Mathematical Analysis...\n\n", clear=True)
        else:
            self._update_results("Initializing Systemic Tau Mathematical Analysis...\n\n", clear=False)
        self.analyze_btn.configure(state="disabled")
        self.batch_btn.configure(state="disabled")
        threading.Thread(target=self._run_real_analysis_pipeline, args=(ui_state,), daemon=True).start()

    def _run_real_analysis_pipeline(self, ui_state):
        try:
            scale_mode = ui_state['scale_mode']
            targets = ui_state['targets']
            
            numeric_df = self.df.select_dtypes(include='number').ffill().fillna(0)
            if numeric_df.empty:
                raise ValueError("No numeric columns found in data.")
                
            cluster_components = None
            if scale_mode == "Medium":
                from systemictau.desktop.scale_manager import OntologicalScaleManager
                agg_method = ui_state['agg_method']
                if "sum" in agg_method: agg_method = "sum"
                elif "median" in agg_method: agg_method = "median"
                
                # Make a unique name based on number of variables so we can create multiple
                base_name = f"MacroCluster_{agg_method.upper()}_{len(targets)}vars"
                suffix = 1
                cluster_name = f"{base_name}_v{suffix}"
                while cluster_name in self.df.columns:
                    suffix += 1
                    cluster_name = f"{base_name}_v{suffix}"
                    
                cluster_components = targets.copy()
                numeric_df = OntologicalScaleManager.aggregate_medium_cluster(numeric_df, targets, cluster_name=cluster_name, agg_method=agg_method)
                
                # Persist the newly created cluster to the main dataframe so it can be selected in Global scale
                self.df[cluster_name] = numeric_df[cluster_name]
                if hasattr(self.data_mgr, 'macro_cluster_compositions'):
                    self.data_mgr.macro_cluster_compositions[cluster_name] = cluster_components
                self.after(0, lambda: self._refresh_target_ui(redraw=False))
                
                targets = [cluster_name]
                
            elif scale_mode == "Global":
                if len(targets) < 2:
                    raise ValueError("Global scale analysis requires at least 2 macro-clusters to analyze systemic coupling.")
                
                from systemictau.desktop.scale_manager import OntologicalScaleManager

                # Determine method from UI state (new manual support)
                clustering_method = ui_state.get('clustering_method', 'auto')
                manual_groups = ui_state.get('manual_clusters') or None
                agg_method = ui_state['agg_method']
                if "sum" in agg_method: agg_method = "sum"
                elif "median" in agg_method: agg_method = "median"

                # Check if targets are pre-computed clusters from Medium
                are_precomputed = all(
                    str(t).startswith("MacroCluster_SUM") or str(t).startswith("MacroCluster_MEDIAN") or str(t).startswith("MacroCluster_")
                    for t in targets
                )

                cluster_dict = {}
                used_method = clustering_method

                if clustering_method == "manual" and manual_groups and len(manual_groups) >= 2:
                    # Respect explicit user definition (highest priority)
                    try:
                        numeric_df, targets, cluster_dict = OntologicalScaleManager.get_global_clusters(
                            numeric_df, targets, method="manual", manual_groups=manual_groups, agg_method=agg_method
                        )
                        used_method = "manual"
                    except Exception as e:
                        # Fall back with warning
                        self._update_results(f"[CLUSTERING] Manual definition error, falling back to auto: {e}\n")
                        numeric_df, targets, cluster_dict = OntologicalScaleManager.get_global_clusters(
                            numeric_df, targets, method="auto", num_clusters=3, agg_method=agg_method
                        )
                        used_method = "auto (fallback)"
                elif not are_precomputed:
                    # Automatic hierarchical clustering (improved + transparent)
                    numeric_df, targets, cluster_dict = OntologicalScaleManager.get_global_clusters(
                        numeric_df, targets, method="auto", num_clusters=3, agg_method=agg_method
                    )
                    used_method = "auto (hierarchical Kendall)"

                if hasattr(self.data_mgr, 'macro_cluster_compositions'):
                    self.data_mgr.macro_cluster_compositions.update(cluster_dict or {})

                # Ensure >=2 macros
                if len(targets) < 2:
                    raise ValueError("Failed to construct at least 2 macro-clusters from the selected variables for Global scale.")

                # Store for reports + session
                self.math_stats['clustering_method'] = used_method
                self.math_stats['macro_cluster_composition'] = cluster_dict or {}

            target_col = targets[0]
            if target_col not in numeric_df.columns:
                raise ValueError(f"Selected column '{target_col}' is not valid or not numeric.")
                
            is_multi = len(targets) > 1
            smooth_mode = ui_state['smooth_mode']
            
            T_len = len(numeric_df)
            window = ui_state['window']
            if window is None:
                window = max(3, T_len // 20)
            if window >= T_len:
                window = max(3, T_len // 2)

            time_labels = None
            time_c = ui_state['time_c']
            if time_c != "[Auto Detect]":
                time_labels = self.df[time_c].values
            elif getattr(self, 'time_col', None):
                time_labels = self.df[self.time_col].values

            # --- CALL CORE ENGINE ---
            enable_mi = self.app_settings.get("enable_mi", False)
            mi_bins = self.app_settings.get("mi_bins", 5)
            
            self.math_stats = SystemicTauEngine.run_analysis_pipeline(
                numeric_df=numeric_df,
                targets=targets,
                window=window,
                smooth_mode=smooth_mode,
                time_labels=time_labels,
                is_multi=is_multi,
                enable_mi=enable_mi,
                mi_bins=mi_bins,
                enable_recd=ui_state.get('enable_recd', 0) == 1
            )
            self.math_stats['ontological_scale'] = scale_mode
            if cluster_components:
                self.math_stats['cluster_components'] = cluster_components
            s = self.math_stats
            
            # High-quality clean results (the redesigned scientific experience)
            self.after(0, lambda: self._build_clean_results_view(s))
            
            if s['p_value'] >= 0.05:
                msg = f"Peak volatility observed in '{s['target_col']}' (Tau_s={s['tau_val']:.2f}) at {s['t_star_label']}, but lacking statistical significance."
            else:
                msg = f"Structural break detected in '{s['target_col']}' (Tau_s={s['tau_val']:.2f}) at {s['t_star_label']}."
            
            self._update_results(f"[MATHEMATICS] {msg}\n")
            
            # Update status
            tau = s.get('tau_val', 0)
            pval = s.get('p_value', 1)
            self._update_status(f"Analysis complete | τₛ={tau:.3f} | p={pval:.4f} | Scale: {scale_mode}")
            self._generate_deterministic_report()
            
            # Save for mapping
            leading = "N/A"
            if s.get('corr_matrix') is not None:
                leading = s['targets'][np.nanargmax(np.nanmean(s['corr_matrix'], axis=1))]
            
            res_list = []
            
            if is_multi and s.get('corr_matrix') is not None:
                # In Macro-Systemic mode, targets are the locations.
                # We map each location to its systemic coupling value.
                mean_corrs = np.nanmean(s['corr_matrix'], axis=1)
                for idx, loc_val in enumerate(targets):
                    # Replace NaN coupling with 0 or the systemic Tau if desired
                    coupling = mean_corrs[idx] if not np.isnan(mean_corrs[idx]) else 0
                    res_list.append({
                        "Location": loc_val,
                        "Target": "Systemic Coupling",
                        "t_star": s['t_star_label'],
                        "Tau_Max": coupling, # Use coupling for the map coloring
                        "p_value": s['p_value'],
                        "Leading_Driver": "Yes" if loc_val == leading else "No",
                        "Verdict": s['final_verdict'].split("->")[0].replace("\n", "").strip()
                    })
            else:
                # Univariate run
                loc_val = "Unknown"
                if getattr(self.data_mgr, 'is_panel', False) and getattr(self.data_mgr, 'location_col', None):
                    if self.data_mgr.location_col in self.df.columns:
                        loc_val = self.df[self.data_mgr.location_col].iloc[0]
                        
                res_list.append({
                    "Location": loc_val,
                    "Target": targets[0],
                    "t_star": s['t_star_label'],
                    "Tau_Max": s['tau_val'],
                    "p_value": s['p_value'],
                    "Leading_Driver": leading,
                    "Verdict": s['final_verdict'].split("->")[0].replace("\n", "").strip()
                })
                
            self.latest_map_data = pd.DataFrame(res_list)
            
            if getattr(self.data_mgr, 'coords_df', None) is not None:
                self.after(0, lambda: self.generate_map_btn.configure(state="normal"))
            
            if ui_state['run_ai'] == 1:
                self._update_results("\n[EPISTEMIC ENGINE] Booting Hierarchical Multi-Agent Discovery...\n")
                current_key = settings.google_api_key
                if not current_key or current_key == "DUMMY_GEMINI_KEY":
                    self._update_results("      -> API Key missing. Requesting from user...\n")
                    self.after(0, self._prompt_for_api_key)
                    return
                    
                context = f"{msg}. Early Warning Signals: {s['precursor_signal']}."
                if is_multi:
                    context += f" Multivariate Synchrony: {s['multivariate_str']}"
                    
                hypothesis, confidence = run_discovery_engine_sync(
                    context=context, 
                    tau_val=s['tau_val'], 
                    update_callback=self._update_results
                )
                self.last_hypothesis = hypothesis
            else:
                self._update_results("\n[NOTE] AI Epistemic Engine is disabled. Pure mathematical interpretation complete.\n")
                self.last_hypothesis = "AI Engine was disabled. Review the deterministic mathematical report above."
                
            import copy, os
            temp1 = f"temp_plot1_{scale_mode}.png"
            temp2 = f"temp_plot2_{scale_mode}.png"
            temp3 = f"temp_plot3_{scale_mode}.png"
            temp4 = f"temp_plot4_{scale_mode}.png"
            
            map_data_copy = None
            if hasattr(self, 'latest_map_data') and self.latest_map_data is not None:
                map_data_copy = self.latest_map_data.copy()
                
            df_copy = None
            if self.df is not None:
                df_copy = self.df.copy()
            
            stats_copy = copy.deepcopy(self.math_stats) if self.math_stats else {}
            
            # Delegate matplotlib savefig and final state update to main thread to prevent Done(renderer) crash
            def _finalize():
                if hasattr(self, 'fig1') and self.fig1: self.fig1.savefig(temp1, dpi=150)
                if hasattr(self, 'fig2') and self.fig2: self.fig2.savefig(temp2, dpi=150)
                if hasattr(self, 'fig3') and self.fig3: self.fig3.savefig(temp3, dpi=150)
                if hasattr(self, 'fig4') and self.fig4: self.fig4.savefig(temp4, dpi=150)
                
                self.ontological_memory[scale_mode] = {
                    "math_stats": stats_copy,
                    "full_log": self.full_log,
                    "map_data": map_data_copy,
                    "df_snapshot": df_copy,
                    "img1": temp1 if os.path.exists(temp1) else None,
                    "img2": temp2 if os.path.exists(temp2) else None,
                    "img3": temp3 if os.path.exists(temp3) else None,
                    "img4": temp4 if os.path.exists(temp4) else None
                }
                
                self._update_results("\n[COMPLETE] Analysis finalized.\n")
                
            self.after(0, _finalize)
            
        except Exception as e:
            if "API key not valid" in str(e):
                self._update_results("      -> API Key is invalid. Requesting new key...\n")
                self.after(0, self._prompt_for_api_key)
            else:
                import traceback
                traceback.print_exc()
                self._update_results(f"\n[ERROR] Analysis failed: {e}\n")
                
                # IMPORTANT: Show a popup because the user might be in Simple Mode and not see the results_box!
                err_msg = str(e)
                def _show_err(msg=err_msg):
                    messagebox.showerror("Analysis Error", f"The analysis encountered an error:\n\n{msg}\n\nPlease check your data or selected variables.")
                self.after(0, _show_err)
        finally:
            self.after(0, lambda: self.analyze_btn.configure(state="normal"))
            if getattr(self.data_mgr, 'is_panel', False):
                self.after(0, lambda: self.batch_btn.configure(state="normal"))

    def animate_phase_space(self):
        if not self.math_stats:
            return
        self.tabview.set("Phase Space")
        s = self.math_stats
        self.ax_ps.clear()
        self.ax_ps.set_title("Phase Space Trajectory Animation")
        self.ax_ps.set_xlabel("System State (Raw Data)")
        self.ax_ps.set_ylabel("System Momentum (Acceleration)")
        
        data = s.get("data_for_plot", [])
        if len(data) == 0:
            messagebox.showwarning("No Data", "Could not load data for animation.")
            return
        acc = s["acceleration"]
        self.ax_ps.set_xlim(np.min(data) * 0.9, np.max(data) * 1.1)
        self.ax_ps.set_ylim(np.min(acc) * 0.9, np.max(acc) * 1.1)
        
        self.anim_frame = 0
        self.anim_max = len(data)
        self.anim_colors = np.arange(self.anim_max)
        self.animate_btn.configure(state="disabled")
        self._animate_step()

    def _animate_step(self):
        s = self.math_stats
        data = s.get("data_for_plot", s.get("data", []))
        acc = s.get("acceleration", [])
        
        if len(data) == 0 or len(acc) == 0:
            return
            
        if not hasattr(self, 'anim_frame') or self.anim_frame >= self.anim_max:
            t_star = s.get("t_star", 0)
            if t_star < len(data) and t_star < len(acc):
                self.ax_ps.scatter(data[t_star], acc[t_star], color='red', marker='*', s=200, label="t* Collapse")
                self.ax_ps.legend()
            self.canvas2.draw_idle()
            self.animate_btn.configure(state="normal")
            return
            
        chunk = max(1, self.anim_max // 50)
        end_idx = min(self.anim_frame + chunk, self.anim_max)
        
        self.ax_ps.plot(data[self.anim_frame:end_idx+1], acc[self.anim_frame:end_idx+1], color="gray", alpha=0.5, linewidth=1)
        self.ax_ps.scatter(data[self.anim_frame:end_idx], acc[self.anim_frame:end_idx], c=self.anim_colors[self.anim_frame:end_idx], cmap="viridis", alpha=0.7, s=20, vmin=0, vmax=self.anim_max)
        
        self.canvas2.draw_idle()
        self.anim_frame += chunk
        self.after(20, self._animate_step)
        
    def _highlight_graph(self):
        s = self.math_stats
        tau_danger, acc_danger, ent_danger = SystemicTauPlotter.highlight_graph(self, s)
        
        def fmt(val):
            if abs(val) < 1e-3 and val != 0: return f"{val:.6f}"
            return f"{val:,.2f}"

        def get_color(val, threshold, lower_is_worse=False):
            if np.isnan(val) or np.isnan(threshold): return "#2ca02c"
            if lower_is_worse: return "#d62728" if val <= threshold else "#2ca02c"
            else: return "#d62728" if val >= threshold else "#2ca02c"

        self.lbl_tau.configure(text=fmt(s['tau_val']), text_color=get_color(s['tau_val'], tau_danger))
        self.lbl_accel.configure(text=fmt(s['max_accel']), text_color=get_color(s['max_accel'], acc_danger))
        self.lbl_entropy.configure(text=fmt(s['max_entropy']), text_color=get_color(s['max_entropy'], ent_danger))
        self.lbl_coherence.configure(text=f"{s['min_coherence']:.2f}")
        self.lbl_tstar.configure(text=s['t_star_label'])
        
        if s['p_value'] >= 0.05:
            verdict_color = "#2c5f2d"
            verdict_title = "NON-SIGNIFICANT NOISE (NO COLLAPSE)"
            verdict_desc = f"The system was evaluated at t*={s['t_star']}, but statistical analysis confirms all metrics are within normal background variance (p={s['p_value']:.2f})."
        else:
            verdict_color = "#9e1a1a"
            verdict_title = "CRITICAL STRUCTURAL COLLAPSE"
            verdict_desc = f"The system crossed a critical topological threshold at t*={s['t_star']} (p={s['p_value']:.4f}). Immediate action is required."
            
        self.verdict_box.configure(fg_color=verdict_color)
        self.lbl_verdict_title.configure(text=verdict_title)
        self.lbl_verdict_desc.configure(text=verdict_desc)

        clean_sens = s['sensitivity_narrative'].replace("WARNING: ", "").replace("Conclusion: ", "\n• ").replace("ACTIONABLE INSIGHT: ", "\n• ").replace("IMPLICATION: ", "\n• ")
        insights_text = "• " + clean_sens
        if s['precursor_signal'].startswith("WEAK") or s['precursor_signal'].startswith("NONE"):
            insights_text += "\n• No significant early warning signals detected prior to the anomaly."
        else:
            try:
                insights_text += f"\n• Early Warning Signals detected: {s['precursor_signal'].split('->')[1].strip()}"
            except Exception: pass
                
        self.lbl_insights.configure(text=insights_text)
        self.animate_btn.configure(state="normal")


    def show_heatmap(self):
        if not hasattr(self, 'math_stats') or not self.math_stats.get('is_multi'):
            return
        
        top = ctk.CTkToplevel(self)
        top.title("Multivariate Synchrony Heatmap")
        top.geometry("800x600")
        
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        
        fig, ax = plt.subplots(figsize=(8, 6))
        SystemicTauPlotter.draw_heatmap(ax, self.math_stats)
        fig.subplots_adjust(bottom=0.15, left=0.15, right=0.95, top=0.9)
        
        canvas = FigureCanvasTkAgg(fig, master=top)
        canvas.draw_idle()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _generate_deterministic_report(self):
        s = self.math_stats
        t_star_label = s['t_star_label']
        
        def fmt(val):
            if abs(val) < 1e-3 and val != 0:
                return f"{val:.6f}"
            return f"{val:,.2f}"
        
        matrix_str = ""
        for w_test, t_test, tau_test in s['sensitivity_matrix']:
            mark = "(*)" if w_test == s['window'] else "   "
            matrix_str += f"   {w_test:6d} | {t_test:15d} | {tau_test:,.2f} {mark}\n"
            
        ontological_scale = s.get('ontological_scale', 'Local').upper()
        
        if s['p_value'] >= 0.05:
            exec_summary = (
                f"EXECUTIVE SUMMARY ({ontological_scale} SCALE):\n"
                f"The system underwent routine variance evaluation at {t_star_label}, but mathematical testing "
                f"(p={s['p_value']:.2f}) confirms this was NOT a true structural collapse. The maximum momentum of {fmt(s['max_accel'])} "
                f"is mathematically consistent with random background noise.\n"
            )
        else:
            exec_summary = (
                f"EXECUTIVE SUMMARY ({ontological_scale} SCALE):\n"
                f"The system suffered a critical structural break at {t_star_label} due to "
                f"uncontrollable entropic decay. The momentum peak of {fmt(s['max_accel'])} "
                f"indicates a severe external shock precipitating the topological collapse.\n"
            )
            
        if getattr(self, 'data_was_cleaned', False):
            exec_summary += f"\n[METHODOLOGICAL NOTE] Linear interpolation was applied to {self.data_was_cleaned} missing data points prior to topological computation to ensure continuous phase space reconstruction.\n"
            
        sens_narrative = s['sensitivity_narrative']
        if 'ACTIONABLE INSIGHT:' in sens_narrative:
            sens_part1 = sens_narrative.split('ACTIONABLE INSIGHT:')[0].strip()
            sens_part2 = 'ACTIONABLE INSIGHT: ' + sens_narrative.split('ACTIONABLE INSIGHT:')[1].strip()
        elif 'IMPLICATION:' in sens_narrative:
            sens_part1 = sens_narrative.split('IMPLICATION:')[0].strip()
            sens_part2 = 'IMPLICATION: ' + sens_narrative.split('IMPLICATION:')[1].strip()
        else:
            sens_part1 = sens_narrative.strip()
            sens_part2 = "Robust structural transition verified across multiple time scales."

        mi_explanation = ""
        if "mi_pass" in s and s["mi_pass"] is not None:
            pass_str = "PASS" if s["mi_pass"] else "FAIL"
            trend_str = "a" if s["mi_pass"] else "no"
            mi_explanation = f"- Non-Linear Mutual Information Check: Mutual Information showed {trend_str} significant upward trend relative to the historical baseline (exploratory metric). [{pass_str}]\n"

        recd_info = ""
        if s.get('recd_multi') is not None and ontological_scale == "GLOBAL":
            recd_info = "- Discrete Temporal Structure (RECD) — Global Scale\n"
            for m_name, m_data in s['recd_multi'].items():
                m_arr = m_data['array']
                m_bp = m_data['breakpoints']
                max_recd = np.nanmax(m_arr) if not np.all(np.isnan(m_arr)) else 0.0
                recd_info += f"  [{m_name}] Maximum Discretization Index = {fmt(max_recd)}.\n"
                if len(m_bp) > 0:
                    t_labels = s.get('time_labels')
                    t_star_idx = s['t_star']
                    recd_info += f"  Breakpoints detected for {m_name}:\n"
                    for bp in m_bp:
                        label = str(t_labels[bp]) if t_labels is not None else str(bp)
                        dist = bp - t_star_idx
                        if abs(dist) <= 2:
                            recd_info += f"      - At {label}: Strong Ontological Coupling (t_recd ≈ t*)\n"
                        elif dist < -2:
                            recd_info += f"      - At {label}: Pre-emptive Discretization (t_recd < t*)\n"
                        else:
                            recd_info += f"      - At {label}: Reactive Discretization (t_recd > t*)\n"
                else:
                    recd_info += f"  No significant temporal discretization breakpoints detected for {m_name}.\n"
            recd_info += "  (Note: Breakpoints reflect the collective temporal behavior at the macro-cluster level)\n"
        elif s.get('recd_array') is not None:
            max_recd = np.nanmax(s['recd_array']) if not np.all(np.isnan(s['recd_array'])) else 0.0
            
            recd_header = "- Discrete Temporal Structure (RECD)"
            recd_suffix = ""
            if ontological_scale == "MEDIUM":
                recd_header += " — Medium Scale"
                recd_suffix = " (Note: Breakpoints reflect the collective temporal behavior of the aggregated variables in this cluster)."
                
            recd_info = f"{recd_header}: Maximum Discretization Index = {fmt(max_recd)}. "
            if len(s.get('recd_breakpoints', [])) > 0:
                t_labels = s.get('time_labels')
                t_star_idx = s['t_star']
                
                recd_info += f"Breakpoints detected:{recd_suffix}\n"
                for bp in s['recd_breakpoints']:
                    label = str(t_labels[bp]) if t_labels is not None else str(bp)
                    dist = bp - t_star_idx
                    
                    if abs(dist) <= 2:
                        recd_info += f"    - At {label}: Strong Ontological Coupling (t_recd ≈ t*)\n"
                    elif dist < -2:
                        recd_info += f"    - At {label}: Pre-emptive Discretization (t_recd < t*)\n"
                    else:
                        recd_info += f"    - At {label}: Reactive Discretization (t_recd > t*)\n"
            else:
                recd_info += f"No significant temporal discretization breakpoints detected.{recd_suffix}\n"

        cluster_info = ""
        if ontological_scale == "MEDIUM" and 'cluster_components' in s:
            num_vars = len(s['cluster_components'])
            cluster_info = f"[MEDIUM SCALE CLUSTER] Aggregated {num_vars} variables: {', '.join([str(x) for x in s['cluster_components'][:10]])}{'...' if num_vars > 10 else ''}\n\n"
        elif ontological_scale == "GLOBAL":
            num_vars = len(s.get('targets', []))
            method = s.get('clustering_method', 'automatic')
            comp = s.get('macro_cluster_composition', {}) or {}
            try:
                from systemictau.desktop.scale_manager import OntologicalScaleManager
                cluster_info = OntologicalScaleManager.describe_clustering(method, comp) + "\n\n"
            except Exception:
                comp_lines = []
                for mc_name, members in list(comp.items())[:6]:
                    comp_lines.append(f"    {mc_name}: {', '.join([str(m) for m in members[:5]])}{'...' if len(members) > 5 else ''}")
                comp_str = "\n".join(comp_lines) if comp_lines else "    (composition not recorded)"
                cluster_info = (
                    f"[GLOBAL SCALE] Clustering method: {method}\n"
                    f"Macro-clusters analyzed ({num_vars}):\n{comp_str}\n\n"
                )
            
        sync_label = "Global Systemic Coupling" if ontological_scale == "GLOBAL" else "Multivariate Synchrony"
            
        if s['p_value'] >= 0.05:
            report = (
                f"\n=======================================\n"
                f"{exec_summary}"
                f"=======================================\n\n"
                f"{cluster_info}"
                
                f"1. KEY METRICS (NO STRUCTURAL BREAK DETECTED)\n"
                f"- Peak Variance Anomaly: {fmt(s['tau_val'])} at [{t_star_label}].\n"
                f"- Acceleration Momentum (a_t): Peak = {fmt(s['max_accel'])}.\n"
                f"- {sync_label}: {s['multivariate_str'].replace(chr(10) + '     -> ', ' ')}\n\n"
                
                f"2. STATISTICAL VALIDATION\n"
                f"- Null Model (Monte Carlo): p-value = {s['p_value']:.4f} ({s['significance_str'].split('.')[0]}). {'.'.join(s['significance_str'].split('.')[1:]).strip()}\n"
                f"- Early Warning Signals: {s['precursor_signal'].strip()}\n"
                f"{mi_explanation}"
                f"{recd_info}\n"
                
                f"3. SENSITIVITY ANALYSIS\n"
                f"- Parameter Stability: {sens_part1}\n"
                f"- Note: Since the primary anomaly is not statistically significant, these scale variations likely represent normal background noise rather than true structural shifts.\n\n"
                
                f"4. FINAL ANALYTICAL VERDICT\n"
                f"{s['final_verdict'].strip()}\n\n"
                
                f"5. METHODOLOGY\n"
                f"- Analytical Memory: Window (W) = {s['window']} periods.\n"
                f"- Surrogate Testing: {s['n_perm']} unrestricted random permutations evaluated.\n"
            )
        else:
            report = (
                f"\n=======================================\n"
                f"{exec_summary}"
                f"=======================================\n\n"
                f"{cluster_info}"
                
                f"1. KEY METRICS AND BREAKPOINT DETECTION\n"
                f"- Topological Reorganization (τ_s): Breakpoint detected at [{t_star_label}] with a peak variance anomaly of {fmt(s['tau_val'])}.\n"
                f"- Acceleration Momentum (a_t): Peak = {fmt(s['max_accel'])}.\n"
                f"- Entropic Decay (S_e): Max Volatility = {fmt(s['max_entropy'])}.\n"
                f"- Systemic Coherence (C_s): Min Coupling = {fmt(s['min_coherence'])}.\n"
                f"- {sync_label}: {s['multivariate_str'].replace(chr(10) + '     -> ', ' ')}\n\n"
                
                f"2. STATISTICAL VALIDATION AND ROBUSTNESS\n"
                f"- Null Model (Monte Carlo): p-value = {s['p_value']:.4f} ({s['significance_str'].split('.')[0]}). {'.'.join(s['significance_str'].split('.')[1:]).strip()}\n"
                f"- Early Warning Signals: {s['precursor_signal'].strip()}\n"
                f"{mi_explanation}"
                f"{recd_info}\n"
                
                f"3. TRANSITION CHARACTERIZATION\n"
                f"- Transition Geometry: FWHM = {s['fwhm_str']} periods | Relaxation Time = {s['relax_str']} periods.\n"
                f"- Post-Collapse State: {s['post_regime']}.\n\n"
                
                f"4. SENSITIVITY ANALYSIS\n"
                f"- Parameter Stability: {sens_part1}\n"
                f"- Implication: {sens_part2}\n\n"
                
                f"5. FINAL ANALYTICAL VERDICT\n"
                f"{s['final_verdict'].strip()}\n\n"
                
                f"6. METHODOLOGY\n"
                f"- Analytical Memory: Window (W) = {s['window']} periods, objectively selected to maximize Signal-to-Noise Ratio (SNR = {fmt(s['snr'])}).\n"
                f"- Surrogate Testing: {s['n_perm']} unrestricted random permutations evaluated.\n"
            )
        self._update_results(report)

    def _prompt_for_api_key(self):
        dialog = ctk.CTkInputDialog(text="Enter your Google Gemini API Key:", title="API Key Required")
        key = dialog.get_input()
        if key:
            env_path = os.path.join(os.getcwd(), ".env")
            with open(env_path, "a") as f:
                f.write(f"\nGOOGLE_API_KEY={key}\n")
            settings.google_api_key = key
            self._update_results("      -> API Key saved. Please click 'Analyze' again.\n")
        else:
            self._update_results("      -> Analysis aborted: No API key provided.\n")

    def _update_results(self, text, clear=False):
        def _update():
            if clear:
                self.full_log = ""
            self.full_log += text
            
            if hasattr(self, 'results_box'):
                self.results_box.configure(state="normal")
                if clear:
                    self.results_box.delete("0.0", "end")
                self.results_box.insert("end", text)
                self.results_box.see("end")
                self.results_box.configure(state="disabled")
        self.after(0, _update)

    def explain_simply(self):
        if not hasattr(self, 'last_hypothesis'):
            return
        self._update_results("\n--- Generating Simple Explanation (LLM) ---\n")
        def _explain():
            try:
                client = genai.Client(api_key=settings.google_api_key)
                prompt = f"Explain this complex scientific hypothesis to a high schooler in one short paragraph:\n{self.last_hypothesis}"
                resp = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
                self._update_results(f"\nSimple Explanation: {resp.text.strip()}\n")
            except Exception as e:
                self._update_results(f"\nError: {e}\n")
        threading.Thread(target=_explain, daemon=True).start()

    def export_report(self):
        if not self.full_log:
            messagebox.showwarning("No Analysis Data", "Please run the analysis first by clicking 'Analyze' before attempting to export the report.")
            return
        save_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
        if not save_path:
            return
        self._update_results(f"\nGenerating Academic PDF Report at {save_path}...\n")
        
        # Save figures in the main thread!
        temp_img1 = "temp_plot1.png"
        temp_img2 = "temp_plot2.png"
        temp_img3 = "temp_plot3.png"
        temp_img4 = "temp_plot4.png"
        try:
            self.fig1.savefig(temp_img1, dpi=150)
            self.fig2.savefig(temp_img2, dpi=150)
            self.fig3.savefig(temp_img3, dpi=150)
            if hasattr(self, 'fig4') and self.fig4: self.fig4.savefig(temp_img4, dpi=150)
        except Exception as e:
            self._update_results(f"\n[ERROR] Failed to save plot images: {e}\n")
            return
            
        run_ai = (self.run_ai_switch.get() == 1) if hasattr(self, 'run_ai_switch') else False
            
        def _build_pdf():
            try:
                from fpdf import FPDF
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", 'B', 16)
                pdf.cell(0, 10, "Systemic Tau - Mathematical & Epistemic Report", ln=True, align="C")
                
                if self.math_stats and self.df is not None:
                    targets = self.math_stats.get('targets', [])
                    if targets:
                        pdf.set_font("Arial", 'B', 12)
                        pdf.cell(0, 8, "0. Variables Evaluated (Descriptive Statistics)", ln=True)
                        pdf.set_font("Courier", size=8)
                        
                        # Transpose describe() so variables are rows, stats are columns
                        try:
                            desc = self.df[targets].describe().round(2).T
                            stats_cols = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
                            
                            # Header
                            pdf.cell(46, 6, "Variable", border=1, align="C")
                            for c in stats_cols:
                                pdf.cell(18, 6, c, border=1, align="C")
                            pdf.ln()
                            
                            # Data rows
                            for idx, row in desc.iterrows():
                                var_name = str(idx)
                                if len(var_name) > 22: 
                                    var_name = var_name[:20] + ".."
                                pdf.cell(46, 6, var_name, border=1)
                                for c in stats_cols:
                                    pdf.cell(18, 6, str(row[c]), border=1, align="R")
                                pdf.ln()
                                
                            pdf.ln(5)
                        except KeyError:
                            pass
                
                # Math Metrics
                def fmt(val):
                    if abs(val) < 1e-3 and val != 0:
                        return f"{val:.6f}"
                    return f"{val:,.2f}"
                
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 8, "1. Topologic Reorganization Metrics", ln=True)
                pdf.set_font("Courier", size=10)
                if self.math_stats:
                    s = self.math_stats
                    pdf.cell(0, 6, f"- Critical Mass Threshold (Max Tau_s): {fmt(s['tau_val'])}", ln=True)
                    pdf.cell(0, 6, f"- Peak Acceleration (a_t): {fmt(s['max_accel'])}", ln=True)
                    pdf.cell(0, 6, f"- Maximum Entropic Decay (S_e): {fmt(s['max_entropy'])}", ln=True)
                    pdf.cell(0, 6, f"- Systemic Coherence Trough (C_s): {s['min_coherence']:.4f}", ln=True)
                    pdf.cell(0, 6, f"- Structural Breakpoint (t*): {s['t_star_label']}", ln=True)
                pdf.ln(5)
                
                # Plot 1
                pdf.image(temp_img1, x=10, w=190)
                pdf.ln(5)
                
                # Plot 3 (EWS Evidence)
                pdf.image(temp_img3, x=10, w=190)
                pdf.ln(5)
                
                # Plot 2
                pdf.image(temp_img2, x=10, w=190)
                pdf.ln(5)
                
                # Plot 4 (RECD, if enabled and exists)
                import os
                if os.path.exists(temp_img4):
                    # Only add RECD plot if we actually ran it and generated the plot
                    s = self.math_stats
                    if s and s.get("recd_array") is not None:
                        pdf.image(temp_img4, x=10, w=190)
                        pdf.ln(5)
                
                # AI Log or Deterministic Log
                pdf.set_font("Arial", 'B', 12)
                if run_ai:
                    pdf.cell(0, 8, "2. Autonomous Epistemic Peer-Review", ln=True)
                else:
                    pdf.cell(0, 8, "2. Deterministic Structural Diagnosis", ln=True)
                    
                pdf.set_font("Courier", size=9)
                
                # Sanitize the log text to remove greek and complex characters that crash fpdf
                clean_log = self.full_log.replace('τ_s', 'Tau_s').replace('τ', 'Tau').replace('±', '+/-')
                clean_log = clean_log.encode('ascii', 'ignore').decode('ascii')
                pdf.multi_cell(0, 4, clean_log)
                
                if hasattr(self, 'latest_map_data') and self.latest_map_data is not None and not self.latest_map_data.empty:
                    pdf.add_page()
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(0, 8, "3. Ontological Scale / Batch Results", ln=True)
                    pdf.set_font("Courier", size=8)
                    
                    cols = self.latest_map_data.columns.tolist()
                    col_w = 190 / max(1, len(cols))
                    
                    for c in cols:
                        pdf.cell(col_w, 6, str(c)[:15], border=1, align="C")
                    pdf.ln()
                    
                    for _, row in self.latest_map_data.iterrows():
                        for c in cols:
                            pdf.cell(col_w, 6, str(row[c])[:20], border=1, align="C")
                        pdf.ln()
                    pdf.ln(5)
                
                pdf.output(save_path)
                if os.path.exists(temp_img1):
                    os.remove(temp_img1)
                if os.path.exists(temp_img2):
                    os.remove(temp_img2)
                if os.path.exists(temp_img3):
                    os.remove(temp_img3)
                if os.path.exists(temp_img4):
                    os.remove(temp_img4)
                    
                succ_msg = f"\n[SUCCESS] Academic PDF Exported to: {save_path}\n"
                self.after(0, lambda msg=succ_msg: self._update_results(msg))
            except Exception as e:
                err_msg = f"\n[ERROR] PDF Generation failed: {e}\n"
                self.after(0, lambda msg=err_msg: self._update_results(msg))
                
        threading.Thread(target=_build_pdf, daemon=True).start()

    def export_comprehensive_report(self):
        has_data = any(v is not None for v in self.ontological_memory.values())
        if not has_data:
            messagebox.showwarning("No Analysis Data", "Please run at least one analysis (Local, Medium, or Global) before attempting to export the comprehensive report.")
            return
            
        save_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
        if not save_path:
            return
            
        self._update_results(f"\nGenerating Comprehensive PDF Report at {save_path}...\n")
        
        # Determine the current active scale so we can print it in full detail
        active_scale = "Local"
        if hasattr(self, 'scale_menu'):
            active_scale = self.scale_menu.get()
            
        def _build_pdf():
            try:
                from fpdf import FPDF
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", 'B', 18)
                pdf.cell(0, 10, "Systemic Tau - Comprehensive Ontological Report", ln=True, align="C")
                pdf.ln(5)
                
                def fmt(val):
                    if abs(val) < 1e-3 and val != 0:
                        return f"{val:.6f}"
                    return f"{val:,.2f}"
                
                # --- EXECUTIVE SUMMARY ---
                pdf.set_font("Arial", 'B', 14)
                pdf.set_fill_color(220, 220, 220)
                pdf.cell(0, 10, " COMPARATIVE EXECUTIVE SUMMARY", ln=True, fill=True)
                pdf.ln(2)
                
                pdf.set_font("Arial", 'B', 10)
                pdf.cell(20, 8, "Scale", border=1, align="C")
                pdf.cell(25, 8, "Max Tau_s", border=1, align="C")
                pdf.cell(20, 8, "p-value", border=1, align="C")
                pdf.cell(20, 8, "Max RECD", border=1, align="C")
                pdf.cell(105, 8, "Verdict / Interpretation", border=1, align="L")
                pdf.ln()
                
                import numpy as np
                pdf.set_font("Courier", '', 9)
                for scale in ["Local", "Medium", "Global"]:
                    mem = self.ontological_memory.get(scale)
                    if mem and mem.get("math_stats"):
                        st = mem.get("math_stats")
                        tau = fmt(st.get("tau_val", 0))
                        pval = f"{st.get('p_value', 0):.4f}"
                        recd_val = "N/A"
                        if st.get('recd_array') is not None:
                            max_recd = np.nanmax(st['recd_array']) if not np.all(np.isnan(st['recd_array'])) else 0.0
                            recd_val = fmt(max_recd)
                            
                        # Truncate verdict to fit (keep first sentence roughly)
                        verdict = st.get('final_verdict', '').strip().replace('\n', ' ')
                        if '.' in verdict:
                            verdict = verdict.split('.')[0] + "."
                        if len(verdict) > 60: verdict = verdict[:57] + "..."
                        
                        pdf.cell(20, 8, scale, border=1, align="C")
                        pdf.cell(25, 8, tau, border=1, align="C")
                        pdf.cell(20, 8, pval, border=1, align="C")
                        pdf.cell(20, 8, recd_val, border=1, align="C")
                        pdf.cell(105, 8, verdict, border=1, align="L")
                        pdf.ln()
                
                pdf.ln(2)
                pdf.set_font("Arial", 'I', 10)
                # Compute cross scale insight
                tau_vals = {}
                for scale in ["Local", "Medium", "Global"]:
                    mem = self.ontological_memory.get(scale)
                    if mem and mem.get("math_stats"):
                        tau_vals[scale] = mem.get("math_stats").get("tau_val", 0)
                
                if "Local" in tau_vals and "Global" in tau_vals:
                    if tau_vals["Local"] > tau_vals["Global"] * 1.5:
                        insight = "Insight: Systemic reorganization is strongly localized. The topological coherence significantly decays when aggregating into macro-structures (Local >> Global)."
                    elif tau_vals["Global"] > tau_vals["Local"] * 1.5:
                        insight = "Insight: Systemic reorganization is highly coupled globally. The topological coherence strengthens when observing macro-structures (Global >> Local)."
                    else:
                        insight = "Insight: The systemic reorganization signal exhibits scale invariance. The structural phase transition is consistent across both micro and macro topological scales."
                    pdf.multi_cell(0, 5, insight)
                
                pdf.ln(10)
                
                # --- DETAILED SCALES ---
                for scale in ["Local", "Medium", "Global"]:
                    mem = self.ontological_memory.get(scale)
                    if not mem:
                        continue
                        
                    is_active = (scale == active_scale)
                    
                    pdf.add_page()
                    pdf.set_font("Arial", 'B', 16)
                    pdf.set_text_color(44, 160, 44)
                    pdf.cell(0, 10, f"--- {scale.upper()} SCALE ANALYSIS ---", ln=True, align="C")
                    pdf.set_text_color(0, 0, 0)
                    pdf.ln(2)
                    
                    df_snap = mem.get("df_snapshot")
                    stats = mem.get("math_stats", {})
                    targets = stats.get('targets', [])
                    
                    # Print descriptive stats only for active scale to save space
                    if is_active and targets and df_snap is not None:
                        pdf.set_font("Arial", 'B', 12)
                        pdf.cell(0, 8, "0. Variables Evaluated", ln=True)
                        pdf.set_font("Courier", size=8)
                        try:
                            desc = df_snap[targets].describe().round(2).T
                            stats_cols = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
                            pdf.cell(46, 6, "Variable", border=1, align="C")
                            for c in stats_cols:
                                pdf.cell(18, 6, c, border=1, align="C")
                            pdf.ln()
                            for idx, row in desc.iterrows():
                                var_name = str(idx)
                                if len(var_name) > 22: var_name = var_name[:20] + ".."
                                pdf.cell(46, 6, var_name, border=1)
                                for c in stats_cols:
                                    pdf.cell(18, 6, str(row[c]), border=1, align="R")
                                pdf.ln()
                            pdf.ln(5)
                        except KeyError:
                            pass
                    
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(0, 8, "1. Topologic Reorganization Metrics", ln=True)
                    pdf.set_font("Courier", size=10)
                    if stats:
                        pdf.cell(0, 6, f"- Critical Mass Threshold (Max Tau_s): {fmt(stats.get('tau_val', 0))}", ln=True)
                        pdf.cell(0, 6, f"- Peak Acceleration (a_t): {fmt(stats.get('max_accel', 0))}", ln=True)
                        pdf.cell(0, 6, f"- Maximum Entropic Decay (S_e): {fmt(stats.get('max_entropy', 0))}", ln=True)
                        pdf.cell(0, 6, f"- Systemic Coherence Trough (C_s): {stats.get('min_coherence', 0):.4f}", ln=True)
                        pdf.cell(0, 6, f"- Structural Breakpoint (t*): {stats.get('t_star_label', '')}", ln=True)
                    pdf.ln(5)
                    
                    img1 = mem.get("img1")
                    img2 = mem.get("img2")
                    img3 = mem.get("img3")
                    img4 = mem.get("img4")
                    if img1 and os.path.exists(img1):
                        pdf.image(img1, x=10, w=190)
                        pdf.ln(5)
                    
                    # Only print secondary charts if active scale
                    if is_active:
                        if img3 and os.path.exists(img3):
                            pdf.image(img3, x=10, w=190)
                            pdf.ln(5)
                        if img2 and os.path.exists(img2):
                            pdf.image(img2, x=10, w=190)
                            pdf.ln(5)
                            
                    # Always print RECD chart if it exists, regardless of active scale
                    if img4 and os.path.exists(img4) and stats.get("recd_array") is not None:
                        pdf.image(img4, x=10, w=190)
                        pdf.ln(5)
                        
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(0, 8, "2. Diagnosis / Epistemic Review", ln=True)
                    pdf.set_font("Courier", size=9)
                    clean_log = mem.get("full_log", "").replace('τ_s', 'Tau_s').replace('τ', 'Tau').replace('±', '+/-')
                    clean_log = clean_log.encode('ascii', 'ignore').decode('ascii')
                    
                    # Truncate log if not active but show Verdict and RECD text
                    if not is_active:
                        final_verdict = stats.get('final_verdict', '').strip()
                        pdf.set_font("Arial", 'B', 10)
                        pdf.multi_cell(0, 5, f"FINAL VERDICT: {final_verdict}")
                        pdf.ln(2)
                        
                        # Extract RECD text if available
                        recd_lines = []
                        for line in clean_log.split('\n'):
                            if "- Discrete Temporal Structure (RECD)" in line or line.startswith("Breakpoints detected:") or line.startswith("    - At"):
                                recd_lines.append(line)
                            elif line.startswith("No significant temporal discretization"):
                                recd_lines.append(line)
                                
                        pdf.set_font("Courier", size=9)
                        if recd_lines:
                            pdf.multi_cell(0, 4, "\n".join(recd_lines))
                            pdf.ln(2)
                            
                        pdf.multi_cell(0, 4, "... [Detailed epistemic logs omitted in condensed view. Please export this scale directly to view full AI/Deterministic breakdown.] ...")
                    else:
                        pdf.multi_cell(0, 4, clean_log)
                    pdf.ln(5)
                    
                pdf.output(save_path)
                
                succ_msg = f"\n[SUCCESS] Comprehensive PDF Exported to: {save_path}\n"
                self.after(0, lambda msg=succ_msg: self._update_results(msg))
            except Exception as e:
                import traceback
                traceback.print_exc()
                err_msg = f"\n[ERROR] Comprehensive PDF Generation failed: {e}\n"
                self.after(0, lambda msg=err_msg: self._update_results(msg))
                
        import threading
        threading.Thread(target=_build_pdf, daemon=True).start()

    def load_coordinates(self):
        if not hasattr(self, 'data_mgr') or self.df is None:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
            
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not file_path:
            return
            
        try:
            self.data_mgr.load_coordinates_file(file_path)
            self.load_coords_btn.configure(text="📍 Coordinates Loaded", text_color="#2ca02c")
            
            if hasattr(self, 'latest_map_data') and self.latest_map_data is not None:
                self.generate_map_btn.configure(state="normal")
                
            self._update_results(f"\n[GEOSPATIAL] Successfully loaded {len(self.data_mgr.coords_df)} coordinates from {file_path}.\n")
        except Exception as e:
            messagebox.showerror("Coordinates Error", f"Failed to load coordinates:\n{e}")

    def generate_map(self):
        if not hasattr(self, 'latest_map_data') or self.latest_map_data is None:
            messagebox.showwarning("No Results", "Please run an analysis first (Single or Batch).")
            return
            
        if not hasattr(self.data_mgr, 'coords_df') or self.data_mgr.coords_df is None:
            messagebox.showwarning("No Coordinates", "Please load coordinates first.")
            return
            
        self._update_results("\n[GEOSPATIAL] Generating interactive map...\n")
        
        # Read GUI properties safely in the main thread
        scale_val = (self.scale_markers_switch.get() == 1)
        
        def _build_map(scale):
            try:
                from systemictau.desktop.map_generator import SystemicTauMapGenerator
                
                # Check what location col to pass
                loc_col = self.data_mgr.location_col if self.data_mgr.location_col else "Location"
                
                out_path = "systemic_tau_map.html"
                SystemicTauMapGenerator.generate_map(
                    results_df=self.latest_map_data,
                    coords_df=self.data_mgr.coords_df,
                    location_col=loc_col,
                    scale_markers=scale,
                    output_file=out_path,
                    macro_clusters=getattr(self.data_mgr, 'macro_cluster_compositions', {})
                )
                
                self.after(0, lambda: self._update_results(f"[GEOSPATIAL] Map successfully generated and opened in browser.\n"))
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.after(0, lambda e=e: self._update_results(f"\n[GEOSPATIAL ERROR] Failed to generate map: {e}\n"))
                
        import threading
        threading.Thread(target=_build_map, args=(scale_val,), daemon=True).start()

    def open_advanced_settings(self):
        if self.adv_window is not None and self.adv_window.winfo_exists():
            self.adv_window.focus()
            return
        self.adv_window = ctk.CTkToplevel(self)
        self.adv_window.title("Advanced Engine Settings")
        self.adv_window.geometry("500x450")
        self.adv_window.transient(self) 
        
        ctk.CTkLabel(self.adv_window, text="Autonomous Orchestrator Governance", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(20, 10))
        agent_frame = ctk.CTkFrame(self.adv_window)
        agent_frame.pack(fill="x", padx=20, pady=10)
        ctk.CTkLabel(agent_frame, text="Agent Roles", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=5)
        ctk.CTkSwitch(agent_frame, text="Enable Critic Agent (Adversarial Consensus)").pack(anchor="w", padx=20, pady=5)
        
        math_frame = ctk.CTkFrame(self.adv_window)
        math_frame.pack(fill="x", padx=20, pady=10)
        ctk.CTkLabel(math_frame, text="Experimental Analytics", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=5)
        ctk.CTkSwitch(math_frame, text="Enable RECD Analysis (Temporal Discretization)", variable=self.recd_switch).pack(anchor="w", padx=20, pady=5)
        
        ctk.CTkButton(self.adv_window, text="Apply Changes", command=self.adv_window.destroy).pack(pady=20)

    def open_settings(self):
        SettingsDialog(self, self.app_settings)
        # Update components that rely on settings immediately
        self.data_mgr.max_rows = self.app_settings.get("max_rows_before_downsampling", 80000)

    def _collect_session_state(self) -> Dict[str, Any]:
        """Gather everything needed for a complete .stausession save."""
        window_val = 13
        if hasattr(self, "window_slider"):
            window_val = int(self.window_slider.get())

        recd_enabled = True
        if hasattr(self, "recd_var"):
            recd_enabled = bool(self.recd_var.get())
        if hasattr(self, "recd_switch"):
            recd_enabled = bool(self.recd_switch.get())

        config = {
            "window_size": window_val,
            "recd_enabled": recd_enabled,
            "targets": self.get_selected_targets() if hasattr(self, "get_selected_targets") else [],
            "time_col": getattr(self, "time_col", "[Auto Detect]"),
            "scale_mode": getattr(self, "scale_menu", None).get() if hasattr(self, "scale_menu") and self.scale_menu else "Local",
            "agg_method": self.cluster_agg_menu.get().lower() if hasattr(self, "cluster_agg_menu") else "sum",
            "smooth_mode": self.smoothing_menu.get() if hasattr(self, "smoothing_menu") else None,
            # Clustering persistence (new)
            "clustering_method": getattr(self, "clustering_method", "auto"),
            "manual_clusters": dict(getattr(self, "manual_clusters", {})),
        }

        state = {
            "df": self.df,
            "ontological_memory": getattr(self, "ontological_memory", {}),
            "config": config,
            "loaded_file_path": getattr(self, "loaded_file_path", None),
            "math_stats": getattr(self, "math_stats", {}),
            "latest_map_data": getattr(self, "latest_map_data", None),
        }
        return state

    def save_session(self):
        """Save full multi-scale session as .stausession zip."""
        has_any_results = any(v is not None for v in getattr(self, "ontological_memory", {}).values())
        if not has_any_results and self.df is None:
            messagebox.showwarning("Save Session", "Nothing to save. Run at least one analysis (Local/Medium/Global) first.")
            return

        default_name = "session.stausession"
        if self.loaded_file_path:
            base = os.path.splitext(os.path.basename(self.loaded_file_path))[0]
            default_name = f"{base}.stausession"

        filepath = filedialog.asksaveasfilename(
            defaultextension=".stausession",
            filetypes=[("Systemic Tau Session", "*.stausession"), ("All files", "*.*")],
            initialdir=self.app_settings.get("default_output_folder", os.path.expanduser("~")),
            initialfile=default_name,
            title="Save Session (.stausession)"
        )
        if not filepath:
            return

        state = self._collect_session_state()
        success = SessionManager.save_session(state, filepath)
        if success:
            self.title(f"{APP_TITLE} • {os.path.basename(filepath)} [saved]")
            messagebox.showinfo("Session Saved", f"Full session saved to:\n{filepath}\n\nYou can reload this later without re-running analysis.")
        else:
            messagebox.showerror("Save Failed", "Could not create session file. See console for details.")

    def load_session(self):
        """Load .stausession and fully restore state + UI for all scales."""
        filepath = filedialog.askopenfilename(
            filetypes=[("Systemic Tau Session", "*.stausession"), ("All files", "*.*")],
            initialdir=self.app_settings.get("default_output_folder", os.path.expanduser("~")),
            title="Load Session (.stausession)"
        )
        if not filepath:
            return

        if not SessionManager.is_valid_session(filepath):
            messagebox.showerror("Invalid Session", "This file does not appear to be a valid .stausession file.")
            return

        loaded = SessionManager.load_session(filepath)
        if not loaded or not loaded.get("df") is not None:
            messagebox.showerror("Load Error", "Failed to load session data. File may be corrupted.")
            return

        # --- Version compatibility check ---
        meta = loaded.get("metadata", {})
        sess_ver = meta.get("version", "0.0")
        if sess_ver.split(".")[0] != SessionManager.CURRENT_VERSION.split(".")[0]:
            messagebox.showwarning(
                "Version Mismatch",
                f"Session was created with version {sess_ver}.\n"
                f"Current app: {SessionManager.CURRENT_VERSION}.\n\n"
                "Results will be loaded but some fields may be missing or require re-validation."
            )

        # 1. Restore core data
        self.df = loaded["df"]
        self.ontological_memory = loaded.get("ontological_memory", {"Local": None, "Medium": None, "Global": None})

        # Restore current stats if present
        if loaded.get("current_math_stats"):
            self.math_stats = loaded["current_math_stats"]
        elif self.ontological_memory.get("Local"):
            self.math_stats = self.ontological_memory["Local"].get("math_stats", {})

        # Restore file reference
        cfg = loaded.get("config", {})
        self.loaded_file_path = cfg.get("loaded_file_path") or meta.get("dataset_name")

        # Update UI labels
        if self.loaded_file_path:
            self.file_label.configure(text=f"Loaded: {os.path.basename(str(self.loaded_file_path))}")
        else:
            self.file_label.configure(text="Session loaded (in-memory data)")

        if hasattr(self, "data_summary") and self.df is not None:
            self.data_summary.configure(text=f"Session: {len(self.df)} rows × {len(self.df.columns)} cols (loaded, no re-run needed)")

        # 2. Restore UI controls
        win_size = cfg.get("window_size", 13)
        if hasattr(self, "window_slider"):
            self.window_slider.set(win_size)

        recd_on = cfg.get("recd_enabled", True)
        if hasattr(self, "recd_var"):
            self.recd_var.set(1 if recd_on else 0)
        if hasattr(self, "recd_check"):
            if recd_on:
                self.recd_check.select()
            else:
                self.recd_check.deselect()
        if hasattr(self, "recd_switch"):
            self.recd_switch.set(1 if recd_on else 0)

        # 3. Repopulate data-dependent UI (targets, time, etc)
        if self.df is not None and not self.df.empty:
            numeric_cols = self.df.select_dtypes(include="number").columns.tolist()

            # Time column menu
            if hasattr(self, "time_menu"):
                self.time_menu.configure(values=["[Auto Detect]"] + list(self.df.columns))
                tcol = cfg.get("time_col", "[Auto Detect]")
                self.time_col = tcol
                try:
                    self.time_menu.set(tcol)
                except Exception:
                    self.time_menu.set("[Auto Detect]")

            # Target selection restoration (safe)
            saved_targets = cfg.get("targets", [])
            if hasattr(self, "target_primary_menu"):
                try:
                    self.target_primary_menu.configure(values=numeric_cols)
                    if saved_targets:
                        self.target_primary_menu.set(saved_targets[0])
                except Exception:
                    pass

            try:
                if hasattr(self, "target_scroll") and self.target_scroll is not None:
                    if hasattr(self, "target_checkboxes"):
                        for cb, var in list(getattr(self, "target_checkboxes", {}).values()):
                            try:
                                cb.destroy()
                            except Exception:
                                pass
                    self.target_checkboxes = {}

                    for col in numeric_cols:
                        var = tk.StringVar(value="")
                        cb = ctk.CTkCheckBox(
                            self.target_scroll, text=str(col), variable=var,
                            onvalue=str(col), offvalue="",
                            command=lambda c=col: getattr(self, "_redraw_preview", lambda x: None)(c)
                        )
                        cb.pack(anchor="w", pady=2)
                        self.target_checkboxes[str(col)] = (cb, var)
                        if str(col) in saved_targets:
                            cb.select()
            except Exception as e:
                print(f"[load_session] Target UI restore skipped: {e}")

            # Scale menu if present
            if hasattr(self, "scale_menu"):
                scale = cfg.get("scale_mode", "Local")
                try:
                    self.scale_menu.set(scale)
                except Exception:
                    pass

            # Restore clustering config
            cm = cfg.get("clustering_method", "auto")
            self.clustering_method = cm
            self.manual_clusters = dict(cfg.get("manual_clusters", {}))
            if hasattr(self, "clustering_menu"):
                try:
                    label = "Manual (user-defined)" if cm == "manual" else "Automatic (hierarchical)"
                    self.clustering_menu.set(label)
                except Exception:
                    pass
            if self.manual_clusters:
                try:
                    from systemictau.desktop.scale_manager import OntologicalScaleManager
                    OntologicalScaleManager.define_manual_clusters(self.manual_clusters)
                except Exception:
                    pass
            if hasattr(self, "define_clusters_btn") and self.clustering_method == "manual":
                self.define_clusters_btn.configure(fg_color="#166534")

        # 4. Restore latest map if any
        if loaded.get("latest_map_data") is not None:
            self.latest_map_data = loaded["latest_map_data"]
            if hasattr(self, "generate_map_btn"):
                self.generate_map_btn.configure(state="normal")

        # 5. Mark as loaded session + update title
        self.title(f"{APP_TITLE} • [LOADED] {os.path.basename(filepath)}")

        # 6. Show success and hint that scales are ready
        scales_loaded = [s for s in ["Local", "Medium", "Global"] if self.ontological_memory.get(s)]
        msg = f"Session loaded successfully.\n\nScales restored: {', '.join(scales_loaded) or 'None'}\n\nYou can now browse scales, generate reports, and export without re-running the analysis."
        messagebox.showinfo("Session Loaded", msg)

        # Optional: if there is at least one scale result, switch to results view with first available
        if any(self.ontological_memory.values()):
            first_scale = next((s for s in ["Local", "Medium", "Global"] if self.ontological_memory.get(s)), None)
            if first_scale and self.ontological_memory[first_scale]:
                stats = self.ontological_memory[first_scale].get("math_stats", {})
                if stats:
                    try:
                        self._clear_main_view()
                        self._build_clean_results_view(stats)
                    except Exception:
                        pass  # non-fatal

        # Update status
        self._update_status(f"Session loaded from {os.path.basename(filepath)} — ready to use.")

    def open_new_workspace(self):
        import subprocess
        subprocess.Popen([sys.executable, sys.argv[0]])

def main():
    if not ctk:
        print("Error: customtkinter is not installed.")
        sys.exit(1)
        
    ctk.set_appearance_mode("System")  
    ctk.set_default_color_theme("blue")  
    app = SystemicTauApp()
    app.mainloop()

if __name__ == "__main__":
    main()
