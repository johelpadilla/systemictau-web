import numpy as np

class SystemicTauPlotter:
    @staticmethod
    def highlight_graph(app_instance, s):
        """
        Updates the Matplotlib axes with the computed math_stats (s).
        app_instance needs to have: ax1, ax1_twin (optional), ax2, ax2_twin, ax3, ax4, ax_ps, 
        ax_ews1, ax_ews2, ax_ews3, df, secondary_menu, fig1, fig2, fig3, canvas1, canvas2, canvas3.
        """
        app_instance.ax1.clear()
        app_instance.ax2.clear()
        app_instance.ax2_twin.clear()
        app_instance.ax3.clear()
        app_instance.ax4.clear()
        app_instance.ax_ps.clear()
        
        t_star = s["t_star"]
        time_index = np.arange(len(s["data_for_plot"]))
        t_star_val = t_star
        
        # Ax1: Tau Series over time
        app_instance.ax1.plot(time_index, s["tau_series"], color="#1f77b4", linewidth=2, label="Systemic Tau")
        app_instance.ax1.axvline(x=t_star_val, color='r', linestyle='--', alpha=0.7)
        
        pre_break_tau = s["tau_series"][:max(1, t_star)]
        tau_danger = np.mean(pre_break_tau) + 3 * np.std(pre_break_tau)
        app_instance.ax1.axhline(y=tau_danger, color='red', linestyle=':', alpha=0.5, label="Historical Stability Limit")
        
        # Multi-Variable Overlay logic on Ax1
        sec_col = app_instance.secondary_menu.get() if hasattr(app_instance, 'secondary_menu') else None
        if sec_col and sec_col != "[None]" and hasattr(app_instance, 'df') and sec_col in app_instance.df.columns:
            if not hasattr(app_instance, 'ax1_twin'):
                app_instance.ax1_twin = app_instance.ax1.twinx()
            app_instance.ax1_twin.clear()
            sec_data = app_instance.df[sec_col].ffill().fillna(0).values
            app_instance.ax1_twin.plot(time_index, sec_data, color="gray", linestyle=":", linewidth=1.5, alpha=0.7, label=sec_col)
            app_instance.ax1_twin.legend(loc="upper right")
        else:
            if hasattr(app_instance, 'ax1_twin'):
                app_instance.ax1_twin.clear()
        
        app_instance.ax1.set_title(f"Dynamic Systemic Tau: {s['target_col']}")
        app_instance.ax1.legend(loc="upper left")
        
        # Ax2: Raw Data & Acceleration
        app_instance.ax2.plot(time_index, s["data_for_plot"], color="#7f7f7f", linewidth=1.5, alpha=0.5, label="Raw Data")
        app_instance.ax2_twin.plot(time_index, s["acceleration"], color="#ff7f0e", linewidth=1.5, label="Acceleration")
        
        pre_break_acc = s["acceleration"][:max(1, t_star)]
        acc_danger = np.mean(pre_break_acc) + 3 * np.std(pre_break_acc)
        app_instance.ax2_twin.axhline(y=acc_danger, color='red', linestyle=':', alpha=0.5, label="Acceleration Limit")
        
        if np.max(s["tau_series"]) > 0:
            tau_scaled = (s["tau_series"] / np.max(s["tau_series"])) * np.max(s["data_for_plot"])
            app_instance.ax2.plot(time_index, tau_scaled, color="#1f77b4", linestyle="--", linewidth=1.5, alpha=0.8, label="Tau (Overlay)")
            
        app_instance.ax2.axvline(x=t_star_val, color='r', linestyle='--', alpha=0.7)
        app_instance.ax2.set_title("Raw Data & Acceleration (a_t)")
        app_instance.ax2.legend(loc="upper left")
        
        # Ax3: Entropic Decay
        app_instance.ax3.plot(time_index, s["entropy"], color="#2ca02c", linewidth=1.5)
        app_instance.ax3.axvline(x=t_star_val, color='r', linestyle='--', alpha=0.7)
        app_instance.ax3.set_title("Entropic Decay (S_e)")
        
        pre_break_ent = s["entropy"][:max(1, t_star)]
        ent_danger = np.mean(pre_break_ent) + 3 * np.std(pre_break_ent)
        app_instance.ax3.axhline(y=ent_danger, color='red', linestyle=':', alpha=0.5, label="Chaos Limit")
        
        # Ax4: Systemic Coherence
        app_instance.ax4.plot(time_index, s["coherence"], color="#d62728", linewidth=1.5)
        app_instance.ax4.axvline(x=t_star_val, color='r', linestyle='--', alpha=0.7)
        app_instance.ax4.set_title("Systemic Coherence (C_s)")
        
        # Removed tight_layout to prevent macOS Matplotlib Done() crash
        app_instance.canvas1.draw_idle()
        
        # EWS Graph (Fig 3)
        app_instance.ax_ews1.clear()
        app_instance.ax_ews2.clear()
        if hasattr(app_instance, 'ax_ews2_twin'):
            app_instance.ax_ews2_twin.clear()
            app_instance.ax_ews2_twin.set_visible(False)
        app_instance.ax_ews3.clear()
        
        app_instance.ax_ews1.plot(time_index, s["ews_var"], color="#ff7f0e", linewidth=1.5)
        app_instance.ax_ews1.axvline(x=t_star_val, color='r', linestyle='--')
        app_instance.ax_ews1.axvspan(max(0, t_star_val - s["window"]), t_star_val, color='red', alpha=0.1, label="Evaluation Window")
        app_instance.ax_ews1.set_title("Rolling Variance (Critical Slowing Down)")
        app_instance.ax_ews1.legend(loc="upper left")
        
        app_instance.ax_ews2.plot(time_index, s["ews_ar1"], color="#2ca02c", linewidth=1.5, label="AR-1 (Linear)")
        app_instance.ax_ews2.axvline(x=t_star_val, color='r', linestyle='--')
        app_instance.ax_ews2.axvspan(max(0, t_star_val - s["window"]), t_star_val, color='red', alpha=0.1)
        
        if s.get("ews_mi") is not None and hasattr(app_instance, 'ax_ews2_twin'):
            app_instance.ax_ews2_twin.set_visible(True)
            app_instance.ax_ews2_twin.plot(time_index, s["ews_mi"], color="blue", linewidth=1.5, linestyle="--", label="Mutual Info (Non-Linear)")
            from matplotlib.ticker import MaxNLocator
            app_instance.ax_ews2_twin.yaxis.set_major_locator(MaxNLocator(nbins=6))
            app_instance.ax_ews2_twin.tick_params(axis='y', colors='blue', labelsize=11, pad=5)
            app_instance.ax_ews2_twin.set_ylabel("Mutual Information", color="blue", fontweight="bold", labelpad=10)
            # Legends
            lines_1, labels_1 = app_instance.ax_ews2.get_legend_handles_labels()
            lines_2, labels_2 = app_instance.ax_ews2_twin.get_legend_handles_labels()
            app_instance.ax_ews2.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left", framealpha=0.9)
            app_instance.ax_ews2.set_title("System Memory: AR-1 (Primary) vs Mutual Information (Exploratory)", fontweight="bold")
        else:
            app_instance.ax_ews2.set_title("AR-1 Autocorrelation (System Memory)")
            app_instance.ax_ews2.legend(loc="upper left")
        
        app_instance.ax_ews3.plot(time_index, s["ews_skew"], color="#9467bd", linewidth=1.5)
        app_instance.ax_ews3.axvline(x=t_star_val, color='r', linestyle='--')
        app_instance.ax_ews3.axvspan(max(0, t_star_val - s["window"]), t_star_val, color='red', alpha=0.1)
        app_instance.ax_ews3.set_title("Rolling Skewness (Regime Asymmetry)")
        
        # Removed tight_layout to prevent macOS Matplotlib Done() crash
        app_instance.canvas3.draw_idle()
        
        # Phase Space (Ax_PS)
        scatter = app_instance.ax_ps.scatter(s["data_for_plot"], s["acceleration"], c=np.arange(len(s["data_for_plot"])), cmap="viridis", alpha=0.7, s=20)
        app_instance.ax_ps.plot(s["data_for_plot"], s["acceleration"], color="gray", alpha=0.3, linewidth=0.5)
        app_instance.ax_ps.scatter(s["data_for_plot"][t_star], s["acceleration"][t_star], color='red', marker='*', s=200, label="t* Collapse")
        app_instance.ax_ps.set_title("Phase Space: Data vs Acceleration")
        app_instance.ax_ps.set_xlabel("System State (Raw Data)")
        app_instance.ax_ps.set_ylabel("System Momentum (Acceleration)")
        app_instance.ax_ps.legend()
        if not hasattr(app_instance, '_cbar_added'):
            app_instance.fig2.colorbar(scatter, ax=app_instance.ax_ps, label="Time Flow")
            app_instance._cbar_added = True
            
        # Removed tight_layout to prevent macOS Matplotlib Done() crash
        app_instance.canvas2.draw_idle()
        
        # Discrete Temporal Structure (RECD)
        if hasattr(app_instance, 'ax_recd') and s.get("recd_array") is not None:
            app_instance.ax_recd.clear()
            recd_arr = s["recd_array"]
            bps = s.get("recd_breakpoints", [])
            
            app_instance.ax_recd.plot(time_index, recd_arr, color="#17becf", linewidth=2.0, label="RECD Index")
            
            has_bp = False
            for bp in bps:
                if bp < len(time_index):
                    app_instance.ax_recd.axvline(x=time_index[bp], color='orange', linestyle='--', linewidth=2.0)
                    has_bp = True
                    
            if has_bp:
                app_instance.ax_recd.plot([], [], color='orange', linestyle='--', linewidth=2.0, label="Temporal Discretization Break")
                
            # Plot t* on the RECD graph for visual comparison
            app_instance.ax_recd.axvline(x=t_star_val, color='red', linestyle='-.', linewidth=2.0, label="t* Collapse")
            # If time_index is numeric and window size is known, we can add a shaded region
            try:
                # If time_index is string (categorical), this might fail, so we wrap in try-except
                x_start = time_index[max(0, t_star - s["window"])]
                app_instance.ax_recd.axvspan(x_start, t_star_val, color='red', alpha=0.1)
            except:
                pass
                
            app_instance.ax_recd.set_title("Discrete Temporal Structure (RECD Index)", fontweight="bold")
            app_instance.ax_recd.set_ylabel("Discretization (0-1)")
            app_instance.ax_recd.set_ylim(-0.05, 1.05)
            app_instance.ax_recd.legend(loc="upper left")
            app_instance.canvas4.draw_idle()
        elif hasattr(app_instance, 'ax_recd'):
            app_instance.ax_recd.clear()
            app_instance.ax_recd.text(0.5, 0.5, "RECD Analysis Disabled.\nEnable in Advanced Engine Settings.", ha='center', va='center')
            app_instance.canvas4.draw_idle()
            
        return tau_danger, acc_danger, ent_danger

    @staticmethod
    def draw_heatmap(ax, s):
        """
        Draws a Kendall Tau Correlation heatmap on the provided matplotlib axis.
        """
        if not s.get("is_multi") or s.get("corr_matrix") is None:
            ax.clear()
            ax.text(0.5, 0.5, "Heatmap not available for univariate data\nor if structural break is too early.", 
                    ha='center', va='center', wrap=True)
            return

        import seaborn as sns
        ax.clear()
        corr = s["corr_matrix"]
        targets = s["targets"]
        
        # Replace diagonal NaN with 1.0 for visualization
        corr_vis = np.nan_to_num(corr, nan=1.0)
        
        sns.heatmap(corr_vis, annot=True, cmap="coolwarm", center=0, 
                    xticklabels=targets, yticklabels=targets, ax=ax, 
                    vmin=-1, vmax=1, fmt=".2f", cbar=True)
        
        ax.set_title("Multivariate Synchrony (Kendall Tau) Pre-Collapse")
        
        # Highlight leading driver
        mean_corrs = np.nanmean(corr, axis=1)
        leading_idx = np.nanargmax(mean_corrs)
        ax.get_yticklabels()[leading_idx].set_weight("bold")
        ax.get_yticklabels()[leading_idx].set_color("red")
        ax.get_xticklabels()[leading_idx].set_weight("bold")
        ax.get_xticklabels()[leading_idx].set_color("red")
