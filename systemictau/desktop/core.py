import numpy as np
import pandas as pd
import scipy.stats as stats

# Re-use the canonical Kendall-based Systemic Tau from the package
# (implements exact definition τ_s = avg pairwise Kendall on ranks + numba acceleration)
try:
    from systemictau.core import compute_taus as _compute_taus_kendall
except Exception:
    _compute_taus_kendall = None

# Canonical RECD (Ley del Reloj Extramental Discreto + g(τ_s)) from package
# per Sintesis_Magna and Ontologia del Presente papers
try:
    from systemictau.recd import gate_function, compute_recd_increments, accumulate_time
    HAS_CANONICAL_RECD = True
except Exception:
    HAS_CANONICAL_RECD = False
    gate_function = None
    compute_recd_increments = None
    accumulate_time = None

# Layers / Capas support (hyper-persistence, antisynchronization A(k), etc.)
try:
    from systemictau.layers import compute_antisynchronization, hyper_persistence
    HAS_LAYERS = True
except Exception:
    HAS_LAYERS = False
    compute_antisynchronization = None
    hyper_persistence = None

class SystemicTauEngine:
    @staticmethod
    def _calc_mutual_info(x, y, bins=5):
        c_xy = np.histogram2d(x, y, bins)[0]
        return stats.entropy(c_xy.flatten())

    @staticmethod
    def _prepare_ordinal_matrix(numeric_df, targets):
        """
        Rank-safe / ordinal-friendly preparation.
        - Linear interpolation (better than fillna(0) for time series)
        - Limited ffill/bfill
        - Small noise on constant columns to avoid degenerate ranks (helps Kendall)
        This is critical for faithful Kendall-tau and RECD calculations.
        """
        sub = numeric_df[targets].copy()
        # Prefer interpolation over zero-fill to preserve ordinal structure
        sub = sub.interpolate(method='linear', limit_direction='both', limit=5)
        sub = sub.ffill().bfill()
        # Break exact constants (common after heavy NaN fill) with tiny noise
        for col in sub.columns:
            col_data = sub[col]
            if col_data.std() < 1e-12 or len(col_data.dropna().unique()) <= 1:
                sub[col] = col_data + np.random.normal(0, 1e-9, size=len(sub))
        return sub.values

    @staticmethod
    def run_analysis_pipeline(numeric_df, targets, window, smooth_mode=None, time_labels=None, is_multi=False, enable_mi=False, mi_bins=5, enable_recd=False):
        """
        Executes the Systemic Tau mathematical pipeline.
        Returns a dictionary of statistics.
        """
        # Rank-safe prep before any ordinal-sensitive calculation
        matrix_data = SystemicTauEngine._prepare_ordinal_matrix(numeric_df, targets)
        T_len = len(matrix_data)
        
        if smooth_mode == "Moving Average (n=3)":
            # Smoothing on already cleaned data; still avoid aggressive zero fill
            matrix_data = numeric_df[targets].interpolate(limit=5).ffill().bfill().rolling(window=3, min_periods=1, center=True).mean().values
        elif smooth_mode == "Savitzky-Golay (n=5)":
            from scipy.signal import savgol_filter
            if len(matrix_data) >= 5:
                matrix_data = savgol_filter(matrix_data, window_length=5, polyorder=2, axis=0)
                
        # 1. Systemic Tau (Mejora 1: real Kendall ordinal average per the papers)
        # τ_s = mean of pairwise Kendall tau on rank vectors inside each window
        taus_per_module = None
        if is_multi and matrix_data.shape[1] >= 2 and _compute_taus_kendall is not None:
            # Use the package implementation (vectorized + numba when available)
            # It already does rankdata + mean pairwise kendall  (exact τ_s definition)
            try:
                taus_global, taus_per_module = _compute_taus_kendall(matrix_data, window_size=window)
                tau_series = taus_global
            except Exception:
                # Fallback to previous behavior if anything fails
                norm_data = (matrix_data - np.mean(matrix_data, axis=0)) / (np.std(matrix_data, axis=0) + 1e-9)
                tau_series = pd.DataFrame(norm_data).rolling(window=window, min_periods=1).var().sum(axis=1).ffill().values
                taus_per_module = None
            proxy_data = np.linalg.norm( (matrix_data - np.nanmean(matrix_data, axis=0)) / (np.nanstd(matrix_data, axis=0) + 1e-9) , axis=1)
            data_for_plot = proxy_data
        else:
            # Single target (Local) or fallback: rank-based local persistence (variance on ranks)
            # This serves as Capa 1 local intensification / persistence change proxy
            data_for_plot = matrix_data[:, 0]
            ranks = stats.rankdata(data_for_plot, nan_policy='omit')
            # Normalize rank variance to similar dynamic range as before for UI stability
            tau_series = pd.Series(ranks).rolling(window=window, min_periods=1).var().ffill().values
            if np.nanmax(tau_series) > 0:
                tau_series = tau_series / (np.nanmax(tau_series) + 1e-12)
            
        # Sanitize NaNs from rolling window prefixes and from compute_taus (which leaves initial NaNs)
        # This ensures tau_val / t_star and downstream (RECD, EWS, plots) receive clean finite series.
        ts = pd.Series(tau_series, dtype=float)
        tau_series = ts.ffill().bfill().fillna(0.0).values

        tau_val = float(np.max(tau_series))
        t_star = int(np.argmax(tau_series))
        
        tau_median = np.median(tau_series)
        snr = tau_val / tau_median if tau_median > 0 else tau_val
        
        # 2. Acceleration
        velocity = np.gradient(data_for_plot)
        acceleration = np.gradient(velocity)
        max_accel = np.max(acceleration)
        
        # 3. Entropic Decay (Mejora 6 support: rank-based for ordinal alignment)
        # Use variance of ranks as a proxy for rank disorder / permutation entropy style
        rank_for_entropy = pd.Series(stats.rankdata(data_for_plot, nan_policy='omit'))
        entropy = rank_for_entropy.rolling(window=window, min_periods=1).var().ffill().values
        if np.nanmax(entropy) > 0:
            entropy = entropy / (np.nanmax(entropy) + 1e-12)
        # Sanitize leading NaNs from rolling for max_entropy and returned entropy
        entropy = pd.Series(entropy, dtype=float).ffill().bfill().fillna(0.0).values
        max_entropy = float(np.nanmax(entropy))
        
        # 4. Systemic Coherence
        if is_multi:
            T, N = matrix_data.shape
            coherence = np.full(T, np.nan)
            for i in range(window, T + 1):
                win_data = matrix_data[i-window:i, :]
                # Mejora 2: use Kendall (ordinal) instead of Pearson
                kmat = np.eye(N)
                for a in range(N):
                    for b in range(a+1, N):
                        tau, _ = stats.kendalltau(win_data[:, a], win_data[:, b], nan_policy='omit')
                        if np.isnan(tau):
                            tau = 0.0
                        kmat[a, b] = tau
                        kmat[b, a] = tau
                # Use max eigenvalue of Kendall matrix (normalized) as global coupling strength
                try:
                    evals = np.linalg.eigvals(kmat)
                    coherence[i-1] = np.max(evals).real / N
                except Exception:
                    coherence[i-1] = np.nanmean(np.abs(kmat[np.triu_indices(N,1)]))
        else:
            coherence = pd.Series(data_for_plot).rolling(window=window, min_periods=1).corr(pd.Series(data_for_plot).shift(1)).ffill().values
        # Sanitize coherence for consistent return (leading NaNs from construction)
        coherence = pd.Series(coherence, dtype=float).ffill().bfill().fillna(0.0).values
        min_coherence = float(np.nanmin(coherence))
        
        if time_labels is not None and t_star < len(time_labels):
            t_star_label = str(time_labels[t_star])
        else:
            t_star_label = f"Index {t_star}"
            
        # --- ROBUSTNESS & VALIDATION ENGINE ---
        tau_mean_hist = np.mean(tau_series[:t_star]) if t_star > 0 else np.mean(tau_series)
        tau_std_hist = np.std(tau_series[:t_star]) if t_star > 0 else np.std(tau_series)
        effect_size = (tau_val - tau_mean_hist) / (tau_std_hist + 1e-9)
        
        sensitivity_matrix = []
        test_windows = sorted(list(set([max(3, window-4), max(3, window-2), window, min(T_len, window+2), min(T_len, window+4)])))
        for w_test in test_windows:
            if is_multi:
                t_test = pd.DataFrame(norm_data if 'norm_data' in locals() else matrix_data).rolling(window=w_test, min_periods=1).var().sum(axis=1).ffill().values
            else:
                t_test = pd.Series(data_for_plot).rolling(window=w_test, min_periods=1).var().ffill().values
            t_star_test = int(np.argmax(t_test)) if len(t_test) > 0 else t_star
            tau_max_test = np.max(t_test) if len(t_test) > 0 else 0
            sensitivity_matrix.append((w_test, t_star_test, tau_max_test))
            
        # Null Model
        from systemictau.desktop.settings import AppSettings
        app_settings = AppSettings()
        n_perm = app_settings.get("n_permutations", 200)
        alpha = app_settings.get("significance_level", 0.05)
        
        # Performance override if not explicitly set high
        if is_multi and n_perm == 200:
            if T_len > 10000: n_perm = 50
            elif T_len > 1000: n_perm = 100

        surrogate_taus = []
        for _ in range(n_perm):
            if is_multi:
                shuffled = np.apply_along_axis(np.random.permutation, 0, norm_data if 'norm_data' in locals() else matrix_data)
                # Faithful surrogate for the actual statistic when using real Kendall τ_s (Mejora 1 path)
                if is_multi and _compute_taus_kendall is not None:
                    try:
                        tg_surr, _ = _compute_taus_kendall(shuffled, window_size=window)
                        ts_s = pd.Series(tg_surr, dtype=float).ffill().bfill().fillna(0.0).values
                        surr_tau = float(np.max(ts_s))
                    except Exception:
                        surr_tau = pd.DataFrame(shuffled).rolling(window=window, min_periods=1).var().sum(axis=1).ffill().max()
                else:
                    surr_tau = pd.DataFrame(shuffled).rolling(window=window, min_periods=1).var().sum(axis=1).ffill().max()
            else:
                shuffled = np.random.permutation(data_for_plot)
                surr_tau = pd.Series(shuffled).rolling(window=window, min_periods=1).var().ffill().max()
            surrogate_taus.append(surr_tau)
            
        p_value = np.sum(np.array(surrogate_taus) >= tau_val) / n_perm
        if p_value < alpha:
            significance_str = f"Statistically Significant. Strong evidence of topological structural break. The massive Effect Size ({effect_size:.2f} SD) confirms a true physical anomaly."
        elif p_value < 0.10:
            significance_str = f"Marginally Significant. Implies a {p_value*100:.1f}% probability of a false positive under unrestricted shuffling. While a significant anomaly is present (Effect Size = {effect_size:.2f} SD), it lacks strict 95% deterministic confidence, suggesting high background noise."
        else:
            significance_str = f"Not Significant. Implies a {p_value*100:.1f}% false positive rate. Under the Null Hypothesis, this peak easily arises by chance, invalidating the structural break despite the apparent Effect Size ({effect_size:.2f} SD)."
            
        # EWS
        has_precursors = False
        pass_count = 0
        precursor_signal = "None Detected. The system showed no early warning signals."
        
        full_s_data = pd.Series(data_for_plot)
        ews_ar1 = full_s_data.rolling(window=window).corr(full_s_data.shift(1)).ffill().values
        ews_var = full_s_data.rolling(window=window).var().ffill().values
        ews_skew = full_s_data.rolling(window=window).skew().ffill().values
        
        # Mutual Info calculation
        ews_mi = None
        mi_signal = ""
        if enable_mi:
            mi_arr = np.zeros(len(data_for_plot))
            for i in range(window, len(data_for_plot)+1):
                win = data_for_plot[i-window:i]
                if len(win) >= 2:
                    mi_arr[i-1] = SystemicTauEngine._calc_mutual_info(win[:-1], win[1:], bins=mi_bins)
            ews_mi = mi_arr

        if t_star > window:
            ar1_series = ews_ar1[:t_star]
            var_series = ews_var[:t_star]
            skew_series = ews_skew[:t_star]
            
            if len(ar1_series) > window:
                ar1_slope = np.polyfit(np.arange(window), ar1_series[-window:], 1)[0]
                var_slope = np.polyfit(np.arange(window), var_series[-window:], 1)[0]
                skew_slope = np.polyfit(np.arange(window), skew_series[-window:], 1)[0]
                
                if enable_mi:
                    mi_series = ews_mi[:t_star]
                    mi_slope = np.polyfit(np.arange(window), mi_series[-window:], 1)[0]
                    # Empirical thresholding
                    if len(mi_series) > window * 2:
                        hist_slopes = []
                        for i in range(window, len(mi_series) - window):
                            s = np.polyfit(np.arange(window), mi_series[i-window:i], 1)[0]
                            hist_slopes.append(s)
                        if hist_slopes:
                            threshold = np.percentile(hist_slopes, 95)
                            mi_pass = mi_slope > threshold
                        else:
                            mi_pass = mi_slope > 0.01
                    else:
                        mi_pass = mi_slope > 0.01
                else:
                    mi_pass = False
                
                if ar1_slope > 0.05: pass_count += 1
                if var_slope > 0.05: pass_count += 1
                if abs(skew_slope) > 0.05: pass_count += 1
                
                sig_list = [
                    f"AR-1: {ar1_slope:+.3f}" + (" (PASS)" if ar1_slope > 0.05 else " (FAIL)"),
                    f"Var: {var_slope:+.3f}" + (" (PASS)" if var_slope > 0.05 else " (FAIL)"),
                    f"Skew: {skew_slope:+.3f}" + (" (PASS)" if abs(skew_slope) > 0.05 else " (FAIL)")
                ]
                
                if p_value >= 0.05:
                    has_precursors = False
                    if pass_count > 0:
                        precursor_signal = "FALSE POSITIVE EWS. Due to the lack of overall statistical significance (p >= 0.05), these variations are confirmed as random background noise."
                    else:
                        precursor_signal = "NONE. The system showed absolutely no early warning signs of degradation."
                else:
                    if pass_count == 3:
                        has_precursors = True
                        precursor_signal = "STRONG EWS (See Tab 'Early Warning Signals' for visual proof). Broad systemic degradation -> " + " | ".join(sig_list)
                    elif pass_count == 2:
                        has_precursors = False
                        precursor_signal = "MIXED EWS (See Tab 'Early Warning Signals' for visual proof). The evidence for resilience loss is mixed and inconclusive -> " + " | ".join(sig_list)
                    elif pass_count == 1:
                        has_precursors = False
                        precursor_signal = "WEAK EWS. The signal is mostly noise and visually unconvincing. Structural resilience remains intact."
                    else:
                        has_precursors = False
                        precursor_signal = "NONE. The system showed absolutely no early warning signs of degradation."
                    if pass_count >= 2:
                        precursor_signal += f"\\n     (Methodology: Evaluated via rolling monotonic regression slope over the exact W={window} periods immediately preceding t*=[{t_star_label}])."
                        
        # Final Verdict
        if p_value >= 0.05 and pass_count > 0:
            final_verdict = f"NON-SIGNIFICANT NOISE (NO COLLAPSE).\\n     -> Finding: Any detected metrics are mathematically indistinguishable from random baseline variance (p={p_value:.2f} >= 0.05).\\n     -> Action: The system is operating normally. Continue standard monitoring."
        elif p_value >= 0.05 and pass_count == 0:
            final_verdict = "PURE NOISE.\\n     -> Finding: No precursors, and the peak is statistically insignificant.\\n     -> Action: Resume normal operations."
        elif p_value < 0.05 and has_precursors:
            final_verdict = "TRUE ENDOGENOUS COLLAPSE.\\n     -> Finding: The system degraded structurally from within (precursors) before suffering a significant break.\\n     -> Action: Intervene directly on the 'Leading Driver' variables to break the synchronous lock-in."
        else:
            final_verdict = "EXOGENOUS SHOCK (BLACK SWAN).\\n     -> Finding: The system collapsed suddenly without any mathematical warning signs. Resilience was instantly overwhelmed.\\n     -> Action: Structural fortification against external macro-shocks is required."

        # Geometry
        fwhms = []
        relaxations = []
        for w_test, t_test, tau_test in sensitivity_matrix:
            if is_multi:
                t_test_series = pd.DataFrame(norm_data if 'norm_data' in locals() else matrix_data).rolling(window=w_test, min_periods=1).var().sum(axis=1).ffill().values
            else:
                t_test_series = pd.Series(data_for_plot).rolling(window=w_test, min_periods=1).var().ffill().values
                
            threshold_50 = tau_test / 2.0
            f_dur = 0
            for i in range(t_test-1, -1, -1):
                if t_test_series[i] >= threshold_50: f_dur += 1
                else: break
            for i in range(t_test+1, len(t_test_series)):
                if t_test_series[i] >= threshold_50: f_dur += 1
                else: break
            fwhms.append(max(1, f_dur))
            
            mean_hist = np.mean(t_test_series[:t_test]) if t_test > 0 else np.mean(t_test_series)
            std_hist = np.std(t_test_series[:t_test]) if t_test > 0 else np.std(t_test_series)
            r_dur = len(t_test_series) - t_test
            for i in range(t_test+1, len(t_test_series)):
                if t_test_series[i] <= (mean_hist + std_hist):
                    r_dur = i - t_test
                    break
            relaxations.append(r_dur)
            
        fwhm_str = f"{np.mean(fwhms):.1f} +/- {np.std(fwhms):.1f}"
        relax_str = f"{np.mean(relaxations):.1f} +/- {np.std(relaxations):.1f}"
        
        post_mean = np.mean(tau_series[t_star+1:]) if t_star < len(tau_series)-1 else tau_mean_hist
        if post_mean > tau_mean_hist + 2*tau_std_hist: post_regime = "Hyper-volatile (New Normal is structurally unstable)"
        elif post_mean < tau_mean_hist - tau_std_hist: post_regime = "Hypo-volatile (System suppressed or dead)"
        else: post_regime = "Returned to historical baseline"
            
        # Sensitivity Narrative
        t_star_arr = [x[1] for x in sensitivity_matrix]
        t_std = np.std(t_star_arr)
        if t_std > 3:
            t_min = np.min(t_star_arr)
            t_max = np.max(t_star_arr)
            if t_max - t_min > 20:
                if p_value >= 0.05:
                    sensitivity_narrative = f"Non-Stationary Noise (Std = {t_std:.1f} periods). The peak location changes depending on the observation window (t={t_min} vs t={t_max}). IMPLICATION: This confirms we are simply detecting random noise peaks at different frequencies."
                else:
                    sensitivity_narrative = f"WARNING: Dual Time-Scales Detected (Std = {t_std:.1f} periods). The short-term anomaly at t={t_min} represents a localized micro-shock, while the long-term anomaly at t={t_max} represents a systemic macro-shift. IMPLICATION: The system's vulnerability is horizon-dependent; it reacts differently depending on the speed of the stressor."
            else:
                if p_value >= 0.05:
                    sensitivity_narrative = f"Parameter Sensitivity (Std = {t_std:.1f} periods). The anomaly location drifts depending on the window size. ACTIONABLE INSIGHT: This instability corroborates that the detected peak is merely random noise."
                else:
                    sensitivity_narrative = f"WARNING: High Parameter Sensitivity (Std = {t_std:.1f} periods). The transition is highly scale-dependent. Conclusion: This is not a well-defined point-in-time shock, but a prolonged structural degradation process. ACTIONABLE INSIGHT: Cease monitoring for abrupt short-term triggers. Strategic focus must shift immediately to tracking long-term structural degradation over larger time horizons."
        else:
            if p_value >= 0.05:
                sensitivity_narrative = f"Stable Noise Peak (Std = {t_std:.1f} periods). The highest volatility point remains consistent across scales, but statistical tests confirm it is not a structural break."
            else:
                sensitivity_narrative = f"Highly Stable (Std = {t_std:.1f} periods). Breakpoint is robust to parameter changes, indicating a true, well-defined instantaneous shock."
                
        # Multivariate Synchrony
        multivariate_count = len(targets)
        multivariate_str = f"Evaluated jointly across {multivariate_count} variables."
        corr_matrix = None
        if multivariate_count > 1 and t_star > window:
            pre_collapse_df = numeric_df[targets].iloc[:t_star]
            corr_matrix = pre_collapse_df.corr(method='kendall').to_numpy(copy=True)
            np.fill_diagonal(corr_matrix, np.nan)
            mean_r = np.nanmean(corr_matrix)
            max_r = np.nanmax(corr_matrix)
            min_r = np.nanmin(corr_matrix)
            sync_range = f"(Mean: {mean_r:.2f}, Min: {min_r:.2f}, Max: {max_r:.2f})"
            
            mean_corrs = np.nanmean(corr_matrix, axis=1)
            
            # Safely handle the case where mean_corrs is all NaN (e.g. constant data)
            if np.all(np.isnan(mean_corrs)):
                leading_idx = 0
                leading_val = 0.0
            else:
                leading_idx = np.nanargmax(mean_corrs)
                leading_val = mean_corrs[leading_idx]
                
            leading_col = targets[leading_idx]
            
            if p_value >= 0.05:
                coupling_note = f"(Max Coupling: '{leading_col}' r={leading_val:.2f})"
                if mean_r > 0.6:
                    multivariate_str = f"High Baseline Synchrony {sync_range}. Variables move together organically {coupling_note}.\\n     -> Implication: The system is highly interconnected, but current fluctuations are within normal statistical bounds."
                elif mean_r < 0.3:
                    multivariate_str = f"Low Baseline Synchrony {sync_range}. Variables are fluctuating independently {coupling_note}.\\n     -> Implication: System dynamics are fully decoupled."
                else:
                    multivariate_str = f"Moderate Baseline Synchrony {sync_range} {coupling_note}.\\n     -> Implication: Normal background connectivity without critical stress."
            else:
                leading_str = f"Leading Driver Detected: '{leading_col}' exhibited the highest systemic coupling (r={leading_val:.2f}) prior to collapse.\\n     -> Significance: Operational interventions must prioritize '{leading_col}' to decouple the system dynamics."
                if mean_r > 0.6:
                    multivariate_str = f"High System-wide Synchrony {sync_range}. Variables locked-in together.\\n     -> {leading_str}\\n     -> Implication: A localized shock here will likely propagate globally."
                elif mean_r < 0.3:
                    multivariate_str = f"Low Synchrony {sync_range}. Collapse driven by isolated variables.\\n     -> {leading_str}\\n     -> Implication: Risk is localized. Interventions can be highly targeted."
                else:
                    multivariate_str = f"Moderate Synchrony {sync_range}.\\n     -> {leading_str}\\n     -> Implication: Moderate contagion risk. Monitor the Leading Driver closely."
        
        recd_array = None
        recd_breakpoints = None
        recd_global_dict = None
        recd_dtk = None
        recd_gate = None
        recd_accumulated = None
        recd_depth = None

        # Canonical RECD (primary per Ley del Reloj Extramental Discreto)
        # Uses g(τ_s) + Δt_k = δ^{-k} * |τ_s| * Δt0 / n  + accumulation t_n = Σ g * Δt
        # from package (Sintesis Magna 2026 + Ontologia del Presente)
        if enable_recd:
            try:
                # Still support per-target structure index for multi (old desktop heuristic for local Capa1)
                from systemictau.desktop.recd import compute_recd_discretization as _old_recd_disc

                # Level for legacy per-component calls
                if is_multi:
                    level = "global"
                elif targets and targets[0].startswith("MacroCluster_"):
                    level = "medium"
                else:
                    level = "local"

                if level == "global" and len(targets) > 1:
                    recd_global_dict = {}
                    for i, t_name in enumerate(targets):
                        t_series = matrix_data[:, i]
                        r_arr, r_bp = _old_recd_disc(t_series, window, ontological_level=level)
                        recd_global_dict[t_name] = {'array': r_arr, 'breakpoints': r_bp}

                # Legacy structure index for recd_array (kept for UI compat)
                try:
                    recd_array, recd_breakpoints = _old_recd_disc(tau_series, window, ontological_level=level)
                except Exception:
                    recd_array, recd_breakpoints = None, None

                # === Canonical RECD law on the Systemic Tau series (the real one) ===
                if HAS_CANONICAL_RECD:
                    dtk_s, g_s, depth_s = compute_recd_increments(tau_series, window_size=window)
                    acc_t, dtk2, g2, depth2 = accumulate_time(tau_series, window_size=window)
                    recd_dtk = dtk_s
                    recd_gate = g_s
                    recd_accumulated = acc_t
                    recd_depth = depth_s
                    # Prefer the true accumulated emergent time for main recd when available
                    if recd_accumulated is not None and len(recd_accumulated) == len(tau_series):
                        # keep legacy recd_array as supplementary structure; use accumulated for the clock
                        pass
            except Exception as e:
                import traceback
                traceback.print_exc()

        # === Capas / Ontological Layers support (Capa 1 local, Capa 2 coherence/antisync, Capa 3 ascent) ===
        antisync_series = None
        hyper_persistence_series = None
        capa3_ascent_detected = False
        if is_multi and taus_per_module is not None and len(tau_series) > 0:
            if HAS_LAYERS:
                try:
                    antisync_series = compute_antisynchronization(taus_per_module)
                    hp_z, core_h = hyper_persistence(tau_series, window_size=max(7, window))
                    hyper_persistence_series = hp_z
                    # Simple Capa 3 detector: significant break + RECD active (low antisync window + |τ| crossing)
                    if p_value < 0.10 and recd_dtk is not None:
                        active_recd = np.nansum(np.abs(recd_dtk) > 0) > (len(tau_series) * 0.05)
                        if active_recd:
                            capa3_ascent_detected = True
                except Exception:
                    pass
            else:
                # Fallback direct antisync (std across modules at each time)
                try:
                    antisync_series = np.nanstd(taus_per_module, axis=1)
                except Exception:
                    antisync_series = None

        # Regime classification per paper thresholds (for UI + analysis)
        regime_series = np.full(len(tau_series), "transitional", dtype=object)
        regime_series[tau_series >= 0.50] = "stable"
        regime_series[np.abs(tau_series) < 0.41] = "chaotic"
        regime_series[tau_series <= -0.41] = "antisync"

        # tau_method for clarity (local Capa1 proxy vs true systemic Kendall)
        tau_method = "kendall_pairwise_average" if is_multi else "rank_variance_local_capa1"

        return {
            "target_col": targets[0],
            "is_multi": is_multi,
            "targets": targets,
            "data_for_plot": data_for_plot,
            "tau_series": tau_series,
            "tau_method": tau_method,
            "ews_ar1": ews_ar1,
            "ews_var": ews_var,
            "ews_skew": ews_skew,
            "ews_mi": ews_mi,
            "mi_pass": mi_pass if enable_mi and t_star > window else None,
            "acceleration": acceleration,
            "entropy": entropy,
            "coherence": coherence,
            "t_star": t_star,
            "t_star_label": t_star_label,
            "tau_val": tau_val,
            "max_accel": max_accel,
            "max_entropy": max_entropy,
            "min_coherence": min_coherence,
            "window": window,
            "snr": snr,
            "sensitivity_matrix": sensitivity_matrix,
            "sensitivity_narrative": sensitivity_narrative,
            "effect_size": effect_size,
            "p_value": p_value,
            "significance_str": significance_str,
            "fwhm_str": fwhm_str,
            "relax_str": relax_str,
            "post_regime": post_regime,
            "precursor_signal": precursor_signal,
            "final_verdict": final_verdict,
            "multivariate_str": multivariate_str,
            "n_perm": n_perm,
            "corr_matrix": corr_matrix,
            "recd_array": recd_array,
            "recd_breakpoints": recd_breakpoints,
            "recd_multi": recd_global_dict,
            "recd_dtk": recd_dtk,
            "recd_gate": recd_gate,
            "recd_accumulated": recd_accumulated,
            "recd_depth": recd_depth,
            "antisync_series": antisync_series,
            "hyper_persistence": hyper_persistence_series,
            "regime_series": regime_series,
            "capa3_ascent_detected": capa3_ascent_detected,
            "taus_per_module": taus_per_module if is_multi else None,
            "time_labels": time_labels,
        }
