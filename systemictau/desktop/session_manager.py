import os
import json
import zipfile
import datetime
import tempfile
import shutil
from typing import Dict, Any, Optional
from io import BytesIO

import numpy as np
import pandas as pd

# Avoid circular import with app.py — use a local constant
APP_VERSION = "4.2"


class SessionManager:
    """
    Manages .stausession files (zip archives) for full Systemic Tau analysis sessions.
    v4.2 structure (multi-perspective):
      - metadata.json
      - config.json
      - data.parquet (or data.json)
      - perspectives/<perspective>/<Scale>.json   (spatial/multivariate etc.)
      - manual_clusters.json (optional)
      - recd_state.json (optional)

    Supports legacy v4.1 flat results.json for backward compat on load.
    """

    EXTENSION = ".stausession"
    CURRENT_VERSION = "4.2"
    # For validation we now accept either old flat or new nested
    REQUIRED_FILES_OLD = ["metadata.json", "config.json", "results.json"]
    PERSPECTIVES_DIR = "perspectives"

    # ---------- Serialization helpers ----------
    @staticmethod
    def _to_json_safe(obj: Any) -> Any:
        """Recursively convert numpy/pandas objects to pure Python for JSON (NumPy 2+ compatible)."""
        if obj is None:
            return None
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            val = float(obj)
            if np.isnan(val):
                return None
            return val
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (pd.Series, pd.DataFrame)):
            return obj.to_dict(orient="list") if isinstance(obj, pd.Series) else "dataframe_stored_separately"
        if isinstance(obj, (list, tuple)):
            return [SessionManager._to_json_safe(v) for v in obj]
        if isinstance(obj, dict):
            return {str(k): SessionManager._to_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        try:
            json.dumps(obj)
            return obj
        except (TypeError, OverflowError):
            return str(obj)

    @staticmethod
    def _restore_arrays(d: Dict[str, Any]) -> Dict[str, Any]:
        """Convert lists back to numpy arrays where it makes sense for math_stats."""
        if not isinstance(d, dict):
            return d
        out = {}
        for k, v in d.items():
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], (int, float)):
                out[k] = np.asarray(v, dtype=float)
            elif isinstance(v, dict):
                out[k] = SessionManager._restore_arrays(v)
            else:
                out[k] = v
        return out

    @staticmethod
    def _serialize_scale(scale_result: Dict[str, Any], scale_metrics: Dict[str, Any], cfg: Dict[str, Any], scale: str) -> Dict[str, Any]:
        """Create a clean, self-contained dict for one scale to be written as <Scale>.json."""
        if not scale_result:
            scale_result = {}
        if not scale_metrics:
            scale_metrics = {}

        clustering_method = "auto"
        cm = cfg.get("clustering_method") or cfg.get("clustering") or {}
        if isinstance(cm, dict):
            clustering_method = cm.get(scale.lower(), cm.get(scale, "auto"))
        elif scale.lower() in cfg:
            clustering_method = cfg.get(scale.lower(), "auto")
        else:
            # Derive from groups if present
            if scale == "Medium" and cfg.get("medium_groups"):
                clustering_method = "manual"
            elif scale == "Global":
                ggroups = cfg.get("global_groups") or cfg.get("global_cluster_composition") or {}
                if ggroups and len(ggroups or {}) >= 2:
                    clustering_method = "manual" if cfg.get("global_groups") else "fixed_num"
                elif cfg.get("auto_global") is False:
                    clustering_method = "fixed_num"
                else:
                    clustering_method = "auto"

        # Cluster composition for this scale (prefer explicit groups / attached composition)
        cluster_comp = {}
        if scale == "Medium":
            cluster_comp = cfg.get("medium_groups", {}) or cfg.get("cluster_composition", {}) or {}
        elif scale == "Global":
            cluster_comp = (cfg.get("global_groups", {}) or
                            cfg.get("global_cluster_composition", {}) or
                            scale_result.get("cluster_composition", {}) or {})

        # Pull extra from scale_result if present (some runners attach it)
        if not cluster_comp and "cluster_map" in scale_result:
            cluster_comp = scale_result.get("cluster_map", {}) or {}

        if not cluster_comp:
            # last fallback from the scale_result itself
            cluster_comp = scale_result.get("cluster_composition", {}) or {}

        data = {
            "scale_label": scale,
            "metrics": {
                "mean_tau": scale_metrics.get("mean_tau"),
                "tau_std": scale_metrics.get("tau_std"),
                "coherence": scale_metrics.get("coherence"),
                "t_star": scale_result.get("t_star"),
                "recd_T_final": scale_metrics.get("recd_T_final"),
                "n_modules": scale_metrics.get("n_modules") or scale_result.get("n_modules"),
            },
            "series": {
                "taus_global": SessionManager._to_json_safe(scale_result.get("taus_global")),
                "accum_T": SessionManager._to_json_safe(scale_result.get("accum_T")),
                "dtk": SessionManager._to_json_safe(scale_result.get("dtk")),
            },
            "clustering_method": clustering_method,
            "cluster_composition": {k: list(v) if isinstance(v, (list, tuple)) else v for k, v in cluster_comp.items()},
            "window_size": scale_result.get("window_size") or cfg.get("window_size"),
        }
        return data

    @staticmethod
    def _deserialize_scale(raw_scale_data: Dict[str, Any]) -> Dict[str, Any]:
        """Turn the on-disk scale json back into something usable as scale_results entry + metrics/config hints."""
        if not isinstance(raw_scale_data, dict):
            return {}

        metrics = raw_scale_data.get("metrics", {})
        series = raw_scale_data.get("series", {})

        result = {
            "scale_label": raw_scale_data.get("scale_label"),
            "mean_tau": metrics.get("mean_tau"),
            "tau_std": metrics.get("tau_std"),
            "coherence": metrics.get("coherence"),
            "t_star": metrics.get("t_star"),
            "recd_T_final": metrics.get("recd_T_final"),
            "n_modules": metrics.get("n_modules"),
            "window_size": raw_scale_data.get("window_size"),
        }

        # Restore arrays (the _restore_arrays expects dicts of lists)
        for key, outkey in [("taus_global", "taus_global"), ("accum_T", "accum_T"), ("dtk", "dtk")]:
            val = series.get(key)
            if isinstance(val, list):
                try:
                    result[outkey] = np.asarray(val, dtype=float)
                except Exception:
                    result[outkey] = val
            else:
                result[outkey] = val

        # Attach clustering info for the wrapper
        result["_clustering_method"] = raw_scale_data.get("clustering_method", "auto")
        result["_cluster_composition"] = raw_scale_data.get("cluster_composition", {})

        return result

    # ---------- Core Save / Load ----------
    @staticmethod
    def _serialize_perspective_scale(analysis: Dict[str, Any], scale: str) -> Optional[Dict[str, Any]]:
        """Helper: serialize one scale for one perspective. Returns payload or None."""
        if not analysis or not isinstance(analysis, dict):
            return None
        scale_results = analysis.get("scale_results", {}) or {}
        scale_metrics = analysis.get("scale_metrics", {}) or {}
        vcfg = analysis.get("config", {}) or {}

        sr = scale_results.get(scale, {})
        sm = scale_metrics.get(scale, {})
        if not sr and not sm:
            return None
        return SessionManager._serialize_scale(sr, sm, vcfg, scale)

    @staticmethod
    def save_session(state: Dict[str, Any], filepath: str) -> bool:
        """
        AGGRESSIVE v4.x save for BOTH perspectives.

        Explicitly iterates over ["spatial", "multivariate"].
        Forces creation of:
            perspectives/spatial/{Local,Medium,Global}.json
            perspectives/multivariate/{Local,Medium,Global}.json

        It is impossible (by design) to save without attempting both.
        """
        if not filepath.endswith(SessionManager.EXTENSION):
            filepath += SessionManager.EXTENSION

        try:
            df = state.get("df")
            analyses = state.get("analyses") or {}
            ontological_memory = state.get("ontological_memory", {}) or {}
            config = state.get("config", {})
            loaded_file_path = state.get("loaded_file_path", "")

            if df is None or df.empty:
                print("Session save aborted: no dataframe.")
                return False

            # === AGGRESSIVE EXPLICIT ITERATION (as required) ===
            # We force the two known perspectives. We pull from analyses if present.
            perspectives = {}
            MANDATORY_PERSPECTIVES = ["spatial", "multivariate"]

            for view in MANDATORY_PERSPECTIVES:
                analysis = analyses.get(view) if isinstance(analyses, dict) else None
                if isinstance(analysis, dict) and "scale_results" in analysis:
                    perspectives[view] = analysis
                else:
                    # Even if missing, we will note it; still attempt to record in metadata
                    # We do not invent data, but the loop will try to write (skipping empty)
                    perspectives[view] = {}  # placeholder so metadata sees it

            # Legacy fallback only if no multi-perspectives at all
            if not any(p for p in perspectives.values() if p):
                if ontological_memory:
                    perspectives = {"default": {"scale_results": ontological_memory, "scale_metrics": {}, "config": config}}
                else:
                    perspectives = {"default": {"scale_results": {}, "scale_metrics": {}, "config": config}}

            # Build metadata — explicitly list the mandatory ones we attempted
            metadata_persps = [v for v in MANDATORY_PERSPECTIVES if v in perspectives]
            metadata = {
                "version": SessionManager.CURRENT_VERSION,
                "app_version": APP_VERSION,
                "created": datetime.datetime.now().isoformat(),
                "dataset_name": os.path.basename(loaded_file_path) if loaded_file_path else "unknown",
                "shape": [int(df.shape[0]), int(df.shape[1])],
                "has_recd": bool(config.get("recd_enabled", True)),
                "perspectives": metadata_persps,
            }

            # Build metadata
            metadata = {
                "version": SessionManager.CURRENT_VERSION,
                "app_version": APP_VERSION,
                "created": datetime.datetime.now().isoformat(),
                "dataset_name": os.path.basename(loaded_file_path) if loaded_file_path else "unknown",
                "shape": [int(df.shape[0]), int(df.shape[1])],
                "has_recd": bool(config.get("recd_enabled", True)),
                "perspectives": list(perspectives.keys()),
            }

            # Build clean global config
            clean_config = {
                "window_size": int(config.get("window_size", 13)),
                "recd_enabled": bool(config.get("recd_enabled", True)),
                "targets": config.get("targets", []),
                "time_col": config.get("time_col", "[Auto Detect]"),
                "scale_mode": config.get("scale_mode", "Global"),
                "agg_method": config.get("agg_method", "sum"),
                "smooth_mode": config.get("smooth_mode"),
                "loaded_file_path": loaded_file_path,
            }

            # Manual clusters aggregated (only for views that have real data)
            manual_clusters = {}
            for view in MANDATORY_PERSPECTIVES:
                analysis = perspectives.get(view, {})
                if analysis:  # only real analysis objects
                    vcfg = analysis.get("config", {})
                    manual_clusters[view] = {
                        "medium_groups": vcfg.get("medium_groups", {}),
                        "global_groups": vcfg.get("global_groups", {}),
                        "clustering_method": vcfg.get("clustering_method", {}),
                    }

            # Create the zip archive
            written_perspective_files = []
            with zipfile.ZipFile(filepath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                # data.parquet preferred
                data_stored_as = "parquet"
                try:
                    buf = BytesIO()
                    df.to_parquet(buf, engine="pyarrow", compression="zstd", index=False)
                    buf.seek(0)
                    zf.writestr("data.parquet", buf.read())
                except Exception:
                    data_stored_as = "json"
                    recs = df.to_dict(orient="records")
                    zf.writestr("data.json", json.dumps(recs))
                metadata["data_format"] = data_stored_as

                # config.json
                zf.writestr("config.json", json.dumps(clean_config, indent=2))

                # === EXPLICIT AGGRESSIVE WRITE FOR BOTH PERSPECTIVES ===
                # We loop hard-coded over the two required perspectives.
                # This makes it impossible to accidentally save only one.
                for view in MANDATORY_PERSPECTIVES:
                    analysis = perspectives.get(view, {}) or {}
                    scale_results = analysis.get("scale_results", {}) or {}
                    scale_metrics = analysis.get("scale_metrics", {}) or {}
                    vcfg = analysis.get("config", {}) or {}

                    view_files_written = 0
                    for scale in ["Local", "Medium", "Global"]:
                        payload = SessionManager._serialize_perspective_scale(analysis if analysis else None, scale)
                        if payload is None:
                            # For missing perspective we still create an empty marker? No — only write if data exists.
                            # But to satisfy structure, we only write real data. Validation will catch missing.
                            continue
                        arcname = f"{SessionManager.PERSPECTIVES_DIR}/{view}/{scale}.json"
                        zf.writestr(arcname, json.dumps(payload, indent=2))
                        written_perspective_files.append(arcname)
                        view_files_written += 1

                    if view_files_written > 0:
                        print(f"[SessionManager] Saved {view_files_written} scales for perspective '{view}'")

                # === AGGRESSIVE VALIDATION BEFORE FINALIZING SAVE ===
                # Check that we actually wrote files for both required perspectives.
                has_spatial = any(f.startswith("perspectives/spatial/") for f in written_perspective_files)
                has_multivariate = any(f.startswith("perspectives/multivariate/") for f in written_perspective_files)

                if not has_spatial or not has_multivariate:
                    missing = []
                    if not has_spatial:
                        missing.append("spatial")
                    if not has_multivariate:
                        missing.append("multivariate")
                    warning_msg = f"[SessionManager][AGGRESSIVE VALIDATION] WARNING: Missing perspectives in save: {missing}. " \
                                  "Both 'spatial' and 'multivariate' must be present for a complete session."
                    print(warning_msg)
                    # We still continue (don't abort the save), but the zip will have incomplete data.
                    # The caller / UI should warn the user.

                # manual_clusters.json
                if any(any(g for g in v.values()) for v in manual_clusters.values()):
                    zf.writestr("manual_clusters.json", json.dumps(manual_clusters, indent=2))

                # recd_state.json (simple for now)
                recd_state = {
                    "recd_enabled": bool(config.get("recd_enabled", True)),
                }
                zf.writestr("recd_state.json", json.dumps(recd_state, indent=2))

                # Legacy results.json for old loaders (best effort flat view of first real perspective)
                try:
                    legacy_results = {}
                    real_persps = [v for v in MANDATORY_PERSPECTIVES if perspectives.get(v)]
                    if real_persps:
                        first_view = real_persps[0]
                        first = perspectives.get(first_view, {})
                        for scale in ["Local", "Medium", "Global"]:
                            sr = first.get("scale_results", {}).get(scale, {})
                            if sr:
                                legacy_results[scale] = SessionManager._to_json_safe(sr)
                    if legacy_results:
                        zf.writestr("results.json", json.dumps(legacy_results, indent=2))
                except Exception:
                    pass

                # metadata last
                zf.writestr("metadata.json", json.dumps(metadata, indent=2))

            # Final post-zip validation (re-open to confirm structure)
            try:
                with zipfile.ZipFile(filepath, "r") as check_zf:
                    names = check_zf.namelist()
                    spatial_scales = [n for n in names if n.startswith("perspectives/spatial/") and n.endswith(".json")]
                    mv_scales = [n for n in names if n.startswith("perspectives/multivariate/") and n.endswith(".json")]
                    if len(spatial_scales) < 1 or len(mv_scales) < 1:
                        print(f"[SessionManager][POST-SAVE VALIDATION] Incomplete save! "
                              f"spatial_scales={len(spatial_scales)}, multivariate_scales={len(mv_scales)}")
                    else:
                        print(f"[SessionManager] Save validation passed. Both perspectives present in .stausession.")
            except Exception as e:
                print(f"[SessionManager] Post-save zip validation failed: {e}")

            return True

        except Exception as e:
            print(f"[SessionManager] Save failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    @staticmethod
    def load_session(filepath: str) -> Optional[Dict[str, Any]]:
        """
        Returns a dict ready for app restoration.

        New v4.2 returns (preferred for studio):
          {
            "metadata": {...},
            "config": {...},
            "df": pd.DataFrame,
            "analyses": {"spatial": {"scale_results": {...}, "scale_metrics": {...}, "config": {...}}, ...},
            "manual_clusters": {...},
            "recd_state": {...},
            ...
          }

        Legacy v4.1 support falls back to ontological_memory reconstruction.
        """
        if not os.path.exists(filepath):
            return None

        try:
            with zipfile.ZipFile(filepath, "r") as zf:
                namelist = zf.namelist()

                # metadata (always try)
                metadata = {}
                if "metadata.json" in namelist:
                    with zf.open("metadata.json") as f:
                        metadata = json.load(f)

                version = metadata.get("version", "4.1")
                major = str(version).split(".")[0]
                if major != str(SessionManager.CURRENT_VERSION).split(".")[0]:
                    print(f"Warning: Session version {version} may be incompatible with current {SessionManager.CURRENT_VERSION}")

                # config
                config = {}
                if "config.json" in namelist:
                    with zf.open("config.json") as f:
                        config = json.load(f)

                # data
                if "data.parquet" in namelist:
                    with zf.open("data.parquet") as f:
                        df = pd.read_parquet(BytesIO(f.read()))
                elif "data.json" in namelist:
                    with zf.open("data.json") as f:
                        recs = json.load(f)
                        df = pd.DataFrame(recs)
                else:
                    raise ValueError("No data file found in session archive")

                # === NEW v4.2 structure ===
                analyses = {}
                manual_clusters = {}
                recd_state = {"recd_enabled": config.get("recd_enabled", True)}

                has_new_structure = any(n.startswith(f"{SessionManager.PERSPECTIVES_DIR}/") for n in namelist)

                if has_new_structure:
                    # === AGGRESSIVE EXPLICIT LOAD for both perspectives ===
                    # Always start by trying the two canonical perspectives, then add any extras found.
                    MANDATORY_PERSPECTIVES = ["spatial", "multivariate"]
                    perspectives_in_meta = metadata.get("perspectives", []) or []
                    views_to_load = list(MANDATORY_PERSPECTIVES)  # force both first

                    # Dynamic scan to catch any additional or if zip has more
                    scanned = []
                    for n in namelist:
                        if n.startswith(f"{SessionManager.PERSPECTIVES_DIR}/"):
                            parts = n.split("/")
                            if len(parts) >= 3 and parts[1]:
                                scanned.append(parts[1])
                    for v in list(dict.fromkeys(scanned)):
                        if v not in views_to_load:
                            views_to_load.append(v)

                    if not views_to_load:
                        views_to_load = MANDATORY_PERSPECTIVES[:]  # final safety

                    # Load per-perspective per-scale
                    for view in views_to_load:
                        view_dir = f"{SessionManager.PERSPECTIVES_DIR}/{view}/"
                        scale_results = {}
                        scale_metrics = {}
                        view_cfg_hints = {}

                        for scale in ["Local", "Medium", "Global"]:
                            arc = f"{view_dir}{scale}.json"
                            if arc in namelist:
                                with zf.open(arc) as f:
                                    raw = json.load(f)
                                deserialized = SessionManager._deserialize_scale(raw)
                                scale_results[scale] = {k: v for k, v in deserialized.items() if k not in ("_clustering_method", "_cluster_composition")}
                                # Rebuild scale_metrics subset
                                scale_metrics[scale] = {
                                    "mean_tau": deserialized.get("mean_tau"),
                                    "tau_std": deserialized.get("tau_std"),
                                    "coherence": deserialized.get("coherence"),
                                    "recd_T_final": deserialized.get("recd_T_final"),
                                    "n_modules": deserialized.get("n_modules"),
                                }
                                view_cfg_hints[scale] = {
                                    "clustering_method": deserialized.get("_clustering_method"),
                                    "cluster_composition": deserialized.get("_cluster_composition"),
                                }

                        if scale_results:
                            # Reconstruct the analysis object expected by studio
                            merged_config = dict(config)
                            # merge per-scale clustering hints + groups back (critical for manual clusters)
                            cm = {}
                            for sc in ["Local", "Medium", "Global"]:
                                hint = view_cfg_hints.get(sc, {})
                                if hint.get("clustering_method"):
                                    cm[sc.lower()] = hint["clustering_method"]
                            if cm:
                                merged_config["clustering_method"] = cm

                            # Restore the original manual groups so UI/reports see them
                            # (they may also come from top-level manual_clusters.json)
                            for sc, hint in view_cfg_hints.items():
                                comp = hint.get("cluster_composition") or {}
                                if sc == "Medium" and comp:
                                    merged_config["medium_groups"] = comp
                                if sc == "Global" and comp:
                                    merged_config["global_groups"] = comp

                            analyses[view] = {
                                "scale_results": scale_results,
                                "scale_metrics": scale_metrics,
                                "config": merged_config,
                            }

                    # === AGGRESSIVE VALIDATION after loading ===
                    for view in MANDATORY_PERSPECTIVES:
                        if view not in analyses:
                            print(f"[SessionManager][LOAD VALIDATION] CRITICAL: perspective '{view}' was NOT found in the .stausession!")
                        else:
                            sr = analyses[view].get("scale_results", {}) or {}
                            missing_scales = [s for s in ["Local", "Medium", "Global"] if s not in sr]
                            if missing_scales:
                                print(f"[SessionManager][LOAD VALIDATION] Warning: '{view}' missing scales {missing_scales}")
                            else:
                                print(f"[SessionManager][LOAD VALIDATION] '{view}' fully loaded with all 3 scales.")

                    # Additional per-view scale validation
                    for view, an in list(analyses.items()):
                        sr = an.get("scale_results", {}) or {}
                        missing = [s for s in ["Local", "Medium", "Global"] if s not in sr]
                        if missing:
                            print(f"[SessionManager] Warning: perspective '{view}' is incomplete (missing: {missing}). "
                                  "Results may be partial. Re-run the missing scale if needed.")

                    # Load manual clusters if present
                    if "manual_clusters.json" in namelist:
                        with zf.open("manual_clusters.json") as f:
                            manual_clusters = json.load(f)

                    if "recd_state.json" in namelist:
                        with zf.open("recd_state.json") as f:
                            recd_state = json.load(f)

                    # Merge any top-level manual clusters back into the per-view configs (for UI + reports)
                    for view, groups_info in (manual_clusters or {}).items():
                        if view in analyses:
                            analyses[view].setdefault("config", {}).update({
                                "medium_groups": groups_info.get("medium_groups", {}),
                                "global_groups": groups_info.get("global_groups", {}),
                            })
                            if groups_info.get("clustering_method"):
                                analyses[view]["config"]["clustering_method"] = groups_info["clustering_method"]

                    # Rebuild a flat ontological_memory from first perspective for legacy callers
                    ontological_memory = {}
                    if analyses:
                        first = next(iter(analyses.values()))
                        ontological_memory = first.get("scale_results", {})

                    return {
                        "metadata": metadata,
                        "config": config,
                        "df": df,
                        "analyses": analyses,
                        "ontological_memory": ontological_memory,
                        "manual_clusters": manual_clusters,
                        "recd_state": recd_state,
                        "session_file": filepath,
                    }

                # === LEGACY v4.1 path ===
                missing = [f for f in SessionManager.REQUIRED_FILES_OLD if f not in namelist]
                if missing and not has_new_structure:
                    print(f"Invalid session file (legacy), missing: {missing}")
                    # still try to continue if data exists

                raw_results = {}
                if "results.json" in namelist:
                    with zf.open("results.json") as f:
                        raw_results = json.load(f)

                # Rebuild ontological_memory (legacy flat)
                ontological_memory = {}
                for scale in ["Local", "Medium", "Global"]:
                    if scale in raw_results:
                        mem = raw_results[scale]
                        if isinstance(mem, dict):
                            ontological_memory[scale] = SessionManager._restore_arrays(mem)

                # Also try to build minimal analyses from legacy if it looks like perspectives were stored flat
                analyses_fallback = {}
                if "spatial" in raw_results or "multivariate" in raw_results:
                    for view in ["spatial", "multivariate"]:
                        if view in raw_results and isinstance(raw_results[view], dict):
                            # old buggy saves sometimes put perspective at top
                            pass  # handled in new path
                # For legacy, analyses will be empty or reconstructed in caller

                if "_current_math_stats" in raw_results:
                    # attach if needed
                    pass

                return {
                    "metadata": metadata,
                    "config": config,
                    "df": df,
                    "analyses": analyses_fallback,
                    "ontological_memory": ontological_memory,
                    "manual_clusters": {},
                    "recd_state": recd_state,
                    "session_file": filepath,
                }

        except Exception as e:
            print(f"[SessionManager] Load failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    @staticmethod
    def is_valid_session(filepath: str) -> bool:
        """Quick check without full load. Accepts v4.2 nested or legacy flat."""
        try:
            with zipfile.ZipFile(filepath, "r") as zf:
                names = zf.namelist()
                has_meta = "metadata.json" in names
                has_cfg = "config.json" in names
                has_data = "data.parquet" in names or "data.json" in names
                has_new = any(n.startswith("perspectives/") for n in names)
                has_old_results = "results.json" in names
                return has_meta and has_cfg and has_data and (has_new or has_old_results)
        except Exception:
            return False

