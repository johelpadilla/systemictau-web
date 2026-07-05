import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

class OntologicalScaleManager:
    """
    Manages the ontological scale of the Systemic Tau analysis.
    Ensures non-linear systemic properties are preserved across scales
    by forbidding linear smoothing techniques (e.g., arithmetic means)
    during cluster aggregation.

    Global scale improvements (v4.x):
    - Automatic clustering uses principled hierarchical clustering on Kendall correlation distance.
    - Full transparency: composition of each MacroCluster is returned.
    - Manual cluster definition supported for user-controlled macro grouping (2-6 clusters).
    """

    SCALE_LOCAL = "Local"
    SCALE_MEDIUM = "Medium"
    SCALE_GLOBAL = "Global"

    # Class-level storage for manual definitions (simple state for desktop session lifetime)
    _manual_cluster_definitions: Dict[str, List[str]] = {}

    @staticmethod
    def aggregate_medium_cluster(df, target_cols, cluster_name="Cluster_1", agg_method="sum"):
        """
        Aggregates multiple target columns into a single meso-scale entity.
        
        Args:
            df (pd.DataFrame): The raw dataframe containing the time series.
            target_cols (list): List of column names to be aggregated into the cluster.
            cluster_name (str): The name of the resulting cluster column.
            agg_method (str): 'sum' (for extensive variables) or 'median' (for intensive variables).
                              Strictly forbids 'mean' to avoid linear smoothing of raw volatility.
                              
        Returns:
            pd.DataFrame: A new dataframe containing the aggregated cluster alongside non-target columns.
        """
        if agg_method not in ["sum", "median"]:
            raise ValueError("Aggregation method must be 'sum' or 'median' to preserve non-linear properties. Linear 'mean' is forbidden by Systemic Tau paradigm.")

        # Copy original dataframe to avoid mutating raw data
        agg_df = df.copy()

        valid_cols = []
        for col in target_cols:
            if col in agg_df.columns:
                valid_cols.append(col)
            else:
                try:
                    if int(col) in agg_df.columns:
                        valid_cols.append(int(col))
                except (ValueError, TypeError):
                    pass
                try:
                    if str(col) in agg_df.columns and str(col) not in valid_cols:
                        valid_cols.append(str(col))
                except Exception:
                    pass

        if not valid_cols:
            raise ValueError(f"None of the target columns {target_cols} were found in the dataset.")

        # Isolate the target columns
        subset = agg_df[valid_cols]

        if agg_method == "sum":
            cluster_series = subset.sum(axis=1)
        elif agg_method == "median":
            cluster_series = subset.median(axis=1)

        # Create a clean dataframe with the cluster
        result_df = pd.DataFrame()
        
        # Preserve non-target columns (like the time column)
        non_target_cols = [c for c in df.columns if c not in target_cols]
        for col in non_target_cols:
            result_df[col] = agg_df[col]
            
        result_df[cluster_name] = cluster_series
        return result_df

    @staticmethod
    def aggregate_global_clusters(df, target_cols, num_clusters=3, agg_method="sum"):
        """
        Automatically groups raw variables into macro-clusters using Correlation-based 
        Hierarchical Clustering. Used as a fallback when the user selects raw variables 
        at the Global scale instead of pre-computed Medium clusters.

        Guarantees at least 2 macro-clusters when >=2 input target_cols (required for Systemic Tau).
        """
        if len(target_cols) < 2:
            # Cannot compute systemic coupling; return as-is (caller must guard)
            return df.copy(), list(target_cols), {}

        # Systemic Tau requires >=2 modules at every scale. Force minimum 2 clusters.
        if num_clusters < 2:
            num_clusters = 2

        if len(target_cols) <= num_clusters:
            # Not enough to split further: each becomes its own macro (still >=2 here)
            return df.copy(), list(target_cols), {}

        if agg_method not in ["sum", "median"]:
            agg_method = "sum"

        # Rank-safe prep for ordinal clustering
        subset = df[target_cols].interpolate(method='linear', limit=5).ffill().bfill()
        
        # Mejora 4: Use Kendall (ordinal) distance instead of Pearson
        # This aligns with the rank-based nature of Systemic Tau / RECD
        corr_matrix = subset.corr(method='kendall').ffill().bfill()
        
        try:
            from scipy.cluster.hierarchy import linkage, fcluster
            import scipy.spatial.distance as ssd
            
            # Distance = 1 - |Kendall tau|  (concordance based)
            dist = 1 - np.abs(corr_matrix.values)
            dist = (dist + dist.T) / 2
            np.fill_diagonal(dist, 0)
            dist = np.clip(dist, 0, 2)
            
            condensed_dist = ssd.squareform(dist)
            Z = linkage(condensed_dist, method='ward')
            labels = fcluster(Z, num_clusters, criterion='maxclust')
        except Exception:
            # Fallback if scipy linkage fails: sequential chunking
            labels = np.array_split(np.arange(len(target_cols)), num_clusters)
            labels_mapped = np.zeros(len(target_cols), dtype=int)
            for i, chunk in enumerate(labels):
                for idx in chunk:
                    labels_mapped[idx] = i + 1
            labels = labels_mapped
            
        agg_df = df.copy()
        new_targets = []
        cluster_components_dict = {}
        
        for i in range(1, num_clusters + 1):
            cluster_cols = [target_cols[j] for j in range(len(target_cols)) if labels[j] == i]
            if not cluster_cols:
                continue
            
            cluster_name = f"MacroCluster_{agg_method.upper()}_{i}_{len(cluster_cols)}vars"
            if agg_method == "sum":
                agg_df[cluster_name] = subset[cluster_cols].sum(axis=1)
            else:
                agg_df[cluster_name] = subset[cluster_cols].median(axis=1)
            
            new_targets.append(cluster_name)
            cluster_components_dict[cluster_name] = cluster_cols

        # Defense in depth: clustering (scipy or fallback) can collapse to <2
        # when the input "modules" (e.g. your hypothetical clusters) are highly correlated.
        # Systemic Tau still needs >=2 at Global.
        if len(new_targets) < 2 and len(target_cols) >= 2:
            # Force a simple non-empty split of the original target_cols
            split = max(1, len(target_cols) // 2)
            for ii, (start, end) in enumerate( [(0, split), (split, len(target_cols))] ):
                part = target_cols[start:end] or target_cols
                cname = f"MacroCluster_FORCED_{agg_method.upper()}_{ii+1}_{len(part)}vars"
                if agg_method == "sum":
                    agg_df[cname] = subset[part].sum(axis=1)
                else:
                    agg_df[cname] = subset[part].median(axis=1)
                if cname not in new_targets:
                    new_targets.append(cname)
                    cluster_components_dict[cname] = part
            # Trim to at least the first 2 if more were added
            new_targets = new_targets[:2] if len(new_targets) > 2 else new_targets

        return agg_df, new_targets, cluster_components_dict

    # ============================================================
    # NEW: Manual Cluster Definition + Unified Global Getter (per Implementation Plan)
    # ============================================================

    @classmethod
    def define_manual_clusters(cls, user_groups: Dict[str, List[Any]]) -> Dict[str, List[str]]:
        """
        Validate and store user-defined macro-clusters for Global scale.
        Supports 2 to 6 clusters. Each cluster must contain >=1 variable.

        Example:
            {
                "Socioeconomic": ["income", "education", "unemployment"],
                "Environmental": ["temp", "precip", "ndvi"],
                "Demographic": ["pop", "density"]
            }

        Returns the cleaned definition dict (cluster_name -> list of var names).
        Raises ValueError on invalid input.
        """
        if not isinstance(user_groups, dict):
            raise ValueError("user_groups must be a dict of {cluster_name: [var1, var2, ...]}")

        if not (2 <= len(user_groups) <= 6):
            raise ValueError("Define between 2 and 6 macro-clusters for Global scale.")

        cleaned: Dict[str, List[str]] = {}
        for raw_name, vars_list in user_groups.items():
            name = str(raw_name).strip()
            if not name:
                name = f"MacroCluster_{len(cleaned) + 1}"
            if not isinstance(vars_list, (list, tuple)):
                vars_list = [vars_list]
            valid = [str(v).strip() for v in vars_list if str(v).strip()]
            if valid:
                # De-dupe within cluster while preserving order
                seen = set()
                deduped = []
                for v in valid:
                    if v not in seen:
                        seen.add(v)
                        deduped.append(v)
                cleaned[name] = deduped

        if len(cleaned) < 2:
            raise ValueError("At least two non-empty macro-clusters are required.")

        cls._manual_cluster_definitions = cleaned
        return cleaned

    @classmethod
    def get_manual_cluster_definitions(cls) -> Dict[str, List[str]]:
        """Return the currently defined manual clusters (may be empty)."""
        return dict(cls._manual_cluster_definitions)

    @classmethod
    def clear_manual_clusters(cls) -> None:
        cls._manual_cluster_definitions = {}

    @staticmethod
    def aggregate_manual_clusters(df: pd.DataFrame, manual_groups: Dict[str, List[str]],
                                  agg_method: str = "sum") -> Tuple[pd.DataFrame, List[str], Dict[str, List[str]]]:
        """
        Build aggregated macro-cluster columns from explicit user groups.
        Returns (augmented_df, list_of_new_macro_names, composition_dict)
        """
        if agg_method not in ("sum", "median"):
            agg_method = "sum"

        agg_df = df.copy()
        new_targets: List[str] = []
        composition: Dict[str, List[str]] = {}

        for cname, cols in manual_groups.items():
            valid = [c for c in cols if c in df.columns]
            if not valid:
                continue
            # Sanitize cluster name for column use
            safe_name = str(cname).replace(" ", "_").replace("/", "_")
            if agg_method == "sum":
                agg_df[safe_name] = df[valid].sum(axis=1)
            else:
                agg_df[safe_name] = df[valid].median(axis=1)
            new_targets.append(safe_name)
            composition[safe_name] = valid

        if len(new_targets) < 2:
            # Caller must handle; we still return what we have
            pass

        return agg_df, new_targets, composition

    @staticmethod
    def get_global_clusters(df: pd.DataFrame,
                            target_cols: List[str],
                            method: str = "auto",
                            num_clusters: int = 3,
                            manual_groups: Optional[Dict[str, List[str]]] = None,
                            agg_method: str = "sum") -> Tuple[pd.DataFrame, List[str], Dict[str, List[str]]]:
        """
        Unified dispatcher for Global scale macro-clustering.

        Args:
            method: "auto" (hierarchical) or "manual"
            num_clusters: used only for auto (2-6 recommended)
            manual_groups: explicit user definition when method="manual"

        Returns:
            (df_with_macros, macro_target_names, composition_dict)
            composition_dict maps macro_name -> original variables (for reports)
        """
        method = (method or "auto").lower().strip()
        if method == "manual":
            groups = manual_groups or OntologicalScaleManager.get_manual_cluster_definitions()
            if not groups:
                raise ValueError("Manual clustering selected but no groups defined. Use define_manual_clusters() first.")
            return OntologicalScaleManager.aggregate_manual_clusters(df, groups, agg_method)

        # Default: automatic (improved transparency + existing robust logic)
        return OntologicalScaleManager.aggregate_global_clusters(
            df, target_cols, num_clusters=num_clusters, agg_method=agg_method
        )

    @staticmethod
    def describe_clustering(method: str, composition: Dict[str, List[str]], num_clusters: Optional[int] = None) -> str:
        """
        Human-readable description for reports / UI.
        """
        method = (method or "auto").lower()
        if method == "manual":
            lines = ["User-defined manual macro-clusters:"]
            for mc, members in composition.items():
                lines.append(f"  • {mc} ← [{', '.join(members)}]")
            return "\n".join(lines)
        else:
            base = "Automatic clustering (Ward hierarchical on Kendall-τ distance matrix)"
            if num_clusters:
                base += f" (k={num_clusters})"
            lines = [base]
            for mc, members in composition.items():
                lines.append(f"  • {mc} ← [{', '.join(members)}]")
            return "\n".join(lines)
