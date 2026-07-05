import os
import folium
import webbrowser
import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as colors

class SystemicTauMapGenerator:
    
    @staticmethod
    def _get_color_from_value(val, vmin, vmax, cmap_name='coolwarm'):
        """Maps a numeric value to a hex color using matplotlib colormaps."""
        if pd.isna(val):
            return "#808080" # Gray for NaN
            
        # Normalize
        if vmax <= vmin:
            norm_val = 0.5
        else:
            norm_val = (val - vmin) / (vmax - vmin)
            
        # Clip to [0, 1]
        norm_val = max(0.0, min(1.0, norm_val))
        
        import matplotlib as mpl
        cmap = mpl.colormaps[cmap_name]
        rgba = cmap(norm_val)
        return colors.to_hex(rgba)

    @staticmethod
    def generate_map(results_df: pd.DataFrame, coords_df: pd.DataFrame, location_col: str, 
                     scale_markers: bool = True, output_file: str = "systemic_tau_map.html", macro_clusters: dict = None):
        """
        Generates an interactive Folium map based on the results and coordinates.
        
        Args:
            results_df: DataFrame containing analysis results (Location, Tau_Max, p_value, etc.)
            coords_df: DataFrame containing mapping (Location, latitude, longitude)
            location_col: Name of the location column
            scale_markers: Whether to scale marker radius by Tau_Max
            output_file: Path to save the HTML map
        """
        if results_df.empty or coords_df.empty:
            raise ValueError("Empty results or coordinates data.")
            
        # Standardize column names for merge
        res_copy = results_df.copy()
        crd_copy = coords_df.copy()
        
        # Assume 'Location' is the key in results_df, and 'location_col' is the key in crd_copy
        if 'Location' not in res_copy.columns:
            raise ValueError("Results DataFrame must contain a 'Location' column.")
            
        if location_col not in crd_copy.columns:
            # If location_col is not found, assume the first column is the location key
            crd_copy = crd_copy.rename(columns={crd_copy.columns[0]: location_col})
            
        # Ensure both merge keys are of the same type (string) to prevent pandas ValueError
        res_copy['Location'] = res_copy['Location'].astype(str).str.replace(r'\.0$', '', regex=True)
        crd_copy[location_col] = crd_copy[location_col].astype(str).str.replace(r'\.0$', '', regex=True)
        
        if macro_clusters:
            # Synthesize centroids for MacroClusters
            new_rows = []
            for loc in res_copy['Location'].unique():
                if loc in macro_clusters:
                    components = macro_clusters[loc]
                    # Find these components in coords
                    clean_components = [str(c).replace('.0', '') for c in components]
                    component_crds = crd_copy[crd_copy[location_col].isin(clean_components)]
                    if not component_crds.empty:
                        centroid_lat = component_crds['latitude'].mean()
                        centroid_lon = component_crds['longitude'].mean()
                        new_rows.append({
                            location_col: loc,
                            'latitude': centroid_lat,
                            'longitude': centroid_lon
                        })
            if new_rows:
                crd_copy = pd.concat([crd_copy, pd.DataFrame(new_rows)], ignore_index=True)
            
        # Merge
        merged = pd.merge(res_copy, crd_copy, left_on='Location', right_on=location_col, how='inner')
        if merged.empty:
            raise ValueError("Could not match any locations between results and coordinates.")
            
        # Determine map center
        center_lat = merged['latitude'].mean()
        center_lon = merged['longitude'].mean()
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=4, tiles='CartoDB positron')
        
        # Calculate color scaling bounds for Tau_Max
        tau_values = pd.to_numeric(merged['Tau_Max'], errors='coerce').dropna()
        if not tau_values.empty:
            tau_min = tau_values.min()
            tau_max = tau_values.max()
        else:
            tau_min, tau_max = 0, 1
            
        for _, row in merged.iterrows():
            lat = row['latitude']
            lon = row['longitude']
            
            if pd.isna(lat) or pd.isna(lon):
                continue
                
            loc_id = row['Location']
            tau_val = row.get('Tau_Max', np.nan)
            
            try:
                tau_val_num = float(tau_val)
            except:
                tau_val_num = np.nan
                
            p_val = row.get('p_value', np.nan)
            verdict = row.get('Verdict', 'N/A')
            leading = row.get('Leading_Driver', 'N/A')
            
            # Determine Color
            color = SystemicTauMapGenerator._get_color_from_value(tau_val_num, tau_min, tau_max, cmap_name='coolwarm')
            
            # Determine Size
            from systemictau.desktop.settings import AppSettings
            app_settings = AppSettings()
            max_size_setting = app_settings.get("map_bubble_max_size", 25)
            
            base_radius = 8
            if scale_markers and not np.isnan(tau_val_num) and tau_max > tau_min:
                norm_tau = (tau_val_num - tau_min) / (tau_max - tau_min)
                radius = base_radius + (norm_tau * (max_size_setting - base_radius))
            else:
                radius = base_radius
                
            # Formatting for popup
            try: p_val_str = f"{float(p_val):.4f}" 
            except: p_val_str = str(p_val)
            
            try: tau_str = f"{float(tau_val):.2f}"
            except: tau_str = str(tau_val)
            
            target_name = row.get('Target', 'Tau_s')
            if target_name == "Systemic Coupling":
                metric_label = "Systemic Coupling (Kendall)"
                tooltip_label = "Kendall"
            else:
                metric_label = "Structural Intensity (Tau_s)"
                tooltip_label = "Tau_s"
                
            macro_label = ""
            if macro_clusters and loc_id in macro_clusters:
                macro_label = f"<span style='color: #d35400; font-size: 0.9em; font-weight: bold;'>[MACRO-CLUSTER CENTROID]</span><br>"

            popup_html = f"""
            <div style="font-family: Arial, sans-serif; width: 250px;">
                <h4 style="margin-bottom: 5px; color: #333;">{loc_id}</h4>
                {macro_label}
                <hr style="margin-top: 0; margin-bottom: 5px;">
                <b>{metric_label}:</b> {tau_str}<br>
                <b>p-value:</b> {p_val_str}<br>
                <b>Verdict:</b> {verdict}<br>
                <b>Leading Driver:</b> {leading}
            </div>
            """
            
            folium.CircleMarker(
                location=[lat, lon],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"{loc_id} ({tooltip_label}: {tau_str})"
            ).add_to(m)
            
            # Draw Convex Hull / Polyline for MacroClusters
            if macro_clusters and loc_id in macro_clusters:
                components = macro_clusters[loc_id]
                clean_components = [str(c).replace('.0', '') for c in components]
                component_crds = crd_copy[crd_copy[location_col].isin(clean_components)]
                if len(component_crds) >= 3:
                    pts = component_crds[['latitude', 'longitude']].values
                    try:
                        from scipy.spatial import ConvexHull
                        hull = ConvexHull(pts)
                        hull_points = pts[hull.vertices]
                        # close the loop
                        hull_points = np.vstack((hull_points, hull_points[0]))
                        folium.Polygon(
                            locations=hull_points.tolist(),
                            color=color,
                            weight=2,
                            fill=True,
                            fill_color=color,
                            fill_opacity=0.3,
                            tooltip=f"{loc_id} Region"
                        ).add_to(m)
                    except Exception:
                        pass
                elif len(component_crds) == 2:
                    pts = component_crds[['latitude', 'longitude']].values.tolist()
                    folium.PolyLine(
                        locations=pts,
                        color=color,
                        weight=3,
                        opacity=0.6,
                        tooltip=f"{loc_id} Link"
                    ).add_to(m)
            
        m.save(output_file)
        
        # Auto-open in browser
        filepath = os.path.abspath(output_file)
        webbrowser.open(f"file://{filepath}")
        
        return filepath
