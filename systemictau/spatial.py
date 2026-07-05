import numpy as np

def spatial_tau(gdf, value_cols: list, geometry_col: str = 'geometry', k_neighbors: int = 4, window_size: int = 13, stride: int = 1):
    """
    Computes Systemic Tau on a spatial dataset (GeoDataFrame) to identify 
    geographic hotspots of synchrony.
    
    Parameters:
    -----------
    gdf : geopandas.GeoDataFrame
        The spatial dataset containing regions/points.
    value_cols : list of str
        The columns containing time series data or multivariate features for each region.
        In an epidemiological context, this could be Dengue cases over T weeks.
    geometry_col : str, optional
        The name of the geometry column. Default is 'geometry'.
    k_neighbors : int, optional
        Number of spatial neighbors to consider for each region using KNN.
    window_size : int, optional
        Window size for Systemic Tau computation.
    stride : int, optional
        Stride for Systemic Tau computation.
        
    Returns:
    --------
    geopandas.GeoDataFrame
        A new GeoDataFrame with added 'spatial_tau' metric per region and a 'hotspot_flag'.
    """
    try:
        import geopandas as gpd
    except ImportError:
        raise ImportError("GIS module requires geopandas. Run 'pip install systemictau[gis]'")
        
    try:
        from libpysal.weights import KNN
    except ImportError:
        raise ImportError("GIS module requires libpysal. Run 'pip install systemictau[gis]'")
        
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError("gdf must be a GeoDataFrame")
        
    from .core import compute_taus
    
    # Ensure geometry is set
    if geometry_col != gdf.active_geometry_name:
        gdf = gdf.set_geometry(geometry_col)
        
    res_gdf = gdf.copy()
    
    # Compute KNN weights
    w = KNN.from_dataframe(res_gdf, k=k_neighbors)
    
    spatial_tau_scores = np.zeros(len(res_gdf))
    
    for i in range(len(res_gdf)):
        neighbors = w.neighbors[i]
        
        # Combine region i and its neighbors
        idx_group = [i] + neighbors
        
        # Extract multivariate matrix X (T, N) where N is (1 + k_neighbors)
        # We assume value_cols are features over time. But wait, if value_cols are
        # time points, X should be of shape (T, N).
        # Typically in GIS, each row is a region, columns are time points: T = len(value_cols)
        # So for group, we extract the matrix and transpose it to shape (T, N)
        X = res_gdf.iloc[idx_group][value_cols].values.T
        
        taus_global, _ = compute_taus(X, window_size=window_size, stride=stride)
        
        # We can take the mean of the temporal tau evolution as the overall spatial synchrony score
        # (Ignoring NaNs)
        spatial_tau_scores[i] = np.nanmean(taus_global)
        
    res_gdf['spatial_tau'] = spatial_tau_scores
    
    # Identify hotspots using IQR method
    Q1 = np.nanpercentile(spatial_tau_scores, 25)
    Q3 = np.nanpercentile(spatial_tau_scores, 75)
    IQR = Q3 - Q1
    threshold = Q3 + 1.5 * IQR
    
    res_gdf['hotspot_flag'] = (res_gdf['spatial_tau'] > threshold).astype(int)
    
    return res_gdf
