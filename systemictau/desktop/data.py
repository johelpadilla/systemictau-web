import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self):
        self.raw_df = None
        self.processed_df = None
        self.coords_df = None
        self.is_panel = False
        self.location_col = None
        self.time_col = None
        self.macro_cluster_compositions = {}
        
    def load_file(self, file_path: str):
        """Loads a file and performs automatic panel data detection."""
        if file_path.endswith('.csv'):
            self.raw_df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            self.raw_df = pd.read_excel(file_path)
            # Handle case where Excel file is actually a CSV dumped into one column
            if len(self.raw_df.columns) == 1 and ',' in str(self.raw_df.columns[0]):
                col_name = self.raw_df.columns[0]
                new_cols = col_name.split(',')
                # Split the single column by comma and expand into multiple columns
                split_df = self.raw_df[col_name].astype(str).str.split(',', expand=True)
                if split_df.shape[1] == len(new_cols):
                    split_df.columns = new_cols
                    # Attempt to convert to numeric where possible
                    for c in split_df.columns:
                        converted = pd.to_numeric(split_df[c], errors='coerce')
                        if not converted.isna().all():
                            # If it's mostly numbers (some NaNs are okay), use it
                            split_df[c] = converted
                    self.raw_df = split_df
        else:
            raise ValueError("Unsupported file format. Please load a CSV or Excel file.")
            
        self._detect_panel_data()
        self._auto_detect_coordinates()
        return self.raw_df

    def _auto_detect_coordinates(self):
        """Attempts to auto-detect lat/lon from the loaded dataframe."""
        if self.raw_df is None or not self.is_panel or not self.location_col:
            return
            
        cols = [str(c).lower().strip() for c in self.raw_df.columns]
        lat_names = ['lat', 'latitude', 'latitud', 'y']
        lon_names = ['lon', 'long', 'longitude', 'longitud', 'x']
        
        lat_c = next((c for c in self.raw_df.columns if str(c).lower().strip() in lat_names), None)
        lon_c = next((c for c in self.raw_df.columns if str(c).lower().strip() in lon_names), None)
        
        if lat_c and lon_c:
            # Extract mapping NID -> (lat, lon)
            mapping = self.raw_df[[self.location_col, lat_c, lon_c]].drop_duplicates(subset=[self.location_col])
            mapping = mapping.rename(columns={lat_c: 'latitude', lon_c: 'longitude'})
            mapping = mapping.dropna(subset=['latitude', 'longitude'])
            if not mapping.empty:
                self.coords_df = mapping
                logger.info(f"Auto-detected {len(self.coords_df)} coordinates from main dataset.")

    def load_coordinates_file(self, file_path: str):
        """Loads a separate CSV file mapping locations to latitude and longitude."""
        if not file_path.endswith('.csv'):
            raise ValueError("Coordinates file must be a CSV.")
            
        df = pd.read_csv(file_path)
        cols = [str(c).lower() for c in df.columns]
        
        # Identify location col (try to match the one in main dataset)
        loc_col = None
        if self.location_col and self.location_col.lower() in cols:
            idx = cols.index(self.location_col.lower())
            loc_col = df.columns[idx]
        else:
            # Try some common names
            for c in df.columns:
                n = str(c).lower()
                if 'nid' in n or 'loc' in n or 'site' in n or 'id' in n:
                    loc_col = c
                    break
            if not loc_col:
                loc_col = df.columns[0] # Fallback to first column
                
        lat_col = next((c for c in df.columns if str(c).lower() in ['lat', 'latitude']), None)
        lon_col = next((c for c in df.columns if str(c).lower() in ['lon', 'long', 'longitude']), None)
        
        if not lat_col or not lon_col:
            raise ValueError("Coordinates file must contain 'latitude' and 'longitude' columns.")
            
        mapping = df[[loc_col, lat_col, lon_col]].copy()
        # Rename loc_col to match main dataset if it exists
        if self.location_col:
            mapping = mapping.rename(columns={loc_col: self.location_col})
        mapping = mapping.rename(columns={lat_col: 'latitude', lon_col: 'longitude'})
        mapping = mapping.dropna(subset=['latitude', 'longitude'])
        
        self.coords_df = mapping
        logger.info(f"Loaded {len(self.coords_df)} coordinates from {file_path}")
        return self.coords_df

    def _detect_panel_data(self):
        """
        Detects if the dataset is Panel Data by looking for:
        1. A column with moderate cardinality (5-300 unique values) representing locations.
        2. A time column with high cardinality.
        """
        if self.raw_df is None or self.raw_df.empty:
            return
            
        # 1. Identify potential location column (moderate cardinality string/int)
        potential_loc_cols = []
        for col in self.raw_df.columns:
            n_unique = self.raw_df[col].nunique()
            if 5 <= n_unique <= 300:
                potential_loc_cols.append(col)
                
        # 2. Identify potential time column (high cardinality or datetime dtype)
        potential_time_cols = []
        for col in self.raw_df.columns:
            if pd.api.types.is_datetime64_any_dtype(self.raw_df[col]):
                potential_time_cols.append(col)
            elif 'time' in str(col).lower() or 'date' in str(col).lower() or 'week' in str(col).lower() or 'year' in str(col).lower():
                potential_time_cols.append(col)
                
        if not potential_time_cols:
            # Fallback: find column with highest unique values that is monotonic or near-monotonic
            for col in self.raw_df.columns:
                n_unique = self.raw_df[col].nunique()
                if n_unique > 50:
                    potential_time_cols.append(col)

        # If we have both, it's likely panel data
        if potential_loc_cols and potential_time_cols:
            # Pick the most likely candidates based on naming heuristics
            loc_col = None
            for col in potential_loc_cols:
                name = str(col).lower()
                if 'nid' in name or 'loc' in name or 'site' in name or 'name' in name or 'city' in name or 'region' in name:
                    loc_col = col
                    break
            if not loc_col:
                loc_col = potential_loc_cols[0]
                
            time_col = None
            for col in potential_time_cols:
                name = str(col).lower()
                if 'week' in name or 'date' in name or 'time' in name:
                    time_col = col
                    break
            if not time_col:
                time_col = potential_time_cols[0]
                
            # Final check: are there multiple locations per time step?
            # If grouping by time yields > 1 row on average, it's cross-sectional/panel
            avg_rows_per_time = self.raw_df.groupby(time_col).size().mean()
            if avg_rows_per_time > 1.5:
                self.is_panel = True
                self.location_col = loc_col
                self.time_col = time_col
                return
                
        self.is_panel = False
        
    def aggregate_panel_data(self, target_col: str):
        """
        Aggregates panel data by time column.
        Sums the primary target variable, averages the rest.
        """
        if not self.is_panel or not self.time_col:
            self.processed_df = self.raw_df.copy()
            return self.processed_df
            
        numeric_cols = self.raw_df.select_dtypes(include='number').columns.tolist()
        if target_col not in numeric_cols and target_col in self.raw_df.columns:
            # Target is not numeric, we can't sum it
            raise ValueError(f"Target column {target_col} must be numeric for aggregation.")
            
        agg_dict = {}
        for col in numeric_cols:
            if col == self.time_col or col == self.location_col:
                continue
            if col == target_col:
                agg_dict[col] = 'sum'
            else:
                agg_dict[col] = 'mean'
                
        # Perform groupby
        aggregated = self.raw_df.groupby(self.time_col).agg(agg_dict).reset_index()
        self.processed_df = aggregated
        return self.processed_df
        
    def pivot_panel_to_systemic(self, target_col: str):
        """
        Pivots the dataset so that each location becomes a column (node in the system).
        This allows Systemic Tau to evaluate the correlation matrix of the entire network.
        """
        if not self.is_panel or not self.time_col or not self.location_col:
            self.processed_df = self.raw_df.copy()
            return self.processed_df
            
        numeric_cols = self.raw_df.select_dtypes(include='number').columns.tolist()
        if target_col not in numeric_cols and target_col in self.raw_df.columns:
            raise ValueError(f"Target column {target_col} must be numeric for systemic pivoting.")
            
        # Pivot table: rows=Time, cols=Location, values=Target
        # If there are duplicates per (time, location), mean them.
        pivoted = self.raw_df.pivot_table(
            index=self.time_col, 
            columns=self.location_col, 
            values=target_col, 
            aggfunc='mean'
        )
        
        # Reset index so time_col becomes a normal column again
        pivoted = pivoted.reset_index()
        
        # Rename columns to ensure they are strings, just in case
        pivoted.columns = [str(c) for c in pivoted.columns]
        
        self.processed_df = pivoted
        return self.processed_df

    def filter_by_location(self, location_values: list):
        """Filters the dataset for specific locations."""
        if not self.is_panel or not self.location_col:
            self.processed_df = self.raw_df.copy()
            return self.processed_df
            
        self.processed_df = self.raw_df[self.raw_df[self.location_col].isin(location_values)].copy()
        return self.processed_df

    def intelligent_downsample(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Intelligently downsamples a DataFrame if it exceeds max_rows_before_downsampling.
        Target size is roughly 10% to 20% of the max limit.
        """
        self.downsampled_flag = False
        n_rows = len(df)
        max_limit = getattr(self, 'max_rows', 80000)
        
        if n_rows <= max_limit:
            return df
            
        logger.info(f"Smart Downsampling triggered. Original size: {n_rows} rows.")
        self.downsampled_flag = True
        
        target_size = max_limit // 5 # Target ~16k rows if limit is 80k
        factor = n_rows // target_size
        
        if factor <= 1:
            return df
            
        numeric_cols = df.select_dtypes(include='number').columns
        other_cols = df.select_dtypes(exclude='number').columns
        
        # Block average for numerics
        df_num = df[numeric_cols].groupby(df.index // factor).mean()
        # Decimate for strings/dates
        df_oth = df[other_cols].iloc[::factor].reset_index(drop=True)
        
        downsampled = pd.concat([df_oth, df_num.reset_index(drop=True)], axis=1)
        logger.info(f"Downsampled size: {len(downsampled)} rows.")
        return downsampled
