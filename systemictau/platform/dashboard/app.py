import streamlit as st
import pandas as pd
import numpy as np
import requests

from systemictau import from_dataframe, ChaosGenerator
from systemictau.visualization import plot_tau_evolution
from systemictau.layers import (
    hyper_persistence, rolling_rqa, critical_mass_metric, 
    compute_antisynchronization, extract_joint_episodes,
    detect_reorganization_frob, detect_reorganization_ks, consensus_transition
)
from systemictau.recd import compute_recd_increments, accumulate_time

try:
    import folium
    from streamlit_folium import st_folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

from systemictau.config import settings

API_URL = settings.api_url

st.sidebar.header("Authentication")
if 'token' not in st.session_state:
    with st.sidebar.form("login_form"):
        st.write("Login to API")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Login"):
            import requests
            try:
                resp = requests.post(f"{API_URL}/token", json={"username": username, "password": password})
                if resp.status_code == 200:
                    st.session_state['token'] = resp.json()["access_token"]
                    st.sidebar.success("Logged in!")
                else:
                    st.sidebar.error("Invalid credentials")
            except Exception as e:
                st.sidebar.error(f"Login failed: {e}")
else:
    st.sidebar.success("Authenticated")
    if st.sidebar.button("Logout"):
        del st.session_state['token']

st.set_page_config(page_title="Systemic Tau Platform", page_icon="📈", layout="wide")

st.title("Systemic Tau v5.0 : The Ontological Dynamics Platform")
st.markdown("Real-time topological data analysis and causal mapping for complex systems.")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Data Input", "Systemic Tau Computation", "Layer Analysis", "Spatial Analysis", "Neo4j Knowledge Graph", "Live Feed"])

with tab1:
    st.header("Data Input")
    input_mode = st.radio("Select input mode:", ["Synthetic (Chaos Generator)", "Upload CSV"])
    
    df = None
    if input_mode == "Synthetic (Chaos Generator)":
        col1, col2, col3 = st.columns(3)
        with col1:
            n_steps = st.slider("Time steps (T)", 100, 2000, 500)
        with col2:
            n_comp = st.slider("Number of components (N)", 2, 20, 5)
        with col3:
            coupling = st.slider("Coupling Strength", 0.0, 0.5, 0.1)
        
        if st.button("Generate Data"):
            X = ChaosGenerator.logistic_map_coupled(n_steps, n_comp, coupling=coupling)
            df = pd.DataFrame(X, columns=[f"V{i+1}" for i in range(n_comp)])
            st.session_state['df'] = df
            st.success("Synthetic data generated!")
            
    else:
        uploaded_file = st.file_uploader("Upload CSV", type="csv")
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state['df'] = df
                st.success("File uploaded!")
            except Exception as e:
                st.error(f"Error loading CSV: {e}")
            
    if 'df' in st.session_state:
        st.write("Preview:")
        st.dataframe(st.session_state['df'].head())

with tab2:
    st.header("Systemic Tau ($\\tau_s$)")
    
    if 'df' in st.session_state:
        window_size = st.slider("Sliding Window Size", 3, 100, 13)
        stride = st.slider("Stride", 1, 10, 1)
        
        if st.button("Compute Systemic Tau"):
            with st.spinner("Computing..."):
                try:
                    res = from_dataframe(st.session_state['df'], window_size=window_size, stride=stride)
                    st.session_state['res'] = res
                    st.success(f"Computed in {res.metadata['computation_time_seconds']:.2f} seconds.")
                except Exception as e:
                    st.error(f"Computation failed: {e}")
                    
        if 'res' in st.session_state:
            st.subheader("Evolution Plot")
            fig = plot_tau_evolution(st.session_state['res'].taus_global).figure
            st.pyplot(fig)
    else:
        st.info("Please load data in the first tab.")

with tab3:
    st.header("Ontological Layers & Joint Episodes")
    
    if 'res' in st.session_state:
        res = st.session_state['res']
        taus = res.taus_global
        
        st.sidebar.header("Layer Parameters")
        theta_A = st.sidebar.slider("Theta A (Antisync)", 0.0, 0.5, 0.04)
        theta_M = st.sidebar.slider("Theta M (Critical Mass)", 0.0, 5.0, 1.0)
        D_min = st.sidebar.slider("Minimum Duration (D_min)", 1, 100, 30)
        
        if st.button("Extract Layers & Detect Ascent"):
            with st.spinner("Extracting Layers..."):
                try:
                    # Layer computations
                    hp_z, core_hyper = hyper_persistence(taus)
                    lam, tt = rolling_rqa(taus)
                    M = critical_mass_metric(hp_z, lam, tt)
                    A = compute_antisynchronization(res.taus_per_module)
                    
                    episodes = extract_joint_episodes(A, M, theta_A=theta_A, D_min=D_min, theta_M=theta_M)
                    
                    # Ascent Detection
                    t_frob, _ = detect_reorganization_frob(res.taus_per_module)
                    dtk = compute_recd_increments(taus)
                    t_ks, _ = detect_reorganization_ks(dtk)
                    t_star = consensus_transition(t_frob, t_ks)
                    
                    st.session_state['episodes'] = episodes
                    st.session_state['t_star'] = t_star
                    st.session_state['M'] = M
                    st.session_state['dtk'] = dtk
                    
                    st.success(f"Found {len(episodes)} Joint Episodes. Consensus t* = {t_star}")
                except Exception as e:
                    st.error(f"Layer extraction failed: {e}")
                    
        if 'episodes' in st.session_state:
            st.subheader("Joint Episodes Table")
            df_ep = pd.DataFrame(st.session_state['episodes'])
            if not df_ep.empty:
                st.dataframe(df_ep)
            else:
                st.info("No joint episodes detected with current parameters.")
                
            st.subheader("RECD Time Evolution")
            T_series = accumulate_time(st.session_state['dtk'])
            st.line_chart(T_series, y_label="Discrete Extramental Time (T)")
            
            # Export functionality
            csv_export = df_ep.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Export Episodes CSV",
                data=csv_export,
                file_name='joint_episodes.csv',
                mime='text/csv',
            )
    else:
        st.info("Compute Systemic Tau first.")

with tab4:
    st.header("Spatial / GIS Analysis")
    st.markdown("If your dataset contains spatial coordinates, you can map synchrony hotspots.")
    
    if 'df' in st.session_state:
        df_geo = st.session_state['df']
        has_geo = 'geometry' in df_geo.columns or ('lat' in df_geo.columns and 'lon' in df_geo.columns)
        
        if has_geo:
            st.success("Spatial coordinates detected.")
            
            k = st.slider("KNN Neighbors", 1, 10, 4)
            window = st.slider("Spatial Time Window", 3, 50, 13)
            
            if st.button("Compute Spatial Hotspots"):
                with st.spinner("Computing Spatial Systemic Tau..."):
                    try:
                        from systemictau.spatial import spatial_tau
                        # Select temporal columns dynamically (dummy logic assumes non-geometry numeric cols)
                        numeric_cols = df_geo.select_dtypes(include=[np.number]).columns.tolist()
                        
                        res_gdf = spatial_tau(df_geo, value_cols=numeric_cols, geometry_col='geometry' if 'geometry' in df_geo.columns else 'lat', k_neighbors=k, window_size=window)
                        st.session_state['res_gdf'] = res_gdf
                        st.success("Spatial computation complete!")
                    except Exception as e:
                        st.error(f"GIS Error: {e}")
            
            if 'res_gdf' in st.session_state and HAS_FOLIUM:
                st.subheader("Hotspot Map")
                # Simple interactive map using Folium
                m = folium.Map(location=[0, 0], zoom_start=2)
                
                # Check if it's points
                for idx, row in st.session_state['res_gdf'].iterrows():
                    color = 'red' if row.get('hotspot_flag', 0) == 1 else 'blue'
                    # Simplified coordinate extraction depending on if it's Point geometries
                    try:
                        lat, lon = row.geometry.y, row.geometry.x
                        folium.CircleMarker(
                            location=[lat, lon],
                            radius=row['spatial_tau'] * 10,
                            color=color,
                            fill=True,
                            popup=f"Tau: {row['spatial_tau']:.2f}"
                        ).add_to(m)
                    except Exception:
                        pass
                st_folium(m, width=700, height=500)
                
        else:
            st.warning("No spatial coordinates ('geometry' or 'lat'/'lon') found in the dataset.")
    else:
        st.info("Please load data in the first tab.")

with tab5:
    st.header("Autonomous AI & Knowledge Graph")
    st.markdown("Query the Neo4j graph for historical ontological ascents and LLM reports.")
    
    tenant_id = st.text_input("Tenant ID", value="default")
    if st.button("Fetch Graph History"):
        try:
            if 'token' not in st.session_state:
                st.warning("Please login first")
            else:
                headers = {"Authorization": f"Bearer {st.session_state['token']}"}
                response = requests.get(f"{API_URL}/graph/tenant/{tenant_id}/epistemic", headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    st.success("Epistemic Graph Context Retrieved")
                    for item in data.get("history", []):
                        with st.expander(f"Transition at t* = {item.get('t_star')} (Tau={item.get('tau')})"):
                            st.markdown(f"**Event:** {item.get('description')}")
                            if item.get("claim"):
                                st.markdown(f"**Hypothesis:** {item.get('claim')} (Conf: {item.get('confidence')})")
                                if item.get("source"):
                                    st.markdown(f"**Evidence:** {item.get('summary')} (Source: {item.get('source')})")
                else:
                    st.error(f"API Error: {response.text}")
        except Exception as e:
            st.error(f"Connection failed: {e}")

with tab6:
    st.header("Live WebSocket Feed")
    st.markdown("Stream real-time anomalies directly from Kafka.")
    if st.button("Poll Latest Stream Events"):
        try:
            from websockets.sync.client import connect
            ws_url = API_URL.replace("http://", "ws://").replace("https://", "wss://") + "/ws/stream"
            with connect(ws_url) as websocket:
                # Wait briefly for up to 3 messages
                websocket.timeout = 2.0
                st.success("Connected to live stream. Listening...")
                for i in range(3):
                    try:
                        message = websocket.recv(timeout=2.0)
                        st.info(f"New Transition Detected: {message}")
                    except TimeoutError:
                        break
                        
        except Exception as e:
            st.error(f"WebSocket Error: {e}")
