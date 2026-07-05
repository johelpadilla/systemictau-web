"""
Systemic Tau - Generador de Evidencia para el Árbol Diádico (Teorema v24)
==============================================================================
Configuración óptima:
- 200 simulaciones
- N = 10,000 por simulación
- 10 componentes
- Transición gradual con variabilidad
- Fases etiquetadas + variable coupling_strength
- Métricas completas (τs + hp_z + Joint Episodes)
"""

import numpy as np
import pandas as pd
from numba import njit
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURACIÓN (Optimizada para <20MB)
# ============================================================
N_SIMULATIONS = 50
N = 1000
N_COMPONENTS = 10
WINDOW_SIZE = 13
THETA_A = 0.05
D_MIN = 10

np.random.seed(42)

# ============================================================
# FUNCIONES
# ============================================================

def generate_multivariate_series(n, n_comp, trans_start, trans_dur):
    """Genera datos con transición gradual de acoplamiento."""
    t = np.arange(n)
    coupling = np.zeros(n)
    coupling[:trans_start] = 0.15
    
    end = min(trans_start + trans_dur, n)
    coupling[trans_start:end] = np.linspace(0.15, 0.75, end - trans_start)
    coupling[end:] = 0.75
    
    base = np.random.randn(n)
    data = np.zeros((n, n_comp))
    
    for i in range(n_comp):
        data[:, i] = base * coupling + np.random.randn(n) * 0.8
    return data, coupling

@njit
def _kendall_tau_numba(x, y):
    n = len(x)
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            sign_x = 1 if x[i] < x[j] else (-1 if x[i] > x[j] else 0)
            sign_y = 1 if y[i] < y[j] else (-1 if y[i] > y[j] else 0)
            if sign_x * sign_y > 0:
                concordant += 1
            elif sign_x * sign_y < 0:
                discordant += 1
    total = n * (n - 1) / 2
    if total == 0:
        return 0.0
    return (concordant - discordant) / total

@njit
def compute_systemic_tau_rolling(data, window):
    n, n_comp = data.shape
    tau_series = np.full(n, np.nan)
    
    for t in range(window, n):
        w = data[t-window:t]
        
        sum_tau = 0.0
        count = 0
        for i in range(n_comp):
            for j in range(i+1, n_comp):
                sum_tau += _kendall_tau_numba(w[:, i], w[:, j])
                count += 1
        
        if count > 0:
            tau_series[t] = sum_tau / count
            
    return tau_series


def compute_hyper_persistence(tau_series, w=20):
    hp = np.full(len(tau_series), np.nan)
    for t in range(w, len(tau_series)):
        win = tau_series[t-w:t]
        std_val = np.nanstd(win)
        if std_val > 0:
            hp[t] = np.abs(np.nanmean(win)) / std_val
    return hp


def detect_joint_episodes(tau_series, theta=0.05, dmin=10):
    episodes = []
    above = tau_series > theta
    start = None
    for t in range(len(tau_series)):
        if above[t] and start is None:
            start = t
        elif not above[t] and start is not None:
            dur = t - start
            if dur >= dmin:
                episodes.append({
                    'start': start,
                    'end': t,
                    'duration': dur,
                    'mean_tau': float(np.nanmean(tau_series[start:t]))
                })
            start = None
    return episodes

def write_df_to_excel(writer, df, base_sheet_name, max_rows=1000000):
    """Escribe un DataFrame a Excel partiéndolo en varias hojas si excede max_rows."""
    n_rows = len(df)
    if n_rows == 0:
        return
    if n_rows <= max_rows:
        df.to_excel(writer, sheet_name=base_sheet_name, index=False)
        return
    
    # Chunking
    num_chunks = int(np.ceil(n_rows / max_rows))
    for i in range(num_chunks):
        start_idx = i * max_rows
        end_idx = min((i + 1) * max_rows, n_rows)
        chunk = df.iloc[start_idx:end_idx]
        sheet_name = f"{base_sheet_name}_{i+1}"
        chunk.to_excel(writer, sheet_name=sheet_name, index=False)


# ============================================================
# BUCLE PRINCIPAL
# ============================================================
all_tau = []
all_hp = []
all_raw = []
all_episodes = []
all_meta = []

print(f"Generando {N_SIMULATIONS} simulaciones...")

for sim in range(N_SIMULATIONS):
    if (sim + 1) % 10 == 0:
        print(f"Completadas {sim + 1}/{N_SIMULATIONS}...")
        
    trans_start = np.random.randint(350, 550)
    trans_dur = np.random.randint(80, 150)
    
    data, coupling = generate_multivariate_series(N, N_COMPONENTS, trans_start, trans_dur)
    tau_s = compute_systemic_tau_rolling(data, WINDOW_SIZE)
    hp_z = compute_hyper_persistence(tau_s)
    episodes = detect_joint_episodes(tau_s, THETA_A, D_MIN)
    
    phase = np.full(N, 'Fase_1_Low', dtype=object)
    phase[trans_start:trans_start+trans_dur] = 'Fase_2_Transition'
    phase[trans_start+trans_dur:] = 'Fase_3_High'
    
    all_tau.append(pd.DataFrame({
        'simulation_id': sim,
        'time': np.arange(N),
        'tau_s': tau_s,
        'coupling_strength': coupling,
        'phase': phase
    }))
    
    all_hp.append(pd.DataFrame({
        'simulation_id': sim,
        'time': np.arange(N),
        'hp_z': hp_z,
        'coupling_strength': coupling,
        'phase': phase
    }))
    
    # Raw Panel - sólo guardar componentes resumidos si se excede memoria? No, todo.
    for c in range(N_COMPONENTS):
        all_raw.append(pd.DataFrame({
            'simulation_id': sim,
            'time': np.arange(N),
            'component': f'Comp_{c}',
            'value': data[:, c],
            'coupling_strength': coupling,
            'phase': phase
        }))
    
    for ep in episodes:
        all_episodes.append({
            'simulation_id': sim,
            'start': ep['start'],
            'end': ep['end'],
            'duration': ep['duration'],
            'mean_tau': ep['mean_tau'],
            'transition_start': trans_start
        })
    
    all_meta.append({
        'simulation_id': sim,
        'transition_start': trans_start,
        'transition_duration': trans_dur,
        'n_episodes': len(episodes)
    })

# ============================================================
# EXPORTAR A EXCEL
# ============================================================
print("Concatenando DataFrames (esto tomará mucha RAM)...")
df_tau = pd.concat(all_tau, ignore_index=True)
df_hp = pd.concat(all_hp, ignore_index=True)
df_raw = pd.concat(all_raw, ignore_index=True)
df_ep = pd.DataFrame(all_episodes)
df_meta = pd.DataFrame(all_meta)

output_path = "Systemic_Tau_Dyadic_Tree_Evidence.xlsx"

print(f"Escribiendo a {output_path} (puede tardar varios minutos)...")
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    write_df_to_excel(writer, df_meta, 'Metadata')
    write_df_to_excel(writer, df_tau, 'Tau_Panel')
    write_df_to_excel(writer, df_hp, 'HyperPersistence_Panel')
    write_df_to_excel(writer, df_raw, 'Raw_Panel')
    if not df_ep.empty:
        write_df_to_excel(writer, df_ep, 'Joint_Episodes')

print(f"\n✅ Archivo generado: {output_path}")
print(f"Total simulaciones: {N_SIMULATIONS}")
