import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from io import BytesIO

# ==============================================================================
# Modelado de E(Teorema24v) sobre datos generados
# ==============================================================================

def E_teorema24v(k):
    """
    Esperanza teórica de |tau_s| para un mapa cuadrático unimodal (Feigenbaum)
    en la órbita superestable de período 2^k.
    """
    return (2**(k - 1)) / (2**k - 1)

def map_tau_to_dyadic_k(tau_val):
    """
    Inversa del Teorema 24v: Dado un valor empírico de |tau|, 
    estima el nivel 'k' de la cascada de bifurcación diádica.
    |tau| = (2^(k-1)) / (2^k - 1)
    
    Resolviendo numéricamente para valores de tau entre (0.5, 1.0].
    Valores <= 0.50 implican k -> infinito (Caos).
    """
    # Usamos valor absoluto y tomamos límite en 0.5 para la zona caótica (Acumulación Feigenbaum)
    t = np.abs(tau_val)
    if pd.isna(t):
        return np.nan
        
    if t <= 0.500001:
        # Caos absoluto / Feigenbaum attractor
        return float('inf')
        
    if t >= 0.999:
        # Sistema totalmente sincronizado (Periodo 2, k=1)
        return 1.0
        
    # k = log2( tau / (2*tau - 1) )
    num = t
    den = 2 * t - 1
    
    # Prevenir divisiones por cero o logaritmos negativos (ya atajados por if t <= 0.5)
    return np.log2(num / den)

def process_evidence_with_theorem(input_path, output_path):
    print(f"Cargando {input_path}...")
    df_tau = pd.read_excel(input_path, sheet_name='Tau_Panel')
    
    print("Aplicando mapeo diádico (k-level) según Teorema 24v...")
    # Calculamos k_level para cada punto de tiempo en cada simulación
    # Usamos el valor absoluto de tau_s
    df_tau['dyadic_k_level'] = df_tau['tau_s'].apply(map_tau_to_dyadic_k)
    
    # También calculamos el periodo estimado de la órbita: P = 2^k
    df_tau['estimated_period'] = 2 ** df_tau['dyadic_k_level']
    
    # Guardamos los resultados
    print(f"Exportando resultados a {output_path}...")
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_tau.to_excel(writer, sheet_name='Dyadic_Tree_Analysis', index=False)
        
    print("¡Análisis completado!")
    
if __name__ == "__main__":
    input_file = "Systemic_Tau_Dyadic_Tree_Evidence.xlsx"
    output_file = "Systemic_Tau_Dyadic_Tree_Analysis.xlsx"
    process_evidence_with_theorem(input_file, output_file)
