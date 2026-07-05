import pandas as pd

print("Cargando datos crudos (puede tardar unos segundos)...")
# Cargar la pestaña Raw_Panel
df_raw = pd.read_excel('Systemic_Tau_Dyadic_Tree_Evidence.xlsx', sheet_name='Raw_Panel')

print("Extrayendo la Simulación 0...")
# Extraer solo la primera simulación para analizarla en la app
df_sim0 = df_raw[df_raw['simulation_id'] == 0].copy()

print("Pivoteando a formato ancho (multivariante)...")
# Pivotear la tabla: filas = time, columnas = component, valores = value
df_wide = df_sim0.pivot(index='time', columns='component', values='value').reset_index()

# Guardar en un CSV limpio y listo para la app
output_file = 'Evidencia_App_Ready.csv'
df_wide.to_csv(output_file, index=False)

print(f"✅ ¡Archivo {output_file} generado con éxito!")
print("Ya puedes subirlo a la pestaña '1. Data Hub' de la aplicación.")
