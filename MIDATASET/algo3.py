import pandas as pd
import numpy as np
import re

# --- CONFIGURACIÓN ---
ARCHIVO_EXCEL = 'MIDATASET/3020501.xlsx'  # Nombre de tu archivo
HOJA_EXCEL = '3020501'

print("="*80)
print("INICIANDO PROCESAMIENTO DE DATOS")
print("="*80)

# --- 1. CARGAR EL ARCHIVO CRUDO ---
print("\n📂 Cargando archivo...")
df_raw = pd.read_excel(ARCHIVO_EXCEL, sheet_name=HOJA_EXCEL, header=None)

print(f"   - Dimensiones del archivo crudo: {df_raw.shape}")

# --- 2. EXTRAER ENCABEZADO ---
print("\n📝 Extrayendo encabezado...")

header_fila4 = df_raw.iloc[4, :].fillna('').astype(str).tolist()
header_fila5 = df_raw.iloc[5, :].fillna('').astype(str).tolist()

# Combinamos los encabezados
header_combinado = []
for i, (col4, col5) in enumerate(zip(header_fila4, header_fila5)):
    if col5.strip() and col5.strip() not in ['nan', '']:
        # Si hay subencabezado, lo usamos como parte del nombre
        if col4.strip():
            nombre = f"{col4}_{col5}".strip()
        else:
            nombre = col5.strip()
    else:
        nombre = col4.strip() if col4.strip() else f"col_{i}"
    
    # Limpiamos el nombre
    nombre = (nombre.replace('\n', '_')
                     .replace(' ', '_')
                     .replace('-', '_')
                     .replace('.', '_')
                     .replace('<', '')
                     .replace('>', '')
                     .replace('___', '_')
                     .replace('__', '_')
                     .strip('_'))
    
    header_combinado.append(nombre)

print(f"   - Encabezado combinado creado")

# --- 3. CARGAR DATOS CON NUEVO ENCABEZADO ---
print("\n📊 Cargando datos con nuevo encabezado...")
df = pd.read_excel(ARCHIVO_EXCEL, sheet_name=HOJA_EXCEL, skiprows=6, header=None, names=header_combinado)

print(f"   - Dimensiones iniciales: {df.shape}")

# --- 4. LIMPIEZA BÁSICA ---
print("\n🧹 Limpiando datos...")

df = df.dropna(how='all')
df = df.dropna(axis=1, how='all')
df = df.dropna(subset=['NIVEL'])

print(f"   - Dimensiones después de limpieza: {df.shape}")

# --- 5. RELLENAR JERARQUÍAS ---
print("\n🏷️ Rellenando jerarquías geográficas...")

cols_jerarquia = ['DEPARTAMENTO', 'PROVINCIA', 'MUNICIPIO/TIOC']
cols_jerarquia = [col for col in cols_jerarquia if col in df.columns]

for col in cols_jerarquia:
    df[col] = df[col].replace(['.', '-', '', 'nan'], np.nan)
    df[col] = df[col].ffill()

print(f"   - Jerarquías procesadas: {cols_jerarquia}")

# --- 6. DEFINIR MAPEO DE COLUMNAS POR AÑO ---
print("\n🔍 Definiendo mapeo de columnas...")

# Este es el mapeo CRUCIAL basado en tu output
mapeo_por_año = {
    2001: {
        '2001_Población': 'Población_Total',
        'col_8': 'Hombres',
        'col_9': 'Mujeres',
        'Tasa_de_alfabetismo': 'Tasa_Total',
        'col_11': 'Tasa_Hombres',
        'col_12': 'Tasa_Mujeres',
        'Brecha_Hombres_Mujeres': 'Brecha'
    },
    2012: {
        '2012_Población': 'Población_Total',
        'col_15': 'Hombres',
        'col_16': 'Mujeres',
        'Tasa_de_alfabetismo.1': 'Tasa_Total',
        'col_18': 'Tasa_Hombres',
        'col_19': 'Tasa_Mujeres',
        'Brecha_Hombre_Mujer': 'Brecha'
    }
}

# Para 2024, asumimos un patrón similar basado en el orden
# Buscamos columnas que empiecen con '2024' o tengan un patrón similar
columnas_2024 = {}
for i, col in enumerate(df.columns):
    if '2024' in col or (i > df.columns.get_loc('Brecha_Hombre_Mujer') and col not in mapeo_por_año[2001].keys() and col not in mapeo_por_año[2012].keys()):
        # Asignamos según el orden relativo
        if '2024_Población' not in columnas_2024 and ('2024' in col or 'Población' in col):
            columnas_2024[col] = 'Población_Total'
        elif len(columnas_2024) == 1:
            columnas_2024[col] = 'Hombres'
        elif len(columnas_2024) == 2:
            columnas_2024[col] = 'Mujeres'
        elif len(columnas_2024) == 3:
            columnas_2024[col] = 'Tasa_Total'
        elif len(columnas_2024) == 4:
            columnas_2024[col] = 'Tasa_Hombres'
        elif len(columnas_2024) == 5:
            columnas_2024[col] = 'Tasa_Mujeres'
        elif len(columnas_2024) == 6:
            columnas_2024[col] = 'Brecha'

if columnas_2024:
    mapeo_por_año[2024] = columnas_2024

print(f"   - Mapeo creado para años: {list(mapeo_por_año.keys())}")

# --- 7. CREAR FORMATO LARGO ---
print("\n🔄 Creando formato largo...")

columnas_fijas = ['N°', 'NIVEL', 'DEPARTAMENTO', 'PROVINCIA', 'MUNICIPIO/TIOC', 'ÁREA_Y_GRUPO_DE_EDAD']
columnas_fijas = [col for col in columnas_fijas if col in df.columns]

rows_list = []
filas_procesadas = 0

for idx, row in df.iterrows():
    filas_procesadas += 1
    if filas_procesadas % 500 == 0:
        print(f"      Procesadas {filas_procesadas} filas...")
    
    # Datos fijos de esta fila
    fixed_data = {col: row[col] for col in columnas_fijas}
    
    # Procesar cada año
    for año, mapeo in mapeo_por_año.items():
        año_tiene_datos = False
        año_data = fixed_data.copy()
        año_data['AÑO'] = año
        
        # Procesar cada variable del año
        for col_origen, var_destino in mapeo.items():
            if col_origen in df.columns and pd.notna(row[col_origen]):
                año_data[var_destino] = row[col_origen]
                año_tiene_datos = True
        
        # Solo agregar si el año tiene al menos un dato
        if año_tiene_datos:
            rows_list.append(año_data)

print(f"   - Total filas procesadas: {filas_procesadas}")
print(f"   - Filas creadas en formato largo: {len(rows_list)}")

# Crear dataframe largo
if rows_list:
    df_long = pd.DataFrame(rows_list)
    print(f"   - Dimensiones df_long: {df_long.shape}")
    print("\n   - Primeras 5 filas de df_long:")
    print(df_long.head())
else:
    print("❌ ERROR: No se crearon filas en formato largo")
    print("\n   Columnas disponibles en df:")
    for i, col in enumerate(df.columns):
        print(f"   {i}: {col}")
    exit()

# --- 8. ORDENAR Y LIMPIAR ---
print("\n✨ Limpiando y ordenando datos...")

# Reemplazar valores que puedan ser strings vacíos
for col in df_long.columns:
    if col not in columnas_fijas and col != 'AÑO':
        df_long[col] = pd.to_numeric(df_long[col], errors='coerce')

# Eliminar filas donde todas las variables numéricas son NaN
vars_numericas = [col for col in df_long.columns if col not in columnas_fijas + ['AÑO']]
df_long = df_long.dropna(subset=vars_numericas, how='all')

print(f"   - Dimensiones después de limpieza: {df_long.shape}")

# --- 9. GUARDAR ARCHIVOS ---
print("\n💾 Guardando archivos...")

# Versión completa
df_long.to_csv('datos_completos_limpios.csv', index=False, encoding='utf-8-sig')
print("   ✅ 'datos_completos_limpios.csv' guardado")

# Versión simplificada con las columnas más importantes
columnas_importantes = ['DEPARTAMENTO', 'PROVINCIA', 'MUNICIPIO/TIOC', 'ÁREA_Y_GRUPO_DE_EDAD', 'AÑO']
vars_importantes = ['Población_Total', 'Hombres', 'Mujeres', 'Tasa_Total', 'Brecha']
for var in vars_importantes:
    if var in df_long.columns:
        columnas_importantes.append(var)

df_simple = df_long[[col for col in columnas_importantes if col in df_long.columns]].copy()
df_simple.to_csv('datos_simplificados.csv', index=False, encoding='utf-8-sig')
print("   ✅ 'datos_simplificados.csv' guardado")

# --- 10. ESTADÍSTICAS FINALES ---
print("\n📊 ESTADÍSTICAS FINALES:")
print("="*50)
print(f"Total de registros: {len(df_long):,}")
print(f"Años disponibles: {sorted(df_long['AÑO'].unique())}")
print(f"Departamentos: {df_long['DEPARTAMENTO'].nunique()}")
if 'PROVINCIA' in df_long.columns:
    print(f"Provincias: {df_long['PROVINCIA'].nunique()}")
if 'MUNICIPIO/TIOC' in df_long.columns:
    print(f"Municipios/TIOCs: {df_long['MUNICIPIO/TIOC'].nunique()}")
print(f"Variables disponibles: {len([c for c in df_long.columns if c not in columnas_fijas and c != 'AÑO'])}")
print("\n   Variables encontradas:")
for var in [c for c in df_long.columns if c not in columnas_fijas and c != 'AÑO']:
    print(f"   - {var}")

print("\n✅ PROCESO COMPLETADO CON ÉXITO")
print("="*50)