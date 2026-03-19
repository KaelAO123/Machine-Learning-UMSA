import pandas as pd
import numpy as np

# --- CONFIGURACIÓN ---
# ¡¡¡MUY IMPORTANTE!!! Ajusta estas variables según tu archivo.
ARCHIVO_EXCEL = 'MIDATASET/3020501.xlsx'  # Nombre de tu archivo
HOJA_EXCEL = '3020501'          # Nombre de la hoja (o usa 0 para la primera)

# --- 1. CARGAR EL ARCHIVO CRUDO ---
# Leemos el archivo sin asumir encabezados, para poder ver toda la estructura.
df_raw = pd.read_excel(ARCHIVO_EXCEL, sheet_name=HOJA_EXCEL, header=None)

print("--- Vista previa de los datos crudos (5 primeras filas) ---")
print(df_raw.head(10).to_string())
print("\n" + "="*80 + "\n")

# --- 2. IDENTIFICAR Y EXTRAER EL ENCABEZADO REAL ---
# El encabezado real está en la fila 5 (índice 5 en base 0, porque la primera fila es índice 0).
# Observando el archivo:
# - Fila 0-3: Metadatos (título, etc.)
# - Fila 4: Nombres de las columnas principales (N°, NIVEL, DEPARTAMENTO, ...)
# - Fila 5: Sub-encabezados para los años (Población, Tasa de alfabetismo, etc.)
# Necesitamos combinar la fila 4 y 5 para tener un encabezado completo.

# Extraemos las dos filas que nos servirán de encabezado
header_fila4 = df_raw.iloc[4, :].fillna('').astype(str).tolist()
header_fila5 = df_raw.iloc[5, :].fillna('').astype(str).tolist()

# Las combinamos. Si el valor de la fila5 está vacío, usamos el de la fila4.
# También limpiamos saltos de línea que puedan causar problemas.
header_combinado = []
for col4, col5 in zip(header_fila4, header_fila5):
    nombre_col = col5.strip() if col5.strip() else col4.strip()
    # Limpiamos caracteres problemáticos para nombres de columna
    nombre_col = (nombre_col.replace('\n', '_')
                             .replace(' ', '_')
                             .replace('<', '')
                             .replace('>', '')
                             .replace('-', '_')
                             .replace('___', '_')
                             .replace('__', '_')
                             .strip('_'))
    header_combinado.append(nombre_col)

print("--- Encabezado generado ---")
print(header_combinado[:10], "...")  # Solo mostramos las primeras 10
print("\n" + "="*80 + "\n")

# --- 3. CARGAR LOS DATOS CON EL NUEVO ENCABEZADO ---
# Cargamos el archivo de nuevo, saltando las primeras 6 filas (0 a 5) y usando el encabezado que creamos.
df = pd.read_excel(ARCHIVO_EXCEL, sheet_name=HOJA_EXCEL, skiprows=6, header=None, names=header_combinado)

print("--- Datos cargados con el nuevo encabezado ---")
print(df.head())
print(f"Dimensiones del dataframe: {df.shape}")

# --- 4. ELIMINAR FILAS Y COLUMBAS VACÍAS ---
# Las primeras columnas de metadatos (N°, NIVEL) pueden tener NaN después de la fila que nos interesa.
# Las columnas de años y tasas también pueden tener NaN al final. Las eliminamos.
df = df.dropna(how='all')          # Elimina filas completamente vacías
df = df.dropna(axis=1, how='all')  # Elimina columnas completamente vacías

# También eliminamos filas donde la columna 'NIVEL' sea NaN, ya que esas son las filas de totales
# que tienen la jerarquía completa (como "Departamento", "Provincia", etc.) y las necesitamos.
# Las filas con NaN en 'NIVEL' son probablemente filas de separación o metadatos.
df = df.dropna(subset=['NIVEL'])

print("\n--- Después de eliminar vacíos ---")
print(df.head())
print(f"Dimensiones del dataframe: {df.shape}")

# --- 5. RELLENAR VALORES DE JERARQUÍA HACIA ADELANTE (VERSIÓN CORREGIDA) ---
# Este es un paso CRUCIAL. Observa que en las columnas DEPARTAMENTO, PROVINCIA, MUNICIPIO/TIOC,
# los valores solo aparecen en la primera fila de cada grupo y luego están vacíos.
# Usamos 'ffill()' (forward fill) para llenar esos vacíos con el último valor válido.

cols_jerarquia = ['DEPARTAMENTO', 'PROVINCIA', 'MUNICIPIO/TIOC']
# Primero, aseguramos que las columnas existan
cols_jerarquia = [col for col in cols_jerarquia if col in df.columns]

for col in cols_jerarquia:
    # Reemplazamos los '.' o '-' que puedan haber por NaN antes de hacer el ffill
    df[col] = df[col].replace(['.', '-', ''], np.nan)
    # Usamos ffill() en lugar de fillna(method='ffill')
    df[col] = df[col].ffill()

print("\n--- Después de rellenar jerarquías ---")
print(df[cols_jerarquia].head(15).to_string())

# --- 6. FILTRAR Y RENOMBRAR PARA EL FORMATO LARGO ---
# Ahora viene la magia. Vamos a pasar de un formato "ancho" (wide) donde cada año tiene varias columnas,
# a un formato "largo" (long) donde una columna 'Año' y una columna 'Variable' identifican cada valor.

# Primero, identificamos las columnas fijas
columnas_fijas = ['N°', 'NIVEL', 'DEPARTAMENTO', 'PROVINCIA', 'MUNICIPIO/TIOC', 'ÁREA_Y_GRUPO_DE_EDAD']
# Nos aseguramos de que solo existan en el dataframe
columnas_fijas = [col for col in columnas_fijas if col in df.columns]

# Las columnas que contienen datos de los años (2001, 2012, 2024) son todas las demás
columnas_variables = [col for col in df.columns if col not in columnas_fijas]

print(f"\nColumnas fijas: {columnas_fijas}")
print(f"Columnas variables ({len(columnas_variables)}): {columnas_variables[:5]}...")

# --- 7. CREAR EL FORMATO LARGO MANUALMENTE (ALTERNATIVA A WIDE_TO_LONG) ---
# Dado que los nombres de columna son complejos, vamos a hacer la transformación de forma manual
# que es más robusta.

# Lista para almacenar las filas del nuevo dataframe
rows_list = []

for idx, row in df.iterrows():
    # Extraemos las columnas fijas una sola vez por fila original
    fixed_data = {col: row[col] for col in columnas_fijas}
    
    # Procesamos las columnas de los años (2001, 2012, 2024)
    for col in columnas_variables:
        if pd.notna(row[col]):  # Solo procesamos si hay valor
            # Intentamos extraer el año del nombre de la columna
            # Los nombres tienen formato como '2001', '2001_1', '2001_2', etc.
            # Tomamos los primeros 4 caracteres como año
            año_str = col[:4] if len(col) >= 4 and col[:4].isdigit() else None
            
            if año_str:
                año = int(año_str)
                # El resto del nombre después del año es el tipo de variable
                tipo_variable = col[5:] if len(col) > 5 else 'dato'
                
                # Creamos una nueva fila
                new_row = fixed_data.copy()
                new_row['AÑO'] = año
                new_row['VARIABLE'] = tipo_variable
                new_row['VALOR'] = row[col]
                rows_list.append(new_row)

# Creamos el nuevo dataframe en formato largo
df_long = pd.DataFrame(rows_list)

print("\n--- Dataframe en formato largo ---")
print(df_long.head(10))
print(f"Dimensiones del dataframe largo: {df_long.shape}")

# --- 8. APLICAR PIVOT PARA TENER VARIABLES COMO COLUMNAS ---
# Ahora tenemos un formato súper-largo con una columna 'VARIABLE' y 'VALOR'.
# Vamos a pivotear para que cada variable sea una columna (Población, Hombres, Mujeres, Tasa, etc.)

# Primero, identificamos los tipos de variables únicos
print("\nTipos de variables encontrados:")
print(df_long['VARIABLE'].value_counts().head(15))

# Hacemos el pivot
df_pivot = df_long.pivot_table(
    index=[col for col in columnas_fijas if col in df_long.columns] + ['AÑO'],
    columns='VARIABLE',
    values='VALOR',
    aggfunc='first'  # En caso de duplicados, tomamos el primero
).reset_index()

print("\n--- Dataframe después de pivotear ---")
print(df_pivot.head())
print(f"Dimensiones del dataframe pivoteado: {df_pivot.shape}")

# --- 9. LIMPIEZA FINAL Y GUARDADO ---
# Reemplazamos los nombres de columna problemáticos
df_pivot.columns = [str(col).replace(' ', '_').replace('-', '_').replace('.', '_') for col in df_pivot.columns]

# Guardamos el resultado a un archivo CSV limpio.
df_pivot.to_csv('datos_limpios_formato_analisis.csv', index=False, encoding='utf-8-sig')
print("\n✅ Archivo 'datos_limpios_formato_analisis.csv' guardado con éxito.")

# También guardamos una versión más simple con las columnas principales
columnas_principales = ['DEPARTAMENTO', 'PROVINCIA', 'MUNICIPIO/TIOC', 'ÁREA_Y_GRUPO_DE_EDAD', 'AÑO']
# Añadimos las columnas de población y tasas si existen
for col in ['Población', 'Hombres', 'Mujeres', 'Tasa_de_alfabetismo']:
    if col in df_pivot.columns:
        columnas_principales.append(col)

df_simple = df_pivot[[col for col in columnas_principales if col in df_pivot.columns]].copy()
df_simple.to_csv('datos_limpios_simple.csv', index=False, encoding='utf-8-sig')
print("✅ Archivo 'datos_limpios_simple.csv' guardado con éxito.")

print(f"\n📊 Estadísticas finales:")
print(f"   - Total de filas: {len(df_pivot)}")
print(f"   - Años disponibles: {sorted(df_pivot['AÑO'].unique())}")
print(f"   - Departamentos: {df_pivot['DEPARTAMENTO'].nunique()}")
print(f"   - Variables disponibles: {len([c for c in df_pivot.columns if c not in columnas_fijas and c != 'AÑO'])}")