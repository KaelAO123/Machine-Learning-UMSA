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
print(header_combinado)
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

# --- 5. RELLENAR VALORES DE JERARQUÍA HACIA ADELANTE ---
# Este es un paso CRUCIAL. Observa que en las columnas DEPARTAMENTO, PROVINCIA, MUNICIPIO/TIOC,
# los valores solo aparecen en la primera fila de cada grupo y luego están vacíos.
# Usamos 'ffill' (forward fill) para llenar esos vacíos con el último valor válido.

cols_jerarquia = ['DEPARTAMENTO', 'PROVINCIA', 'MUNICIPIO/TIOC']
for col in cols_jerarquia:
    # Reemplazamos los '.' o '-' que puedan haber por NaN antes de hacer el ffill
    df[col] = df[col].replace(['.', '-', ''], np.nan)
    df[col] = df[col].ffill()


print("\n--- Después de rellenar jerarquías ---")
print(df[cols_jerarquia].head(15).to_string())

# --- 6. FILTRAR Y RENOMBRAR PARA EL FORMATO LARGO ---
# Ahora viene la magia. Vamos a pasar de un formato "ancho" (wide) donde cada año tiene 7 columnas,
# a un formato "largo" (long) donde una columna 'Año' y una columna 'Variable' identifican cada valor.

# Primero, identificamos los bloques de columnas para cada año.
# Mirando el encabezado, el patrón parece ser: 2001, 2001.1, 2001.2, 2001.3, 2001.4, 2001.5, 2001.6 para el año 2001.
# Pero en nuestro encabezado generado, los nombres deberían ser más descriptivos.
# Hagamos una lista de las columnas que NO son de año (las fijas)
columnas_fijas = ['N°', 'NIVEL', 'DEPARTAMENTO', 'PROVINCIA', 'MUNICIPIO/TIOC', 'ÁREA_Y_GRUPO_DE_EDAD']

# El resto son las variables para los años 2001, 2012 y 2024.
# Asumimos que el orden es: 2001_Población, 2001_Hombres, 2001_Mujeres, 2001_Tasa_Total, 2001_Tasa_Hombres, 2001_Tasa_Mujeres, 2001_Brecha
# Y luego se repite para 2012 y 2024.

# Vamos a usar pd.wide_to_long, que es perfecto para esto, pero necesita que los nombres de columna sigan un patrón.
# Primero, vamos a limpiar los nombres de las columnas de año para que tengan un sufijo identificable.
# Ej: "2001_Población" -> "Población_2001" (wide_to_long usa el sufijo como el identificador de tiempo)

nuevos_nombres = {}
for col in df.columns:
    if col in columnas_fijas:
        nuevos_nombres[col] = col
    else:
        # Buscamos el año en el nombre de la columna. Esto es un poco frágil, pero asumimos que el año está al inicio.
        partes = col.split('_')
        if len(partes) >= 2:
            año = partes[0]
            variable = '_'.join(partes[1:])  # El resto del nombre es la variable
            # Formato: variable_año
            nuevos_nombres[col] = f"{variable}_{año}"

df = df.rename(columns=nuevos_nombres)

print("\n--- Columnas después de renombrar para wide_to_long ---")
print(df.columns.tolist())

# --- 7. APLICAR WIDE_TO_LONG ---
# Ahora sí, aplicamos la transformación.
# Las columnas que no son de año (stubnames) son los nombres de las variables sin el sufijo de año.
# En este caso, nuestras variables son: 'Población', 'Hombres', 'Mujeres', 'Tasa_de_alfabetismo_Total', 'Tasa_de_alfabetismo_Hombres', 'Tasa_de_alfabetismo_Mujeres', 'Brecha__Hombres_-_Mujeres'
# ¡Cuidado! Los nombres deben ser EXACTAMENTE iguales a la parte anterior al sufijo '_AÑO'.

# Extraemos los nombres de las variables únicas (stubnames) de las columnas renombradas.
stubnames = sorted(list(set([col.rsplit('_', 1)[0] for col in df.columns if '_20' in col and col not in columnas_fijas])))

print(f"\nStubnames identificados: {stubnames}")

# Usamos wide_to_long. El 'i' son las columnas de identificación fijas.
# El 'j' será el nombre de la nueva columna que contendrá los años.
df_long = pd.wide_to_long(df,
                          stubnames=stubnames,
                          i=columnas_fijas,
                          j='AÑO',
                          sep='_',
                          suffix='\d{4}')  # Busca sufijos que sean años de 4 dígitos

# Reseteamos el índice para que las columnas de identificación vuelvan a ser columnas normales.
df_long = df_long.reset_index()

print("\n--- Dataframe en formato largo ---")
print(df_long.head())
print(f"Dimensiones del dataframe largo: {df_long.shape}")

# --- 8. LIMPIEZA FINAL Y GUARDADO ---
# Ahora podemos hacer una limpieza final: eliminar filas con muchos NaN, etc.
df_final = df_long.dropna(subset=stubnames, how='all')  # Elimina filas donde todos los valores de las variables son NaN

# Podemos reordenar las columnas para que sea más legible
columnas_ordenadas = ['DEPARTAMENTO', 'PROVINCIA', 'MUNICIPIO/TIOC', 'ÁREA_Y_GRUPO_DE_EDAD', 'AÑO'] + stubnames
df_final = df_final[columnas_ordenadas]

# Convertimos la columna 'AÑO' a entero
df_final['AÑO'] = df_final['AÑO'].astype(int)

print("\n--- Vista final de los datos limpios ---")
print(df_final.head(10).to_string())
print(f"Dimensiones finales: {df_final.shape}")

# Guardamos el resultado a un archivo CSV limpio.
df_final.to_csv('datos_limpios_formato_largo.csv', index=False, encoding='utf-8-sig')
print("\n✅ Archivo 'datos_limpios_formato_largo.csv' guardado con éxito.")