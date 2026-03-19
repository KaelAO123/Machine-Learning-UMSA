import numpy as np
import pandas as pd  # <-- NUEVO: para cargar tu CSV
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score, r2_score, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# %% 1. Cargar TU dataset (NO el de cáncer de mama)
df = pd.read_csv('datos_completos_limpios.csv')

# %% 2. Preparar datos para clasificación (ejemplo: predecir si la brecha es alta o baja)
# Crear una variable objetivo categórica basada en la Brecha
# Por ejemplo: 'Brecha_alta' si la brecha > mediana
mediana_brecha = df['Brecha'].median()
df['Brecha_categoria'] = (df['Brecha'] > mediana_brecha).astype(int)

# Seleccionar características (X) y variable objetivo (y)
# NOTA: Excluimos columnas que no deben ser entrada
feature_columns = ['DEPARTAMENTO', 'PROVINCIA', 'AÑO', 
                   'Población_Total', 'Hombres', 'Mujeres', 
                   'Tasa_Hombres', 'Tasa_Mujeres']

X = df[feature_columns].copy()
y = df['Brecha_categoria'].copy()  # <-- Variable objetivo binaria

# %% 3. Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y  # stratify mantiene proporciones
)

# %% 4. Preprocesamiento (¡LA PARTE CLAVE!)
# Identificar columnas numéricas y categóricas
numeric_features = ['AÑO', 'Población_Total', 'Hombres', 'Mujeres', 
                    'Tasa_Hombres', 'Tasa_Mujeres']
categorical_features = ['DEPARTAMENTO', 'PROVINCIA']

# Crear transformadores
numeric_transformer = StandardScaler()  # <-- IGUAL que en el código original
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# Combinar en un preprocesador (Pipeline de sklearn)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Aplicar preprocesamiento
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(f"Dimensiones de entrada después de procesar: {X_train_processed.shape[1]}")

# %% 5. Crear modelo (¡CASI IGUAL al original!)
model = Sequential()
model.add(Dense(32, input_dim=X_train_processed.shape[1], activation='relu'))
model.add(Dropout(0.3))  # Regularización
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))  # <-- IGUAL: salida binaria

# Compilar (IGUAL)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

# %% 6. Entrenar con Early Stopping (MEJORA)
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

history = model.fit(
    X_train_processed, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# %% 7. Evaluar (MUY SIMILAR)
y_pred_prob = model.predict(X_test_processed).ravel()
y_pred = (y_pred_prob > 0.5).astype(int)

print("\n" + "="*50)
print("RESULTADOS EN TEST")
print("="*50)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=['Brecha Baja', 'Brecha Alta']))

# %% 8. Graficar (IGUAL)
plt.figure(figsize=(14,5))

plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Pérdida entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida validación')
plt.title('Evolución de la pérdida')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Accuracy entrenamiento')
plt.plot(history.history['val_accuracy'], label='Accuracy validación')
plt.title('Evolución del accuracy')
plt.xlabel('Época')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()