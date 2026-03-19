import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# %% 1. Cargar datos
df = pd.read_csv('datos_completos_limpios.csv')

# %% 2. Crear variable objetivo categórica
# Ejemplo: Brecha ALTA si es mayor que la mediana
mediana_brecha = df['Brecha'].median()
df['Brecha_categoria'] = (df['Brecha'] > mediana_brecha).astype(int)
print(f"Clases: 0 = Brecha baja, 1 = Brecha alta")
print(f"Distribución:\n{df['Brecha_categoria'].value_counts()}")

# %% 3. Definir X y y
feature_columns = ['DEPARTAMENTO', 'PROVINCIA', 'AÑO', 
                   'Población_Total', 'Hombres', 'Mujeres', 
                   'Tasa_Hombres', 'Tasa_Mujeres']

X = df[feature_columns].copy()
y = df['Brecha_categoria'].copy()

# %% 4. Dividir
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# %% 5. Preprocesamiento
numeric_features = ['AÑO', 'Población_Total', 'Hombres', 'Mujeres', 
                    'Tasa_Hombres', 'Tasa_Mujeres']
categorical_features = ['DEPARTAMENTO', 'PROVINCIA']

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# %% 6. Pipeline con SVM para clasificación
svm_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('svm', SVC(kernel='rbf', C=100, gamma='scale', probability=True))
])

# Entrenar
print("Entrenando SVM...")
svm_pipeline.fit(X_train, y_train)

# Predecir
y_pred = svm_pipeline.predict(X_test)
y_pred_proba = svm_pipeline.predict_proba(X_test)[:, 1]

# %% 7. Evaluar
print("\n" + "="*50)
print("RESULTADOS SVM CLASIFICACIÓN")
print("="*50)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=['Brecha Baja', 'Brecha Alta']))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Baja', 'Alta'], yticklabels=['Baja', 'Alta'])
plt.xlabel('Predicho')
plt.ylabel('Real')
plt.title('Matriz de Confusión - SVM')
plt.show()