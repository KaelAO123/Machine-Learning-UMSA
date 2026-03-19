# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# %%
# Cargar dataset
df = pd.read_csv("algoK.csv")

print(df.head())

# %%
# Crear variable objetivo (1 si tiene fibra óptica)
df["Fibra"] = df["Observaciones"].apply(
    lambda x: 1 if isinstance(x, str) and "Fibra" in x else 0
)

# %%
# Variables de entrada
X = df[["Departamento", "Provincia", "Municipio"]]
y = df["Fibra"]

# %%
# Convertir texto a números
le = LabelEncoder()

for col in X.columns:
    X[col] = le.fit_transform(X[col].astype(str))

# %%
# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# %%
# Crear modelo Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# %%
# Entrenar modelo
rf.fit(X_train, y_train)

# %%
# Predicciones
y_pred = rf.predict(X_test)

# %%
# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Exactitud: {accuracy:.2f}")

# %%
# Informe de clasificación
print("Informe de clasificación:\n", classification_report(y_test, y_pred))

# %%
# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
print("Matriz de confusión:\n", cm)

# %%
# Graficar matriz de confusión
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.title("Matriz de Confusión - Random Forest")
plt.show()

# %%
# Importancia de variables
importances = rf.feature_importances_

plt.barh(X.columns, importances)
plt.xlabel("Importancia")
plt.title("Importancia de variables en Random Forest")
plt.show()