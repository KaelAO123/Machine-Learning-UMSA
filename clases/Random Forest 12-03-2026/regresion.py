# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math

# %%
# Cargar dataset
df = pd.read_csv("algoK.csv")

print(df.head())

# %%
# Crear variable objetivo: 1 si tiene fibra
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
# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# %%
# Crear modelo Random Forest Regressor
rf_regressor = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# %%
# Entrenar modelo
rf_regressor.fit(X_train, y_train)

# %%
# Predicciones
y_pred = rf_regressor.predict(X_test)

# %%
# Ejemplo de predicción para una nueva localidad
ejemplo = X.iloc[[0]]
prediccion = rf_regressor.predict(ejemplo)

print("Probabilidad estimada de fibra:", prediccion[0])

# %%
# Métricas de regresión
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R²: {r2}")

# %%
# Importancia de variables
importances = rf_regressor.feature_importances_

plt.barh(X.columns, importances)
plt.xlabel("Importancia")
plt.title("Importancia de variables - Random Forest Regressor")
plt.show()