# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# %%
# Cargar dataset
df = pd.read_csv("algoK.csv")

print(df.head())

# %%
# Crear variable objetivo (1 si tiene fibra)
df["Fibra"] = df["Observaciones"].apply(
    lambda x: 1 if isinstance(x,str) and "Fibra" in x else 0
)

# %%
# Variables de entrada
X = df[["Departamento","Provincia","Municipio"]]
y = df["Fibra"]

# %%
# Convertir texto a números
le = LabelEncoder()

for col in X.columns:
    X[col] = le.fit_transform(X[col].astype(str))

# %%
# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# %%
# Definir modelos
models = {

    "Gradient Boosting": GradientBoostingClassifier(random_state=42),

    "AdaBoost": AdaBoostClassifier(random_state=42),

    "XGBoost": XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    ),

    "LightGBM": LGBMClassifier(random_state=42),

    "CatBoost": CatBoostClassifier(
        verbose=0,
        random_state=42
    )
}

# %%
results = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    results[name] = {
        "model": model,
        "accuracy": acc,
        "pred": y_pred
    }

    print(f"{name} Accuracy: {acc:.4f}")

# %%
# Gráfico comparativo
plt.figure(figsize=(8,5))

plt.bar(
    results.keys(),
    [results[m]["accuracy"] for m in results]
)

plt.title("Comparación de modelos Boosting")
plt.ylabel("Accuracy")
plt.ylim(0,1)

plt.show()

# %%
# Matrices de confusión

fig, axes = plt.subplots(2,3, figsize=(15,10))
axes = axes.flatten()

for i,(name,res) in enumerate(results.items()):

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        res["pred"],
        ax=axes[i],
        cmap="Blues"
    )

    axes[i].set_title(name)

# eliminar subplot vacío
for j in range(len(results), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()