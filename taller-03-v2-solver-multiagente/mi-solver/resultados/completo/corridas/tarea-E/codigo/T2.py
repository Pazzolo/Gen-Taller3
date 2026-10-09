# T2 — Selección de alpha por validación cruzada (solo conjunto de entrenamiento)
# Pipeline StandardScaler + Ridge, GridSearchCV sobre numpy.logspace(-3, 3, 13),
# KFold(n_splits=5, shuffle=True, random_state=7), scoring='r2'.
# El conjunto de prueba NO se usa en esta subtarea.

import json
import numpy as np

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# ----------------------------------------------------------------------
# Reproducción de la partición de T1 (el script es independiente)
# ----------------------------------------------------------------------
datos = load_diabetes()
X, y = datos.data, datos.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=7
)

n_train = int(X_train.shape[0])
n_test = int(X_test.shape[0])

# ----------------------------------------------------------------------
# Parte 2 — Pipeline + GridSearchCV (solo con X_train, y_train)
# ----------------------------------------------------------------------
pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("ridge", Ridge()),
    ]
)

alphas = np.logspace(-3, 3, 13)
param_grid = {"ridge__alpha": alphas}

cv = KFold(n_splits=5, shuffle=True, random_state=7)

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring="r2",
    n_jobs=-1,
)

# Se ajusta únicamente con el conjunto de entrenamiento
grid.fit(X_train, y_train)

alpha_elegido = float(grid.best_params_["ridge__alpha"])
r2_medio_cv = float(grid.best_score_)
indice_mejor = int(grid.best_index_)

mean_scores = [float(m) for m in grid.cv_results_["mean_test_score"]]
std_scores = [float(s) for s in grid.cv_results_["std_test_score"]]

# ----------------------------------------------------------------------
# Guardar todas las cifras en resultados.json (sin redondear)
# ----------------------------------------------------------------------
try:
    with open("resultados.json", "r") as f:
        resultados = json.load(f)
    if not isinstance(resultados, dict):
        resultados = {}
except Exception:
    resultados = {}

resultados.update(
    {
        "T2_alphas_evaluados": [float(a) for a in alphas],
        "T2_r2_medio_cv_por_alpha": mean_scores,
        "T2_r2_std_cv_por_alpha": std_scores,
        "T2_indice_alpha_elegido": indice_mejor,
        "T2_alpha_elegido": alpha_elegido,
        "T2_r2_medio_cv": r2_medio_cv,
        "T2_n_splits_cv": 5,
        "T2_cv_shuffle": True,
        "T2_cv_random_state": 7,
        "T2_scoring": "r2",
        "T2_n_train_usado": n_train,
        "T2_n_test_reservado": n_test,
    }
)

with open("resultados.json", "w") as f:
    json.dump(resultados, f, indent=2)

# ----------------------------------------------------------------------
# Salida en consola (cifras principales con 4 decimales)
# ----------------------------------------------------------------------
print("Parte 2 — Seleccion de alpha por validacion cruzada (solo train)")
print(f"Train usado: {n_train} ejemplos | Test reservado (no usado aqui): {n_test}")
print("Pipeline: StandardScaler + Ridge")
print("Grid: numpy.logspace(-3, 3, 13) | CV: KFold(5, shuffle=True, random_state=7) | scoring='r2'")
print("")
print("R2 medio de CV por alpha:")
for a, m, s in zip(alphas, mean_scores, std_scores):
    marca = "  <-- mejor" if float(a) == alpha_elegido else ""
    print(f"  alpha={float(a):.4f} | R2 CV medio={m:.4f} (+/- {s:.4f}){marca}")
print("")
print(f"Alpha elegido: {alpha_elegido:.4f}")
print(f"R2 medio de validacion cruzada: {r2_medio_cv:.4f}")
print("Cifras guardadas en resultados.json")
