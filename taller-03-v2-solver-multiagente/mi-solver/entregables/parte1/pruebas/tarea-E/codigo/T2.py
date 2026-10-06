# -*- coding: utf-8 -*-
"""
Subtarea T2 — Pipeline StandardScaler + Ridge con selección de alpha por
validación cruzada (GridSearchCV), usando SOLO el conjunto de entrenamiento.

Datos: sklearn.datasets.load_diabetes
División: train_test_split(test_size=0.25, random_state=7)  (misma que T1)
Grid:     13 valores de numpy.logspace(-3, 3, 13)
CV:       KFold(n_splits=5, shuffle=True, random_state=7), scoring='r2'
Figura:   no

Nota: en esta subtarea NO se predice sobre ningún conjunto. El único R2
reportado es el medio de validación cruzada calculado por GridSearchCV
dentro del conjunto de entrenamiento. El conjunto de prueba queda reservado.
"""

import json
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


# ----------------------------------------------------------------------
# 1) Datos y única división entrenamiento/prueba (idéntica a la T1)
# ----------------------------------------------------------------------
data = load_diabetes()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=7
)

# El conjunto de prueba NO se usa en esta subtarea (ni para ajustar el
# escalador, ni para elegir alpha). Solo se entrena con X_train / y_train.


# ----------------------------------------------------------------------
# 2) Pipeline StandardScaler + Ridge
# ----------------------------------------------------------------------
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge())
])

# 13 valores de alpha en escala logarítmica de 10^-3 a 10^3
alphas = np.logspace(-3, 3, 13)
param_grid = {'ridge__alpha': alphas}


# ----------------------------------------------------------------------
# 3) GridSearchCV con KFold(5, shuffle=True, random_state=7) y scoring='r2'
#    Ajustado únicamente con el conjunto de entrenamiento. El R2 medio de
#    validación cruzada lo calcula GridSearchCV internamente: en cada fold
#    el modelo se ajusta con 4 folds y se evalúa con el fold restante, por
#    lo que nunca se predice sobre los mismos datos con los que se ajustó.
# ----------------------------------------------------------------------
cv = KFold(n_splits=5, shuffle=True, random_state=7)

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='r2',
    cv=cv
)
grid.fit(X_train, y_train)

best_alpha = float(grid.best_params_['ridge__alpha'])
best_cv_r2 = float(grid.best_score_)
best_index = int(grid.best_index_)

# Resultados completos de la malla (media y desviación del R2 de CV por alpha)
mean_scores = np.asarray(grid.cv_results_['mean_test_score'], dtype=float)
std_scores = np.asarray(grid.cv_results_['std_test_score'], dtype=float)


# ----------------------------------------------------------------------
# 4) Guardar TODAS las cifras en resultados.json (conservando claves previas)
# ----------------------------------------------------------------------
nuevas_cifras = {
    't2_n_alphas_evaluados': int(len(alphas)),
    't2_alphas_evaluados': [float(a) for a in alphas],
    't2_r2_medio_cv_por_alpha': [float(s) for s in mean_scores],
    't2_r2_std_cv_por_alpha': [float(s) for s in std_scores],
    't2_alpha_elegido': best_alpha,
    't2_r2_medio_cv': best_cv_r2,
    't2_indice_mejor_alpha': best_index,
    't2_n_splits_cv': 5,
    't2_shuffle_cv': True,
    't2_random_state_cv': 7,
    't2_scoring': 'r2',
    't2_n_ejemplos_entrenamiento': int(X_train.shape[0]),
    't2_n_ejemplos_prueba_reservados': int(X_test.shape[0]),
}

try:
    with open('resultados.json', 'r', encoding='utf-8') as f:
        resultados = json.load(f)
    if not isinstance(resultados, dict):
        resultados = {}
except Exception:
    resultados = {}

resultados.update(nuevas_cifras)

with open('resultados.json', 'w', encoding='utf-8') as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)


# ----------------------------------------------------------------------
# 5) Salida por consola (cifras principales con 4 decimales)
# ----------------------------------------------------------------------
print("=== Subtarea T2: pipeline StandardScaler + Ridge, alpha por validacion cruzada ===")
print("Datos: sklearn.datasets.load_diabetes")
print(f"Entrenamiento usado para el grid: {X_train.shape[0]} ejemplos "
      f"(prueba reservada: {X_test.shape[0]} ejemplos, sin usar aqui)")
print("Pipeline: StandardScaler -> Ridge")
print("Busqueda: GridSearchCV sobre 13 valores de np.logspace(-3, 3, 13)")
print("CV: KFold(n_splits=5, shuffle=True, random_state=7), scoring='r2'")
print("")
print("R2 medio de CV por alpha:")
for a, m, s in zip(alphas, mean_scores, std_scores):
    marca = "  <-- mejor" if float(a) == best_alpha else ""
    print(f"  alpha={float(a):10.4f} : R2 medio CV = {m:.4f} (+/- {s:.4f}){marca}")
print("")
print(f"Alpha elegido: {best_alpha:.4f}")
print(f"R2 medio de validacion cruzada (5 folds): {best_cv_r2:.4f}")
print("Cifras guardadas en resultados.json")
