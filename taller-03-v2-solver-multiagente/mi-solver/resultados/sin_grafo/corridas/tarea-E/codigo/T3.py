# -*- coding: utf-8 -*-
"""
SUBTAREA T3: Evaluación única sobre el conjunto de prueba.
- Recalcula de forma independiente la división 75/25 (random_state=7) y la selección
  de alpha por validación cruzada (para reproducir el alpha elegido en T2).
- Con ese alpha, evalúa UNA sola vez el pipeline StandardScaler+Ridge sobre la prueba:
  reporta R2 y RMSE.
- Compara con LinearRegression (mismo StandardScaler) evaluada sobre la misma prueba.
- Sin figuras. Guarda todas las cifras en resultados.json.
"""

import json
import math

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# ---------------------------------------------------------------------------
# 1) Datos y división idéntica a T1 (test 25%, random_state=7)
# ---------------------------------------------------------------------------
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=7
)

# ---------------------------------------------------------------------------
# 2) Reproducción de la selección de alpha de T2 (SOLO entrenamiento)
#    Rejilla: numpy.logspace(-3, 3, 13); CV: KFold(5, shuffle=True, random_state=7)
# ---------------------------------------------------------------------------
alphas = np.logspace(-3, 3, 13)
pipe_ridge = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge()),
])
cv = KFold(n_splits=5, shuffle=True, random_state=7)
grid = GridSearchCV(
    estimator=pipe_ridge,
    param_grid={"ridge__alpha": alphas},
    cv=cv,
    scoring="r2",
)
grid.fit(X_train, y_train)  # solo entrenamiento: la prueba no participa aquí

alpha_recalculado = float(grid.best_params_["ridge__alpha"])
r2_cv_alpha = float(grid.best_score_)

# Alpha elegido en T2 (referencia del enunciado); verificamos que la reproducción coincide
alpha_t2 = 0.31622776601683794
coincide = math.isclose(alpha_recalculado, alpha_t2, rel_tol=1e-12, abs_tol=1e-12)
alpha_final = alpha_t2 if coincide else alpha_recalculado

# ---------------------------------------------------------------------------
# 3) Evaluación ÚNICA sobre el conjunto de prueba: Ridge con el alpha elegido
# ---------------------------------------------------------------------------
pipe_ridge_final = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=alpha_final)),
])
pipe_ridge_final.fit(X_train, y_train)
y_pred_ridge = pipe_ridge_final.predict(X_test)

r2_prueba_ridge = float(r2_score(y_test, y_pred_ridge))
rmse_prueba_ridge = float(np.sqrt(mean_squared_error(y_test, y_pred_ridge)))

# ---------------------------------------------------------------------------
# 4) Comparación: LinearRegression con el mismo StandardScaler, misma prueba
# ---------------------------------------------------------------------------
pipe_linreg = Pipeline([
    ("scaler", StandardScaler()),
    ("linreg", LinearRegression()),
])
pipe_linreg.fit(X_train, y_train)
y_pred_linreg = pipe_linreg.predict(X_test)

r2_prueba_linreg = float(r2_score(y_test, y_pred_linreg))
rmse_prueba_linreg = float(np.sqrt(mean_squared_error(y_test, y_pred_linreg)))

# Diferencias (Ridge - Lineal)
dif_r2 = float(r2_prueba_ridge - r2_prueba_linreg)
dif_rmse = float(rmse_prueba_ridge - rmse_prueba_linreg)

mejor_modelo = "Ridge" if r2_prueba_ridge >= r2_prueba_linreg else "LinearRegression"

# ---------------------------------------------------------------------------
# 5) Guardado de todas las cifras en resultados.json
# ---------------------------------------------------------------------------
resultados = {
    "subtarea": "T3_evaluacion_en_prueba",
    "dataset": "sklearn.datasets.load_diabetes",
    "test_size": 0.25,
    "random_state": 7,
    "n_ejemplos_entrenamiento": int(X_train.shape[0]),
    "n_ejemplos_prueba": int(X_test.shape[0]),
    "pipeline_ridge": "StandardScaler + Ridge",
    "pipeline_lineal": "StandardScaler + LinearRegression",
    "rejilla_alpha": "numpy.logspace(-3, 3, 13)",
    "cv": "KFold(n_splits=5, shuffle=True, random_state=7)",
    "scoring": "r2",
    "alpha_elegido_T2": float(alpha_t2),
    "alpha_recalculado_grid": float(alpha_recalculado),
    "alpha_recalculado_coincide_con_T2": bool(coincide),
    "alpha_usado_en_prueba": float(alpha_final),
    "r2_medio_cv_alpha_elegido": float(r2_cv_alpha),
    "r2_prueba_ridge": r2_prueba_ridge,
    "rmse_prueba_ridge": rmse_prueba_ridge,
    "r2_prueba_lineal": r2_prueba_linreg,
    "rmse_prueba_lineal": rmse_prueba_linreg,
    "diferencia_r2_ridge_menos_lineal": dif_r2,
    "diferencia_rmse_ridge_menos_lineal": dif_rmse,
    "mejor_modelo_en_prueba": mejor_modelo,
}

with open("resultados.json", "w") as f:
    json.dump(resultados, f, indent=2)

# ---------------------------------------------------------------------------
# 6) Impresión de las cifras principales (4 decimales)
# ---------------------------------------------------------------------------
print("=== T3: Evaluación única sobre el conjunto de prueba ===")
print(f"Entrenamiento: {X_train.shape[0]} ejemplos | Prueba: {X_test.shape[0]} ejemplos (test_size=0.25, random_state=7)")
print("La prueba se usa una sola vez, con el alpha ya elegido en T2 (solo entrenamiento/CV).")
print("----------------------------------------------------")
print(f"Alpha elegido en T2: {alpha_t2:.4f}")
print(f"Alpha reproducido con GridSearchCV: {alpha_recalculado:.4f} (coincide: {coincide})")
print(f"R2 medio de validación cruzada (alpha elegido): {r2_cv_alpha:.4f}")
print("----------------------------------------------------")
print("Modelo 1) StandardScaler + Ridge (alpha elegido):")
print(f"  R2 prueba  = {r2_prueba_ridge:.4f}")
print(f"  RMSE prueba = {rmse_prueba_ridge:.4f}")
print("Modelo 2) StandardScaler + LinearRegression (sin regularizar):")
print(f"  R2 prueba  = {r2_prueba_linreg:.4f}")
print(f"  RMSE prueba = {rmse_prueba_linreg:.4f}")
print("----------------------------------------------------")
print(f"Diferencia R2 (Ridge - Lineal): {dif_r2:.4f}")
print(f"Diferencia RMSE (Ridge - Lineal): {dif_rmse:.4f}")
print(f"Mejor modelo en prueba (por R2): {mejor_modelo}")
print("Resultados guardados en resultados.json")
