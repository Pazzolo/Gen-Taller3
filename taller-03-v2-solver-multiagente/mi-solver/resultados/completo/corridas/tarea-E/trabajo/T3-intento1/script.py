# -*- coding: utf-8 -*-
"""
Subtarea T3 — Evaluación en prueba (una sola vez).

Script independiente: recalcula la partición train/test y la selección de alpha
por validación cruzada (usando SOLO el train, igual que en la subtarea anterior),
y luego evalúa UNA sola vez sobre el conjunto de prueba:
  - Ridge con el alpha elegido por CV (mismo pipeline escalado: StandardScaler + Ridge).
  - LinearRegression sin regularizar (mismo escalado: StandardScaler + LinearRegression).
Guarda todas las cifras en resultados.json. No genera figuras.
"""

import json
import math

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Misma partición 75/25 que en la subtarea anterior (T2)
RANDOM_STATE_SPLIT = 42


def main():
    resultados = {}

    # ------------------------------------------------------------------
    # Datos y partición (idéntica a la subtarea anterior: 75% train / 25% test)
    # ------------------------------------------------------------------
    datos = load_diabetes()
    X, y = datos.data, datos.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE_SPLIT
    )

    resultados["T3_dataset"] = "diabetes"
    resultados["T3_n_features"] = float(X.shape[1])
    resultados["T3_n_train"] = float(X_train.shape[0])
    resultados["T3_n_test"] = float(X_test.shape[0])
    resultados["T3_random_state_split"] = float(RANDOM_STATE_SPLIT)

    # ------------------------------------------------------------------
    # Re-selección de alpha por validación cruzada usando SOLO el train
    # (misma configuración que la Parte 2 de la tarea)
    # ------------------------------------------------------------------
    alphas = np.logspace(-3, 3, 13)
    pipe_ridge = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())])
    cv = KFold(n_splits=5, shuffle=True, random_state=7)
    grid = GridSearchCV(
        estimator=pipe_ridge,
        param_grid={"ridge__alpha": alphas},
        cv=cv,
        scoring="r2",
    )
    grid.fit(X_train, y_train)  # la prueba no participa en ninguna decisión

    alpha_elegido = float(grid.best_params_["ridge__alpha"])
    r2_cv_elegido = float(grid.best_score_)

    resultados["T3_alphas_evaluados"] = [float(a) for a in alphas]
    resultados["T3_r2_medio_cv_por_alpha"] = [
        float(v) for v in grid.cv_results_["mean_test_score"]
    ]
    resultados["T3_r2_std_cv_por_alpha"] = [
        float(v) for v in grid.cv_results_["std_test_score"]
    ]
    resultados["T3_indice_alpha_elegido"] = float(grid.best_index_)
    resultados["T3_alpha_elegido"] = alpha_elegido
    resultados["T3_r2_medio_cv_alpha_elegido"] = r2_cv_elegido
    resultados["T3_n_splits_cv"] = 5.0
    resultados["T3_cv_shuffle"] = True
    resultados["T3_cv_random_state"] = 7.0
    resultados["T3_scoring"] = "r2"

    # ------------------------------------------------------------------
    # Evaluación UNICA sobre el conjunto de prueba
    # ------------------------------------------------------------------
    # Ridge con el alpha elegido (mismo pipeline escalado), ajustado con todo el train
    pipe_ridge_final = Pipeline(
        [("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha_elegido))]
    )
    pipe_ridge_final.fit(X_train, y_train)
    pred_ridge_test = pipe_ridge_final.predict(X_test)
    r2_ridge = float(r2_score(y_test, pred_ridge_test))
    rmse_ridge = float(math.sqrt(mean_squared_error(y_test, pred_ridge_test)))

    # LinearRegression sin regularizar (mismo escalado), ajustado con todo el train
    pipe_lin = Pipeline(
        [("scaler", StandardScaler()), ("linreg", LinearRegression())]
    )
    pipe_lin.fit(X_train, y_train)
    pred_lin_test = pipe_lin.predict(X_test)
    r2_lin = float(r2_score(y_test, pred_lin_test))
    rmse_lin = float(math.sqrt(mean_squared_error(y_test, pred_lin_test)))

    resultados["T3_ridge_r2_test"] = r2_ridge
    resultados["T3_ridge_rmse_test"] = rmse_ridge
    resultados["T3_linreg_r2_test"] = r2_lin
    resultados["T3_linreg_rmse_test"] = rmse_lin
    resultados["T3_diferencia_r2_ridge_menos_lineal"] = float(r2_ridge - r2_lin)
    resultados["T3_diferencia_rmse_lineal_menos_ridge"] = float(rmse_lin - rmse_ridge)
    resultados["T3_modelo_con_mejor_r2_test"] = (
        "ridge" if r2_ridge >= r2_lin else "lineal"
    )
    resultados["T3_modelo_con_menor_rmse_test"] = (
        "ridge" if rmse_ridge <= rmse_lin else "lineal"
    )

    # ------------------------------------------------------------------
    # Guardado de resultados
    # ------------------------------------------------------------------
    with open("resultados.json", "w", encoding="utf-8") as f:
        json.dump({"resultados": resultados}, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Salida por pantalla (cifras principales con 4 decimales)
    # ------------------------------------------------------------------
    print("Parte 3 — Evaluacion en prueba (una sola vez)")
    print(
        f"Train: {X_train.shape[0]} ejemplos | Test: {X_test.shape[0]} ejemplos "
        "(la prueba se usa solo aqui)"
    )
    print(
        f"Alpha elegido por CV (solo train): {alpha_elegido:.4f} "
        f"| R2 medio de CV: {r2_cv_elegido:.4f}"
    )
    print()
    print(f"{'Modelo':<30}{'R2 (test)':>12}{'RMSE (test)':>14}")
    print(f"{'Ridge (alpha de CV)':<30}{r2_ridge:>12.4f}{rmse_ridge:>14.4f}")
    print(f"{'LinearRegression (sin reg.)':<30}{r2_lin:>12.4f}{rmse_lin:>14.4f}")
    print()
    print(f"Diferencia R2 (Ridge - Lineal): {r2_ridge - r2_lin:+.4f}")
    print(f"Diferencia RMSE (Lineal - Ridge): {rmse_lin - rmse_ridge:+.4f}")
    print()
    print("Cifras guardadas en resultados.json")


if __name__ == "__main__":
    main()
