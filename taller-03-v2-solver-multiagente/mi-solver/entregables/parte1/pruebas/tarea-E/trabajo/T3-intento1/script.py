# -*- coding: utf-8 -*-
"""
Subtarea T3: con el alpha elegido en T2, evaluar sobre el conjunto de prueba
(usado una sola vez) el R2 y el RMSE del Ridge, y comparar con LinearRegression
con el mismo escalado (StandardScaler) evaluada sobre la misma prueba.

El script es independiente: recalcula el alpha de T2 con exactamente el mismo
procedimiento (GridSearchCV, KFold(n_splits=5, shuffle=True, random_state=7),
scoring="r2", alphas = np.logspace(-3, 3, 13)) usando SOLO el conjunto de
entrenamiento. El conjunto de prueba se usa una única vez, al final.
"""

import json
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

RUTA_RESULTADOS = "resultados.json"


def main():
    # ------------------------------------------------------------------
    # 1) Datos y única división entrenamiento/prueba (idéntica a T1)
    # ------------------------------------------------------------------
    datos = load_diabetes()
    X, y = datos.data, datos.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=7
    )

    # ------------------------------------------------------------------
    # 2) Re-cálculo del alpha elegido en T2 (solo con el entrenamiento)
    # ------------------------------------------------------------------
    alphas = np.logspace(-3, 3, 13)

    pipe_ridge = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge()),
    ])

    cv = KFold(n_splits=5, shuffle=True, random_state=7)
    busqueda = GridSearchCV(
        estimator=pipe_ridge,
        param_grid={"ridge__alpha": alphas},
        cv=cv,
        scoring="r2",
    )
    # La prueba NO participa en esta búsqueda
    busqueda.fit(X_train, y_train)

    alpha_elegido = float(busqueda.best_params_["ridge__alpha"])
    r2_cv_alpha_elegido = float(busqueda.best_score_)
    r2_medio_cv_por_alpha = [float(v) for v in busqueda.cv_results_["mean_test_score"]]
    r2_std_cv_por_alpha = [float(v) for v in busqueda.cv_results_["std_test_score"]]

    # ------------------------------------------------------------------
    # 3) Evaluación ÚNICA en prueba: Ridge con el alpha elegido
    #    (escalador ajustado solo con entrenamiento, dentro del pipeline)
    # ------------------------------------------------------------------
    ridge_final = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha_elegido)),
    ])
    ridge_final.fit(X_train, y_train)
    y_pred_ridge = ridge_final.predict(X_test)
    ridge_r2_prueba = float(r2_score(y_test, y_pred_ridge))
    ridge_rmse_prueba = float(np.sqrt(mean_squared_error(y_test, y_pred_ridge)))

    # ------------------------------------------------------------------
    # 4) Comparación: LinearRegression con el mismo escalado,
    #    evaluada sobre la misma prueba
    # ------------------------------------------------------------------
    linreg_final = Pipeline([
        ("scaler", StandardScaler()),
        ("linreg", LinearRegression()),
    ])
    linreg_final.fit(X_train, y_train)
    y_pred_linreg = linreg_final.predict(X_test)
    linreg_r2_prueba = float(r2_score(y_test, y_pred_linreg))
    linreg_rmse_prueba = float(np.sqrt(mean_squared_error(y_test, y_pred_linreg)))

    # Diferencias (positivas => Ridge mejor en prueba)
    diff_r2 = float(ridge_r2_prueba - linreg_r2_prueba)
    diff_rmse = float(linreg_rmse_prueba - ridge_rmse_prueba)
    if diff_r2 > 0:
        mejor_r2 = "ridge"
    elif diff_r2 < 0:
        mejor_r2 = "linear_regression"
    else:
        mejor_r2 = "empate"

    # ------------------------------------------------------------------
    # 5) Guardar TODAS las cifras calculadas en resultados.json
    # ------------------------------------------------------------------
    resultados = {
        "t3_n_ejemplos_entrenamiento": float(X_train.shape[0]),
        "t3_n_ejemplos_prueba": float(X_test.shape[0]),
        "t3_n_alphas_evaluados": float(len(alphas)),
        "t3_alphas_evaluados": [float(a) for a in alphas],
        "t3_r2_medio_cv_por_alpha": r2_medio_cv_por_alpha,
        "t3_r2_std_cv_por_alpha": r2_std_cv_por_alpha,
        "t3_alpha_elegido": alpha_elegido,
        "t3_r2_medio_cv_alpha_elegido": r2_cv_alpha_elegido,
        "t3_ridge_r2_prueba": ridge_r2_prueba,
        "t3_ridge_rmse_prueba": ridge_rmse_prueba,
        "t3_linreg_r2_prueba": linreg_r2_prueba,
        "t3_linreg_rmse_prueba": linreg_rmse_prueba,
        "t3_diferencia_r2_ridge_menos_lineal": diff_r2,
        "t3_diferencia_rmse_lineal_menos_ridge": diff_rmse,
        "t3_modelo_con_mejor_r2_prueba": mejor_r2,
    }

    with open(RUTA_RESULTADOS, "w") as f:
        json.dump(resultados, f, indent=2)

    # ------------------------------------------------------------------
    # 6) Salida por pantalla (cifras principales con 4 decimales)
    # ------------------------------------------------------------------
    print("=== Subtarea T3: evaluacion unica en el conjunto de prueba ===")
    print("Datos: sklearn.datasets.load_diabetes")
    print(
        "Entrenamiento: %d ejemplos | Prueba: %d ejemplos (usada una sola vez, aqui)"
        % (X_train.shape[0], X_test.shape[0])
    )
    print("Alpha re-elegido con el mismo procedimiento de T2 (solo entrenamiento):")
    print(
        "  alpha = %.4f | R2 medio de CV (5 folds) = %.4f"
        % (alpha_elegido, r2_cv_alpha_elegido)
    )
    print()
    print("Resultados sobre el conjunto de prueba:")
    print(
        "  Ridge (alpha=%.4f):                R2 = %.4f | RMSE = %.4f"
        % (alpha_elegido, ridge_r2_prueba, ridge_rmse_prueba)
    )
    print(
        "  LinearRegression (mismo escalado): R2 = %.4f | RMSE = %.4f"
        % (linreg_r2_prueba, linreg_rmse_prueba)
    )
    print()
    print("Diferencia R2   (Ridge - Lineal):   %.4f" % diff_r2)
    print("Diferencia RMSE (Lineal - Ridge):   %.4f" % diff_rmse)
    print("Mejor R2 en prueba: %s" % mejor_r2)
    print("Cifras guardadas en resultados.json")


if __name__ == "__main__":
    main()
