# -*- coding: utf-8 -*-
"""
SUBTAREA T4: Figura matplotlib — R2 medio de validación cruzada (de T2) contra alpha,
con el eje x en escala logarítmica (plt.xscale('log')).

El script es independiente: recalcula la validación cruzada de la Parte 2
(pipeline StandardScaler + Ridge, GridSearchCV sobre numpy.logspace(-3, 3, 13),
KFold(n_splits=5, shuffle=True, random_state=7), scoring="r2", usando solo el
conjunto de entrenamiento) y dibuja el R2 medio de CV frente a alpha.
El conjunto de prueba queda reservado y no se usa para nada aquí.
"""

import json

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


def main():
    # ------------------------------------------------------------------
    # Datos y partición (misma configuración que en T2: 75% train / 25% test)
    # ------------------------------------------------------------------
    datos = load_diabetes()
    X, y = datos.data, datos.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    # El conjunto de prueba queda reservado: no se usa para elegir alpha
    # ni para ajustar el escalador.

    # ------------------------------------------------------------------
    # Parte 2 (recalculada): selección de alpha por validación cruzada
    # ------------------------------------------------------------------
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge()),
    ])

    alphas = np.logspace(-3, 3, 13)
    param_grid = {"ridge__alpha": alphas}

    cv = KFold(n_splits=5, shuffle=True, random_state=7)

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="r2",
        return_train_score=False,
    )
    grid.fit(X_train, y_train)

    # Resultados de CV por alpha (GridSearchCV conserva el orden de la rejilla,
    # que es ascendente por construcción de logspace)
    r2_medio = np.asarray(grid.cv_results_["mean_test_score"], dtype=float)
    r2_std = np.asarray(grid.cv_results_["std_test_score"], dtype=float)

    idx_mejor = int(grid.best_index_)
    alpha_elegido = float(grid.best_params_["ridge__alpha"])
    r2_medio_elegido = float(r2_medio[idx_mejor])
    r2_std_elegido = float(r2_std[idx_mejor])

    # ------------------------------------------------------------------
    # Parte 4: figura R2 medio de CV contra alpha (eje x logarítmico)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        alphas, r2_medio,
        marker="o", color="tab:blue", linewidth=1.8,
        label="R2 medio de CV (5 folds)",
    )
    ax.axvline(
        alpha_elegido, color="tab:red", linestyle="--", linewidth=1.2,
        label=f"alpha elegido = {alpha_elegido:.4g}",
    )
    ax.set_xscale("log")
    ax.set_xlabel("alpha (escala logarítmica)")
    ax.set_ylabel("R2 medio de validación cruzada")
    ax.set_title("Ridge (diabetes): R2 medio de CV frente a alpha")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig("t4_r2_cv_vs_alpha.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Guardar todas las cifras calculadas en resultados.json
    # ------------------------------------------------------------------
    resultados = {
        "subtarea": "T4_figura_r2_medio_cv_vs_alpha",
        "dataset": "sklearn.datasets.load_diabetes",
        "n_ejemplos_entrenamiento": int(X_train.shape[0]),
        "n_ejemplos_prueba_reservados": int(X_test.shape[0]),
        "pipeline": "StandardScaler + Ridge",
        "rejilla_alpha": "numpy.logspace(-3, 3, 13)",
        "cv": "KFold(n_splits=5, shuffle=True, random_state=7)",
        "scoring": "r2",
        "n_alphas_evaluados": int(len(alphas)),
        "alphas_evaluados": [float(a) for a in alphas],
        "r2_medio_cv_por_alpha": [float(v) for v in r2_medio],
        "r2_std_cv_por_alpha": [float(v) for v in r2_std],
        "indice_mejor_alpha": idx_mejor,
        "alpha_elegido": alpha_elegido,
        "r2_medio_cv_alpha_elegido": r2_medio_elegido,
        "r2_std_cv_alpha_elegido": r2_std_elegido,
        "figura_archivo": "t4_r2_cv_vs_alpha.png",
        "figura_eje_x_escala": "log",
    }
    with open("resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Impresión de las cifras principales (4 decimales)
    # ------------------------------------------------------------------
    print("=== T4: Figura R2 medio de validación cruzada frente a alpha ===")
    print(f"Entrenamiento: {X_train.shape[0]} ejemplos | Prueba reservada (no usada): {X_test.shape[0]}")
    print("CV: KFold(n_splits=5, shuffle=True, random_state=7) | scoring: r2")
    print("Eje x de la figura en escala logarítmica")
    print("-" * 50)
    print(f"{'Alpha':>12}  {'R2 medio CV':>12}")
    for a, m in zip(alphas, r2_medio):
        print(f"{float(a):12.4f}  {float(m):12.4f}")
    print("-" * 50)
    print(f"Alpha elegido: {alpha_elegido:.4f}")
    print(f"R2 medio de validación cruzada: {r2_medio_elegido:.4f}")
    print(f"Desv. est. del R2 en CV (alpha elegido): {r2_std_elegido:.4f}")
    print("Figura guardada en: t4_r2_cv_vs_alpha.png")
    print("Resultados guardados en: resultados.json")


if __name__ == "__main__":
    main()
