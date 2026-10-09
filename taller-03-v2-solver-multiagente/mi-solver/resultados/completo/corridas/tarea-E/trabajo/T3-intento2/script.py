```python
# -*- coding: utf-8 -*-
"""
Subtarea T3 — Parte 3: evaluación en prueba (una sola vez).

Esta subtarea DEPENDE de T2: hay que evaluar el Ridge con EL alpha elegido en la
Parte 2 y sobre EL MISMO conjunto de prueba reservado allí. Como el script es
independiente, procede así:

  1) Reconstruye la partición EXACTA de T2 (train 331 / test 111). El
     random_state del split de T2 no fue 42, así que se busca de forma
     determinista: para cada candidato se hace el split y se corre la CV del
     primer alpha con la configuración exacta de T2 (pipeline
     StandardScaler+Ridge, KFold(5, shuffle=True, random_state=7),
     scoring='r2'); los candidatos que reproducen el R2 CV medio de T2 para
     alpha=0.001 se verifican re-ejecutando el GridSearchCV completo de T2 y
     exigiendo que las 13 medias CV, el índice del mejor alpha y el alpha
     elegido coincidan cifra a cifra con los resultados de T2.

  2) Con la partición verificada, el alpha elegido sale del GridSearchCV
     re-ejecutado SOLO con el train y se comprueba que es igual a
     T2_alpha_elegido (0.31622776601683794). Si la partición no pudiera
     reproducirse, se aplica el fallback indicado por el revisor: tomar el
     alpha directamente de los resultados de T2.

  3) Evalúa UNA sola vez sobre la prueba: Ridge(alpha elegido) y
     LinearRegression sin regularizar, ambos con el mismo escalado
     (StandardScaler -> modelo), ajustados con todo el train.

Guarda todas las cifras en resultados.json (conservando las claves previas que
existan) y no genera figuras.
"""

import json
import math

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ------------------- Configuración fija (idéntica a T2 / Parte 2) -------------------
TEST_SIZE = 0.25
ALPHAS = [float(a) for a in np.logspace(-3, 3, 13)]
N_SPLITS_CV = 5
CV_RANDOM_STATE = 7
SCORING = "r2"
TOL = 1e-8  # tolerancia para exigir reproducción cifra a cifra

# ---------------- Referencias de T2 (resultados de la subtarea previa) --------------
T2_R2_MEDIO_CV_POR_ALPHA = [
    0.47238576495065043, 0.4723869184448282, 0.4723904754415161,
    0.472400842474254, 0.472425566795397, 0.4724424112965881,
    0.472221058274312, 0.4714130997804261, 0.47077524438376156,
    0.4698055461897265, 0.4569728732889732, 0.39947277791759284,
    0.2790407278411223,
]
T2_ALPHA_ELEGIDO = 0.31622776601683794
T2_INDICE_ALPHA_ELEGIDO = 5
T2_N_TRAIN = 331
T2_N_TEST = 111


def hacer_pipeline_ridge(alpha=None):
    if alpha is None:
        return Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())])
    return Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha))])


def hacer_split(X, y, random_state, metodo):
    """Partición candidata 75/25 (train 331 / test 111)."""
    if metodo == "train_test_split":
        # Nota: con test_size=0.25, 111 o train_size=331 el resultado es
        # idéntico para un mismo random_state (mismo n_test y mismo barajado).
        return train_test_split(X, y, test_size=TEST_SIZE, random_state=random_state)
    # metodo == "permutation_rng": barajado manual con el generador moderno
    rng = np.random.default_rng(random_state)
    perm = rng.permutation(X.shape[0])
    n_test = int(math.ceil(TEST_SIZE * X.shape[0]))
    ind_test = perm[:n_test]
    ind_train = perm[n_test:]
    return X[ind_train], X[ind_test], y[ind_train], y[ind_test]


def r2_cv_medio_primer_alpha(X_train, y_train, kf):
    """R2 medio de CV (configuración exacta de T2) para el primer alpha."""
    scores = cross_val_score(
        hacer_pipeline_ridge(ALPHAS[0]), X_train, y_train, cv=kf, scoring=SCORING
    )
    return float(np.mean(scores))


def grid_completo(X_train, y_train, kf):
    """GridSearchCV exacto de la Parte 2, usando SOLO el train."""
    grid = GridSearchCV(
        hacer_pipeline_ridge(),
        param_grid={"ridge__alpha": ALPHAS},
        cv=kf,
        scoring=SCORING,
    )
    grid.fit(X_train, y_train)
    return grid


def buscar_particion_de_T2(X, y, kf):
    """Busca la partición cuya CV reproduce cifra a cifra los resultados de T2."""
    ref0 = T2_R2_MEDIO_CV_POR_ALPHA[0]
    comunes = [
        42, 7, 0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 17, 20, 21, 23, 24,
        25, 50, 77, 99, 100, 101, 123, 202, 321, 333, 404, 500, 555, 666, 777,
        888, 999, 1000, 1234, 2023, 2024, 2025, 6013, 60132025, 31415, 31416,
        2718, 27182,
    ]
    comunes_set = set(comunes)
    planes = [
        ("train_test_split",
         comunes + [r for r in range(5001) if r not in comunes_set]),
        ("permutation_rng",
         comunes + [r for r in range(2001) if r not in comunes_set]),
    ]
    for metodo, candidatos in planes:
        probados = 0
        for rs in candidatos:
            X_tr, X_te, y_tr, y_te = hacer_split(X, y, rs, metodo)
            probados += 1
            if probados % 500 == 0:
                print(f"  ...buscando particion de T2 ({probados} candidatos probados)")
            if X_tr.shape[0] != T2_N_TRAIN or X_te.shape[0] != T2_N_TEST:
                continue
            # Filtro rápido con el primer alpha
            if abs(r2_cv_medio_primer_alpha(X_tr, y_tr, kf) - ref0) > TOL:
                continue
            # Verificación completa contra T2
            grid = grid_completo(X_tr, y_tr, kf)
            medias = [float(v) for v in grid.cv_results_["mean_test_score"]]
            coincide = (
                np.allclose(
                    np.array(medias),
                    np.array(T2_R2_MEDIO_CV_POR_ALPHA),
                    rtol=0.0,
                    atol=TOL,
                )
                and int(grid.best_index_) == T2_INDICE_ALPHA_ELEGIDO
                and abs(float(grid.best_params_["ridge__alpha"]) - T2_ALPHA_ELEGIDO)
                <= 1e-12
            )
            if coincide:
                return {
                    "metodo": metodo,
                    "random_state": int(rs),
                    "grid": grid,
                    "X_train": X_tr,
                    "X_test": X_te,
                    "y_train": y_tr,
                    "y_test": y_te,
                }
    return None


def main():
    datos = load_diabetes()
    X, y = datos.data, datos.target
    kf = KFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=CV_RANDOM_STATE)

    # Cargar resultados previos del archivo (si existen) para conservar sus claves
    previos = {}
    try:
        with open("resultados.json", "r", encoding="utf-8") as f:
            contenido = json.load(f)
        if isinstance(contenido, dict):
            previos = contenido.get("resultados", contenido)
            if not isinstance(previos, dict):
                previos = {}
    except Exception:
        previos = {}

    print("Parte 3 — Evaluacion en prueba (una sola vez)")
    print("Reconstruyendo la particion exacta de T2 (train 331 / test 111)...")

    hallazgo = buscar_particion_de_T2(X, y, kf)

    if hallazgo is not None:
        rs = hallazgo["random_state"]
        metodo = hallazgo["metodo"]
        grid = hallazgo["grid"]
        X_train, X_test = hallazgo["X_train"], hallazgo["X_test"]
        y_train, y_test = hallazgo["y_train"], hallazgo["y_test"]
        split_reproducida = True
        alpha_elegido = float(grid.best_params_["ridge__alpha"])
        alpha_fuente = (
            "GridSearchCV re-ejecutado sobre el train de T2 "
            "(verificado cifra a cifra contra T2)"
        )
        print(f"  Particion de T2 reproducida: metodo={metodo}, random_state={rs}")
    else:
        # Fallback (corrección del revisor): si no se puede garantizar la
        # reproducción de la partición, se toma el alpha directamente de los
        # resultados de T2.
        rs = 42
        metodo = "train_test_split"
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=rs
        )
        split_reproducida = False
        alpha_elegido = float(T2_ALPHA_ELEGIDO)
        alpha_fuente = "T2_alpha_elegido (particion de
