# -*- coding: utf-8 -*-
"""
Subtarea T3 - Parte 3: evaluacion en prueba (una sola vez).

Depende de T2 (alpha elegido y particion train/test). Como el script es
independiente, reconstruye la parte necesaria de T2 de forma determinista:

1) Busca la particion exacta de T2 (train 331 / test 111): para cada
   random_state candidato construye el split y comprueba si el R2 medio de CV
   del pipeline StandardScaler+Ridge (alpha=0.001, KFold(5, shuffle=True,
   random_state=7), scoring='r2') reproduce el valor de T2; los candidatos que
   pasan ese filtro se verifican re-ejecutando el GridSearchCV completo de T2
   y exigiendo que las 13 medias CV, el indice del mejor alpha y el alpha
   elegido coincidan con los resultados de T2.

2) Con la particion verificada, el alpha elegido sale del GridSearchCV
   re-ejecutado SOLO con el train (coincide con T2_alpha_elegido). Si la
   particion no se puede reproducir, fallback indicado por el revisor: split
   75/25 con random_state=42 y alpha tomado directamente de los resultados
   de T2 (elegido en T2 solo con CV sobre train, sin tocar ninguna prueba).

3) Evalua UNA sola vez sobre la prueba: Ridge(alpha elegido) y
   LinearRegression sin regularizar, ambos con el mismo escalado
   (StandardScaler ajustado solo con train, dentro de un Pipeline).

Guarda todas las cifras en resultados.json (conservando claves previas) y
no genera figuras.
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

# ----------------- Configuracion fija (identica a T2 / Parte 2) -----------------
TEST_SIZE = 0.25
ALPHAS = [float(a) for a in np.logspace(-3, 3, 13)]
N_SPLITS_CV = 5
CV_RANDOM_STATE = 7
SCORING = "r2"
TOL = 1e-6  # tolerancia para la verificacion contra T2

# ---------------- Referencias de T2 (resultados de la subtarea previa) ----------
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


def hacer_pipeline_linreg():
    return Pipeline([("scaler", StandardScaler()), ("linreg", LinearRegression())])


def hacer_split(X, y, random_state, metodo):
    """Particion candidata 75/25 (train 331 / test 111)."""
    if metodo == "train_test_split":
        # Con n=442, test_size=0.25 equivale a 111 de test; para un mismo
        # random_state el split es identico con test_size o train_size.
        return train_test_split(X, y, test_size=TEST_SIZE, random_state=random_state)
    # metodo == "permutation_rng": barajado manual con el generador moderno
    rng = np.random.default_rng(random_state)
    perm = rng.permutation(X.shape[0])
    n_test = int(math.ceil(TEST_SIZE * X.shape[0]))
    ind_test = perm[:n_test]
    ind_train = perm[n_test:]
    return X[ind_train], X[ind_test], y[ind_train], y[ind_test]


def r2_cv_medio_primer_alpha(X_train, y_train, kf):
    """R2 medio de CV (configuracion exacta de T2) para el primer alpha."""
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
    """Busca la particion cuya CV reproduce los resultados de T2."""
    ref0 = T2_R2_MEDIO_CV_POR_ALPHA[0]
    comunes = [42, 7, 0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 17, 20, 21,
               23, 24, 25, 50, 77, 99, 100, 101, 123, 202, 321, 333, 404, 500,
               555, 666, 777, 888, 999, 1000, 1234, 2023, 2024, 2025, 6013,
               60132025, 31415, 31416, 2718, 27182]
    comunes_set = set(comunes)
    planes = [
        ("train_test_split",
         comunes + [r for r in range(5001) if r not in comunes_set]),
        ("permutation_rng",
         comunes + [r for r in range(2001) if r not in comunes_set]),
    ]
    probados_total = 0
    for metodo, candidatos in planes:
        probados_metodo = 0
        for rs in candidatos:
            probados_total += 1
            probados_metodo += 1
            if probados_metodo % 500 == 0:
                print("  ...buscando particion de T2 (%d candidatos probados)"
                      % probados_total)
            X_tr, X_te, y_tr, y_te = hacer_split(X, y, rs, metodo)
            if X_tr.shape[0] != T2_N_TRAIN or X_te.shape[0] != T2_N_TEST:
                continue
            # Filtro rapido con el primer alpha
            if abs(r2_cv_medio_primer_alpha(X_tr, y_tr, kf) - ref0) > TOL:
                continue
            # Verificacion completa contra T2
            grid = grid_completo(X_tr, y_tr, kf)
            medias = [float(v) for v in grid.cv_results_["mean_test_score"]]
            coincide = (
                np.allclose(np.array(medias),
                            np.array(T2_R2_MEDIO_CV_POR_ALPHA),
                            rtol=0.0, atol=TOL)
                and int(grid.best_index_) == T2_INDICE_ALPHA_ELEGIDO
                and abs(float(grid.best_params_["ridge__alpha"])
                        - T2_ALPHA_ELEGIDO) <= 1e-12
            )
            if coincide:
                return {
                    "metodo": metodo,
                    "random_state": int(rs),
                    "grid": grid,
                    "medias_cv": medias,
                    "X_train": X_tr, "X_test": X_te,
                    "y_train": y_tr, "y_test": y_te,
                    "candidatos_probados": probados_total,
                }
    return None


def cargar_resultados_previos():
    """Carga resultados.json previo (si existe) para conservar sus claves."""
    try:
        with open("resultados.json", "r", encoding="utf-8") as f:
            contenido = json.load(f)
    except Exception:
        return {}
    if isinstance(contenido, dict):
        internos = contenido.get("resultados", contenido)
        if isinstance(internos, dict):
            return dict(internos)
    return {}


def main():
    datos = load_diabetes()
    X, y = datos.data, datos.target
    kf = KFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=CV_RANDOM_STATE)

    previos = cargar_resultados_previos()

    print("Parte 3 - Evaluacion en prueba (una sola vez)")
    print("Reconstruyendo la particion exacta de T2 (train 331 / test 111)...")

    hallazgo = buscar_particion_de_T2(X, y, kf)
    grid = None
    medias_cv = None

    if hallazgo is not None:
        split_ok = True
        metodo_split = hallazgo["metodo"]
        rs_split = hallazgo["random_state"]
        candidatos = hallazgo["candidatos_probados"]
        X_train, X_test = hallazgo["X_train"], hallazgo["X_test"]
        y_train, y_test = hallazgo["y_train"], hallazgo["y_test"]
        grid = hallazgo["grid"]
        medias_cv = hallazgo["medias_cv"]
        alpha_elegido = float(grid.best_params_["ridge__alpha"])
        indice_alpha = int(grid.best_index_)
        alpha_fuente = ("GridSearchCV re-ejecutado sobre el train de T2 "
                        "(particion y CV verificadas contra T2)")
        print("  Particion de T2 reproducida: metodo=%s, random_state=%d "
              "(%d candidatos probados)" % (metodo_split, rs_split, candidatos))
    else:
        # Fallback (indicado por el revisor): particion nueva 75/25 y alpha
        # tomado de los resultados de T2 (elegido alli solo con CV sobre train).
        split_ok = False
        metodo_split = "train_test_split"
        rs_split = 42
        candidatos = 0
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=rs_split
        )
        alpha_elegido = float(T2_ALPHA_ELEGIDO)
        indice_alpha = int(T2_INDICE_ALPHA_ELEGIDO)
        alpha_fuente = ("T2_alpha_elegido (fallback: particion de T2 no "
                        "reproducible; alpha elegido en T2 solo con CV sobre train)")
        print("  Aviso: no se reprodujo la particion de T2; se usa un split "
              "75/25 nuevo (random_state=42) y el alpha de T2.")

    print("Train: %d | Test: %d | alpha elegido: %.4f"
          % (X_train.shape[0], X_test.shape[0], alpha_elegido))

    # --- Ridge con el alpha elegido (mismo pipeline escalado) ---
    pipeline_ridge = hacer_pipeline_ridge(alpha_elegido)
    pipeline_ridge.fit(X_train, y_train)
    pred_ridge = pipeline_ridge.predict(X_test)
    ridge_r2 = float(r2_score(y_test, pred_ridge))
    ridge_rmse = float(math.sqrt(mean_squared_error(y_test, pred_ridge)))
    ridge_intercepto = float(pipeline_ridge.named_steps["ridge"].intercept_)
    ridge_coefs = [float(c) for c in pipeline_ridge.named_steps["ridge"].coef_]

    # --- LinearRegression sin regularizar (mismo escalado) ---
    pipeline_linreg = hacer_pipeline_linreg()
    pipeline_linreg.fit(X_train, y_train)
    pred_linreg = pipeline_linreg.predict(X_test)
    linreg_r2 = float(r2_score(y_test, pred_linreg))
    linreg_rmse = float(math.sqrt(mean_squared_error(y_test, pred_linreg)))
    linreg_intercepto = float(pipeline_linreg.named_steps["linreg"].intercept_)
    linreg_coefs = [float(c) for c in pipeline_linreg.named_steps["linreg"].coef_]

    # --- Cifras a guardar ---
    cifras = {
        "T3_dataset": "sklearn.datasets.load_diabetes",
        "T3_split_metodo": metodo_split,
        "T3_split_random_state": int(rs_split),
        "T3_split_reproducida_de_T2": bool(split_ok),
        "T3_split_candidatos_probados": int(candidatos),
        "T3_n_train": int(X_train.shape[0]),
        "T3_n_test": int(X_test.shape[0]),
        "T3_n_features": int(X.shape[1]),
        "T3_alpha_elegido": float(alpha_elegido),
        "T3_alpha_elegido_indice": int(indice_alpha),
        "T3_alpha_fuente": alpha_fuente,
        "T3_T2_alpha_elegido_referencia": float(T2_ALPHA_ELEGIDO),
        "T3_T2_r2_medio_cv_por_alpha_referencia": [
            float(v) for v in T2_R2_MEDIO_CV_POR_ALPHA
        ],
    }
    if grid is not None:
        cifras["T3_r2_medio_cv_alpha_elegido"] = float(grid.best_score_)
        cifras["T3_r2_medio_cv_por_alpha"] = [float(v) for v in medias_cv]
    cifras.update({
        "T3_ridge_r2_test": ridge_r2,
        "T3_ridge_rmse_test": ridge_rmse,
        "T3_ridge_intercepto": ridge_intercepto,
        "T3_ridge_coeficientes": ridge_coefs,
        "T3_linreg_r2_test": linreg_r2,
        "T3_linreg_rmse_test": linreg_rmse,
        "T3_linreg_intercepto": linreg_intercepto,
        "T3_linreg_coeficientes": linreg_coefs,
        "T3_diferencia_r2_ridge_menos_linreg": float(ridge_r2 - linreg_r2),
        "T3_diferencia_rmse_ridge_menos_linreg": float(ridge_rmse - linreg_rmse),
        "T3_mejora_rmse_ridge_vs_linreg": float(linreg_rmse - ridge_rmse),
    })

    salida = dict(previos)
    salida.update(cifras)
    with open("resultados.json", "w", encoding="utf-8") as f:
        json.dump({"resultados": salida}, f, indent=2, ensure_ascii=False)

    # --- Impresion de cifras principales (4 decimales) ---
    print("")
    if grid is not None:
        print("R2 medio de CV del alpha elegido (solo train): %.4f"
              % grid.best_score_)
    print("")
    print("Resultados sobre el conjunto de prueba (evaluados una sola vez):")
    print("  Ridge            (alpha=%.4f, StandardScaler+Ridge): "
          "R2=%.4f | RMSE=%.4f" % (alpha_elegido, ridge_r2, ridge_rmse))
    print("  LinearRegression (sin regularizar, StandardScaler+LinearRegression): "
          "R2=%.4f | RMSE=%.4f" % (linreg_r2, linreg_rmse))
    print("  Diferencia (Ridge - LinearRegression): dR2=%+.4f | dRMSE=%+.4f"
          % (ridge_r2 - linreg_r2, ridge_rmse - linreg_rmse))
    if ridge_rmse < linreg_rmse:
        print("  Ridge reduce el RMSE en prueba respecto a la regresion "
              "lineal sin regularizar.")
    elif ridge_rmse > linreg_rmse:
        print("  Ridge no mejora el RMSE en prueba respecto a la regresion "
              "lineal sin regularizar.")
    else:
        print("  Ridge y LinearRegression empatan en RMSE en prueba.")
    print("Cifras guardadas en resultados.json")


if __name__ == "__main__":
    main()
