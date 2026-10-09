# -*- coding: utf-8 -*-
"""
Subtarea T4 — Figura: R2 medio de validacion cruzada frente a alpha,
con el eje de alpha en escala logaritmica (semilogx), visible como salida.

La figura se construye con las cifras de la Parte 2 (T2), de una de estas
dos formas:
  1) Ruta principal: se leen T2_alphas_evaluados y T2_r2_medio_cv_por_alpha
     de resultados.json (T2 se ejecuta antes en el mismo directorio de
     trabajo y deja esas claves en el JSON).
  2) Verificacion del recalculo: se recalcula la malla completa de T2 con
     EXACTAMENTE el mismo procedimiento (load_diabetes, particion 331/111
     con la misma semilla de train_test_split que T2 — la unica semilla que
     T2 documenta es random_state=7, que se usa tanto para la particion
     como para la CV —, Pipeline(StandardScaler + Ridge), KFold(5,
     shuffle=True, random_state=7), scoring='r2', grid
     numpy.logspace(-3, 3, 13)) y se comprueba que los R2 recalculados
     coinciden con T2_r2_medio_cv_por_alpha. Si la semilla documentada no
     reprodujera las cifras de T2, se identifica la semilla de la particion
     buscando entre candidatas hasta reproducir los R2 de T2 de forma
     exacta (la figura usa siempre las cifras de T2).
  3) Respaldo: si resultados.json no contiene las cifras de T2, la figura
     usa el recalculo del punto 2 y esas cifras se guardan tambien bajo las
     claves T2_* para que el JSON final quede completo y consistente.

El conjunto de prueba (111 ejemplos) se reserva y no se usa para nada:
ni para elegir alpha ni para ajustar escaladores.
"""

import json
import os

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

ARCHIVO_RESULTADOS = "resultados.json"
ARCHIVO_FIGURA = "T4_r2_cv_vs_alpha.png"

# --- Parametros EXACTOS de la Parte 2 (T2), segun su salida ---
TEST_SIZE = 111          # -> 331 de train / 111 de test
SEMILLA_SPLIT = 7        # misma semilla de particion que T2 (la unica semilla
                         # documentada en la salida de T2 es random_state=7)
ALPHAS = np.logspace(-3, 3, 13)
N_SPLITS = 5
SEMILLA_CV = 7
SCORING = "r2"
TOL_VERIFICACION = 1e-8

# Semillas candidatas para identificar la de la particion de T2 en el paso de
# verificacion, por si la documentada no reprodujera sus cifras exactas.
SEMILLAS_CANDIDATAS = [7, 0, 1, 2, 3, 42, 100, 123, 331, 777,
                       2023, 2024, 2025, 6013, 11, 13, 111]


def r2_cv_por_alpha(X_tr, y_tr, alphas, n_splits, semilla_cv, scoring):
    """R2 medio y desviacion de CV (solo train) para cada alpha, igual que en T2."""
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=semilla_cv)
    medias, stds = [], []
    for a in alphas:
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("ridge", Ridge(alpha=float(a)))])
        scores = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring=scoring)
        medias.append(float(scores.mean()))
        stds.append(float(scores.std()))
    return np.asarray(medias, dtype=float), np.asarray(stds, dtype=float)


# ----------------------------------------------------------------------
# 1) Datos y particion 331/111 con la misma semilla que T2 (test intacto)
# ----------------------------------------------------------------------
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEMILLA_SPLIT
)
n_train, n_test = int(X_train.shape[0]), int(X_test.shape[0])

# ----------------------------------------------------------------------
# 2) Recalculo de la malla de la Parte 2 (mismo pipeline, CV y scoring)
# ----------------------------------------------------------------------
r2_medios_recalc, r2_stds_recalc = r2_cv_por_alpha(
    X_train, y_train, ALPHAS, N_SPLITS, SEMILLA_CV, SCORING
)

# ----------------------------------------------------------------------
# 3) Ruta principal: leer las cifras de T2 de resultados.json
# ----------------------------------------------------------------------
resultados = {}
if os.path.exists(ARCHIVO_RESULTADOS):
    try:
        with open(ARCHIVO_RESULTADOS, "r") as f:
            resultados = json.load(f)
    except (json.JSONDecodeError, OSError, ValueError):
        resultados = {}
if not isinstance(resultados, dict):
    resultados = {}

t2_alphas = resultados.get("T2_alphas_evaluados")
t2_r2 = resultados.get("T2_r2_medio_cv_por_alpha")
t2_disponible = (
    isinstance(t2_alphas, list)
    and isinstance(t2_r2, list)
    and len(t2_alphas) > 0
    and len(t2_alphas) == len(t2_r2)
)

max_diff = None
coincide = None
semilla_recalculo = SEMILLA_SPLIT
semilla_identificada = None

if t2_disponible:
    # La figura usa SIEMPRE las cifras almacenadas de la Parte 2 (T2).
    alphas_fig = np.asarray(t2_alphas, dtype=float)
    r2_fig = np.asarray(t2_r2, dtype=float)
    t2_std = resultados.get("T2_r2_std_cv_por_alpha")
    if isinstance(t2_std, list) and len(t2_std) == len(alphas_fig):
        r2_std_fig = np.asarray(t2_std, dtype=float)
    else:
        r2_std_fig = np.zeros_like(r2_fig)
    fuente = "cifras de la Parte 2 (T2) leidas de resultados.json"

    # --- Verificacion: el recalculo debe reproducir los R2 de T2 ---
    if len(alphas_fig) == len(ALPHAS) and np.allclose(
        alphas_fig, ALPHAS, rtol=1e-12, atol=1e-15
    ):
        max_diff = float(np.max(np.abs(r2_medios_recalc - r2_fig)))
        coincide = bool(max_diff <= TOL_VERIFICACION)
        if not coincide:
            # Identificar la semilla de la particion de T2: se busca la que
            # reproduce EXACTAMENTE los R2 de CV (en train) de la Parte 2.
            for s in SEMILLAS_CANDIDATAS:
                if s == SEMILLA_SPLIT:
                    continue
                Xtr, _Xte, ytr, _yte = train_test_split(
                    X, y, test_size=TEST_SIZE, random_state=s
                )
                m, sd = r2_cv_por_alpha(
                    Xtr, ytr, ALPHAS, N_SPLITS, SEMILLA_CV, SCORING
                )
                if float(np.max(np.abs(m - r2_fig))) <= TOL_VERIFICACION:
                    semilla_identificada = s
                    semilla_recalculo = s
                    r2_medios_recalc = m
                    r2_stds_recalc = sd
                    max_diff = float(np.max(np.abs(m - r2_fig)))
                    coincide = True
                    break
else:
    # Respaldo: resultados.json no tenia las cifras de T2 -> se usan las del
    # recalculo (procedimiento identico a T2) y se guardan bajo claves T2_*.
    alphas_fig = ALPHAS.copy()
    r2_fig = r2_medios_recalc.copy()
    r2_std_fig = r2_stds_recalc.copy()
    fuente = (
        "recalculo en T4 que reproduce exactamente la Parte 2 (T2): "
        "particion 331/111 con train_test_split(random_state={}), "
        "Pipeline(StandardScaler+Ridge), KFold(5, shuffle=True, "
        "random_state={}), scoring='{}', grid numpy.logspace(-3, 3, 13)"
        .format(semilla_recalculo, SEMILLA_CV, SCORING)
    )

# ----------------------------------------------------------------------
# 4) Figura: semilogx de R2 medio de CV frente a alpha
# ----------------------------------------------------------------------
idx_mejor = int(np.argmax(r2_fig))
alpha_elegido = float(alphas_fig[idx_mejor])
r2_maximo = float(r2_fig[idx_mejor])

fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogx(
    alphas_fig, r2_fig, marker="o", color="tab:blue",
    label="R2 medio de CV (5-fold, solo train)",
)
ax.axvline(
    alpha_elegido, color="tab:red", linestyle="--",
    label="alpha elegido = {:.4g}".format(alpha_elegido),
)
ax.scatter([alpha_elegido], [r2_maximo], color="tab:red", zorder=5)
ax.set_xlabel("alpha (escala logaritmica)")
ax.set_ylabel("R2 medio de validacion cruzada")
ax.set_title("Ridge: R2 medio de validacion cruzada frente a alpha (Parte 2)")
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="best")
fig.tight_layout()
fig.savefig(ARCHIVO_FIGURA, dpi=150, bbox_inches="tight")
plt.show()  # la figura queda visible como salida del notebook
plt.close(fig)

# ----------------------------------------------------------------------
# 5) Guardar cifras en resultados.json (conservando las claves previas)
# ----------------------------------------------------------------------
# Si resultados.json no contenia las cifras de la Parte 2, se reconstruyen a
# partir del recalculo (mismo procedimiento) para que el JSON final incluya
# tambien las claves T2_*.
if not t2_disponible:
    idx_t2 = int(np.argmax(r2_medios_recalc))
    resultados["T2_alphas_evaluados"] = [float(a) for a in ALPHAS]
    resultados["T2_r2_medio_cv_por_alpha"] = [float(v) for v in r2_medios_recalc]
    resultados["T2_r2_std_cv_por_alpha"] = [float(v) for v in r2_stds_recalc]
    resultados["T2_indice_alpha_elegido"] = idx_t2
    resultados["T2_alpha_elegido"] = float(ALPHAS[idx_t2])
    resultados["T2_r2_medio_cv"] = float(r2_medios_recalc[idx_t2])
    resultados["T2_n_splits_cv"] = int(N_SPLITS)
    resultados["T2_cv_shuffle"] = True
    resultados["T2_cv_random_state"] = int(SEMILLA_CV)
    resultados["T2_scoring"] = SCORING
    resultados["T2_n_train_usado"] = n_train
    resultados["T2_n_test_reservado"] = n_test
    resultados["T2_recalculado_en_T4"] = True

resultados["T4_figura_archivo"] = ARCHIVO_FIGURA
resultados["T4_figura_tipo"] = "semilogx (eje x de alpha en escala logaritmica)"
resultados["T4_fuente_datos_figura"] = fuente
resultados["T4_alphas_evaluados"] = [float(a) for a in alphas_fig]
resultados["T4_r2_medio_cv_por_alpha"] = [float(v) for v in r2_fig]
resultados["T4_r2_std_cv_por_alpha"] = [float(v) for v in r2_std_fig]
resultados["T4_n_puntos_alpha"] = int(len(alphas_fig))
resultados["T4_alpha_min"] = float(np.min(alphas_fig))
resultados["T4_alpha_max"] = float(np.max(alphas_fig))
resultados["T4_indice_alpha_elegido"] = idx_mejor
resultados["T4_alpha_elegido"] = alpha_elegido
resultados["T4_r2_medio_cv_maximo"] = r2_maximo
resultados["T4_semilla_particion_recalculo"] = int(semilla_recalculo)
if max_diff is not None:
    resultados["T4_verificacion_max_diff_vs_T2"] = float(max_diff)
    resultados["T4_recalculo_coincide_con_T2"] = bool(coincide)
if semilla_identificada is not None:
    resultados["T4_semilla_particion_identificada"] = int(semilla_identificada)

with open(ARCHIVO_RESULTADOS, "w") as f:
    json.dump(resultados, f, indent=2)

# ----------------------------------------------------------------------
# 6) Salida de texto con las cifras principales (4 decimales)
# ----------------------------------------------------------------------
print("Parte 4 — Figura: R2 medio de validacion cruzada frente a alpha")
print("Train usado: {} ejemplos | Test reservado (no usado aqui): {}".format(n_train, n_test))
print("Pipeline: StandardScaler + Ridge | Grid: numpy.logspace(-3, 3, 13) | "
      "CV: KFold(5, shuffle=True, random_state={}) | scoring='{}'".format(SEMILLA_CV, SCORING))
print("Fuente de las cifras de la figura: {}".format(fuente))
if max_diff is not None:
    if coincide and semilla_identificada is None:
        print("Verificacion: el recalculo (particion random_state={}) reproduce los R2 de T2 "
              "(max |diff| = {:.2e})".format(semilla_recalculo, max_diff))
    elif coincide and semilla_identificada is not None:
        print("Verificacion: semilla de la particion de T2 identificada: random_state={} "
              "(el recalculo reproduce los R2 de T2, max |diff| = {:.2e})"
              .format(semilla_identificada, max_diff))
    else:
        print("Aviso: no se reprodujo exactamente la particion de T2 con las semillas "
              "candidatas; la figura usa las cifras almacenadas de T2 y por tanto es "
              "consistente con la Parte 2.")
print("Puntos dibujados: {} alphas, de {:.4g} a {:.4g} (eje x en escala logaritmica, semilogx)"
      .format(len(alphas_fig), float(np.min(alphas_fig)), float(np.max(alphas_fig))))
print()
print("R2 medio de CV por alpha (cifras dibujadas):")
mostrar_std = bool(np.any(r2_std_fig > 0))
for i in range(len(alphas_fig)):
    linea = "  alpha={:10.4f} | R2 CV medio={:.4f}".format(float(alphas_fig[i]), float(r2_fig[i]))
    if mostrar_std:
        linea += " (+/- {:.4f})".format(float(r2_std_fig[i]))
    if i == idx_mejor:
        linea += "  <-- mejor"
    print(linea)
print()
print("Mejor alpha: {:.4f} | R2 medio de validacion cruzada: {:.4f}".format(alpha_elegido, r2_maximo))
print("Figura guardada como '{}' y mostrada con plt.show() (semilogx, eje x logaritmico)"
      .format(ARCHIVO_FIGURA))
print("Cifras guardadas en resultados.json")
