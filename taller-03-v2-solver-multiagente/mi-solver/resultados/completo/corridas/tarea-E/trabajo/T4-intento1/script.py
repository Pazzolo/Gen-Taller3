# -*- coding: utf-8 -*-
"""
Subtarea T4 — Figura: R2 medio de validacion cruzada frente a alpha,
con el eje de alpha en escala logaritmica (semilogx).

El script es independiente: reutiliza las cifras de la Parte 2 (T2) guardadas
en resultados.json si estan disponibles (para que la figura coincida
exactamente con lo calculado en T2); si no existen, las recalcula desde cero
con el mismo procedimiento (load_diabetes, split 331/111, Pipeline
StandardScaler+Ridge, KFold(5, shuffle=True, random_state=7), scoring='r2',
grid numpy.logspace(-3, 3, 13)). El conjunto de prueba no se usa en ningun
momento.
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

# ----------------------------------------------------------------------
# 1) Cargar resultados previos si existen (cifras de T2)
# ----------------------------------------------------------------------
resultados = {}
if os.path.exists(ARCHIVO_RESULTADOS):
    try:
        with open(ARCHIVO_RESULTADOS, "r") as f:
            resultados = json.load(f)
    except (json.JSONDecodeError, OSError):
        resultados = {}

datos_t2_ok = (
    "T2_alphas_evaluados" in resultados
    and "T2_r2_medio_cv_por_alpha" in resultados
    and len(resultados["T2_alphas_evaluados"]) > 0
    and len(resultados["T2_alphas_evaluados"])
    == len(resultados["T2_r2_medio_cv_por_alpha"])
)

# ----------------------------------------------------------------------
# 2) Obtener alphas y R2 medios de CV (de T2 o recalculados)
# ----------------------------------------------------------------------
if datos_t2_ok:
    alphas = np.asarray(resultados["T2_alphas_evaluados"], dtype=float)
    r2_medios = np.asarray(resultados["T2_r2_medio_cv_por_alpha"], dtype=float)
    if "T2_r2_std_cv_por_alpha" in resultados and len(
        resultados["T2_r2_std_cv_por_alpha"]
    ) == len(alphas):
        r2_stds = np.asarray(resultados["T2_r2_std_cv_por_alpha"], dtype=float)
    else:
        r2_stds = np.zeros_like(r2_medios)
    fuente = "cifras de la Parte 2 (T2) leidas de resultados.json"
else:
    # Recalculo independiente con el mismo procedimiento de la Parte 2.
    # El test se reserva pero no se usa aqui para nada.
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=111, random_state=42
    )  # 331 train / 111 test; el test queda intacto
    alphas = np.logspace(-3, 3, 13)
    cv = KFold(n_splits=5, shuffle=True, random_state=7)
    r2_medios, r2_stds = [], []
    for a in alphas:
        pipe = Pipeline(
            [("scaler", StandardScaler()), ("ridge", Ridge(alpha=float(a)))]
        )
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="r2")
        r2_medios.append(float(scores.mean()))
        r2_stds.append(float(scores.std()))
    alphas = np.asarray(alphas, dtype=float)
    r2_medios = np.asarray(r2_medios, dtype=float)
    r2_stds = np.asarray(r2_stds, dtype=float)
    fuente = (
        "recalculado en T4: load_diabetes, train=331/test=111, "
        "Pipeline(StandardScaler+Ridge), KFold(5, shuffle=True, "
        "random_state=7), scoring='r2', grid logspace(-3, 3, 13)"
    )

# ----------------------------------------------------------------------
# 3) Mejor alpha segun el R2 medio de CV (solo con cifras de CV en train)
# ----------------------------------------------------------------------
idx_mejor = int(np.argmax(r2_medios))
alpha_elegido = float(alphas[idx_mejor])
r2_max = float(r2_medios[idx_mejor])

# ----------------------------------------------------------------------
# 4) Figura: semilogx de R2 medio de CV frente a alpha
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogx(
    alphas, r2_medios, marker="o", color="tab:blue",
    label="R2 medio de CV (5-fold)",
)
ax.axvline(
    alpha_elegido, color="tab:red", linestyle="--",
    label="alpha elegido = {:.4g}".format(alpha_elegido),
)
ax.scatter([alpha_elegido], [r2_max], color="tab:red", zorder=5)
ax.set_xlabel("alpha (escala logaritmica)")
ax.set_ylabel("R2 medio de validacion cruzada")
ax.set_title("Ridge: R2 medio de CV frente a alpha (solo train, test intacto)")
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="best")
fig.tight_layout()
fig.savefig(ARCHIVO_FIGURA, dpi=150, bbox_inches="tight")
plt.show()  # la figura queda visible como salida del notebook
plt.close(fig)

# ----------------------------------------------------------------------
# 5) Guardar todas las cifras en resultados.json (sin borrar las previas)
# ----------------------------------------------------------------------
resultados["T4_figura_archivo"] = ARCHIVO_FIGURA
resultados["T4_figura_tipo"] = "semilogx (eje x en escala logaritmica)"
resultados["T4_fuente_datos_figura"] = fuente
resultados["T4_alphas_evaluados"] = [float(a) for a in alphas]
resultados["T4_r2_medio_cv_por_alpha"] = [float(v) for v in r2_medios]
resultados["T4_r2_std_cv_por_alpha"] = [float(v) for v in r2_stds]
resultados["T4_n_puntos_alpha"] = int(len(alphas))
resultados["T4_alpha_min"] = float(alphas.min())
resultados["T4_alpha_max"] = float(alphas.max())
resultados["T4_indice_alpha_elegido"] = idx_mejor
resultados["T4_alpha_elegido"] = alpha_elegido
resultados["T4_r2_medio_cv_maximo"] = r2_max

with open(ARCHIVO_RESULTADOS, "w") as f:
    json.dump(resultados, f, indent=2)

# ----------------------------------------------------------------------
# 6) Salida de texto con las cifras principales (4 decimales)
# ----------------------------------------------------------------------
print("Parte 4 — Figura: R2 medio de validacion cruzada frente a alpha")
print("Fuente de las cifras: {}".format(fuente))
print(
    "Puntos dibujados: {} alphas, de {:.4g} a {:.4g} (eje x logaritmico)".format(
        len(alphas), alphas.min(), alphas.max()
    )
)
print()
print("R2 medio de CV por alpha (lo que se dibuja):")
for a, m in zip(alphas, r2_medios):
    marca = "  <-- mejor" if float(a) == alpha_elegido else ""
    print("  alpha={:10.4f} | R2 CV medio={:.4f}{}".format(float(a), float(m), marca))
print()
print("Mejor alpha: {:.4f} | R2 medio de CV maximo: {:.4f}".format(alpha_elegido, r2_max))
print("Figura guardada como '{}' (semilogx, eje x en escala logaritmica)".format(ARCHIVO_FIGURA))
print("Cifras guardadas en resultados.json")
