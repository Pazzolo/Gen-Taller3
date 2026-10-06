# -*- coding: utf-8 -*-
"""
T3 — Curva de aprendizaje: modelo generativo (GaussianNB) contra modelo
discriminativo (Regresión logística) sobre Breast Cancer Wisconsin (Diagnostic).

Pasos:
1) Recalcula la división 70/30 estratificada de T1 (random_state=42): es la
   única división usada en toda la tarea; el conjunto de prueba no se toca
   para ajustar nada.
2) Para cada tamaño (5 %, 10 %, 25 %, 50 %, 100 % del entrenamiento) extrae
   una submuestra ESTRATIFICADA del conjunto de entrenamiento
   (random_state=42).
3) Entrena ambos modelos con cada submuestra y evalúa SIEMPRE sobre el mismo
   conjunto de prueba.
4) Dibuja la exactitud contra el número de ejemplos de entrenamiento (una
   curva por modelo) y guarda la figura como curva_aprendizaje.png.
5) Guarda todas las cifras calculadas en resultados.json. Las fracciones de
   entrenamiento se registran en PORCENTAJE (5.0, 10.0, ..., 100.0) para que
   no se confundan con métricas de exactitud (que viven en [0, 1]).
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.3
# Tamaños de entrenamiento en PORCENTAJE (así se guardan en el JSON);
# la proporción interna se deriva dividiendo entre 100.
FRACCIONES_PCT = [5.0, 10.0, 25.0, 50.0, 100.0]
ETIQUETAS = ["5pct", "10pct", "25pct", "50pct", "100pct"]
ARCHIVO_RESULTADOS = "resultados.json"
ARCHIVO_FIGURA = "curva_aprendizaje.png"


def cargar_resultados_previos(ruta):
    """Devuelve el dict previo de resultados.json (si existe y es válido)."""
    if os.path.exists(ruta):
        try:
            with open(ruta, "r", encoding="utf-8") as f:
                datos = json.load(f)
            if isinstance(datos, dict):
                return datos
        except (OSError, ValueError):
            pass
    return {}


def main():
    # ------------------------------------------------------------------
    # 1) Datos y división de T1 (se recalcula de forma idéntica)
    # ------------------------------------------------------------------
    datos = load_breast_cancer()
    X, y = datos.data, datos.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    n_train_total = int(X_train.shape[0])
    n_prueba = int(X_test.shape[0])

    # ------------------------------------------------------------------
    # 2-3) Submuestras estratificadas del entrenamiento, entrenamiento
    #      y evaluación sobre el MISMO conjunto de prueba
    # ------------------------------------------------------------------
    tamanos = []        # número real de ejemplos de cada submuestra
    n_clase0_sub = []
    n_clase1_sub = []
    acc_nb = []         # exactitud del modelo generativo (GaussianNB)
    acc_lr = []         # exactitud del modelo discriminativo (Reg. logística)

    print("=== T3: Curva de aprendizaje (generativo vs discriminativo) ===")
    print("Datos: Breast Cancer Wisconsin (Diagnostic)")
    print("Division 70/30 estratificada, random_state=%d (misma que T1)" % RANDOM_STATE)
    print("Entrenamiento: %d ejemplos | Prueba: %d ejemplos (fija en todas las evaluaciones)"
          % (n_train_total, n_prueba))
    print()
    encabezado = "%9s | %8s | %14s | %14s | %12s" % (
        "tamano", "n_train", "acc GaussNB", "acc RegLog", "dif RL-NB")
    print(encabezado)
    print("-" * len(encabezado))

    for pct in FRACCIONES_PCT:
        frac = pct / 100.0
        if frac >= 1.0:
            X_sub, y_sub = X_train, y_train
        else:
            X_sub, _, y_sub, _ = train_test_split(
                X_train, y_train,
                train_size=frac,
                stratify=y_train,
                random_state=RANDOM_STATE,
            )

        n_sub = int(X_sub.shape[0])
        tamanos.append(n_sub)
        n_clase0_sub.append(int(np.sum(y_sub == 0)))
        n_clase1_sub.append(int(np.sum(y_sub == 1)))

        # --- Modelo generativo: Naive Bayes gaussiano ---
        nb = GaussianNB()
        nb.fit(X_sub, y_sub)
        a_nb = float(accuracy_score(y_test, nb.predict(X_test)))

        # --- Modelo discriminativo: regresión logística ---
        # (el escalador se ajusta SOLO con la submuestra de entrenamiento,
        #  nunca con el conjunto de prueba)
        lr = Pipeline([
            ("escalador", StandardScaler()),
            ("modelo", LogisticRegression(max_iter=10000)),
        ])
        lr.fit(X_sub, y_sub)
        a_lr = float(accuracy_score(y_test, lr.predict(X_test)))

        acc_nb.append(a_nb)
        acc_lr.append(a_lr)

        print("%8.0f%% | %8d | %14.4f | %14.4f | %12.4f"
              % (pct, n_sub, a_nb, a_lr, a_lr - a_nb))

    # ------------------------------------------------------------------
    # Resumen impreso (4 decimales)
    # ------------------------------------------------------------------
    print()
    print("Mejor modelo por tamano de entrenamiento (exactitud en prueba):")
    ganadores = []
    for et, n, a_nb, a_lr in zip(ETIQUETAS, tamanos, acc_nb, acc_lr):
        if a_lr > a_nb:
            g = "RegresionLogistica"
        elif a_nb > a_lr:
            g = "GaussianNB"
        else:
            g = "empate"
        ganadores.append(g)
        print("  n=%3d (%s): %s (NB=%.4f, RL=%.4f)" % (n, et, g, a_nb, a_lr))

    # ------------------------------------------------------------------
    # 4) Figura: exactitud vs número de ejemplos de entrenamiento
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    ax.plot(tamanos, acc_nb, "o-", color="tab:blue",
            label="GaussianNB (generativo)")
    ax.plot(tamanos, acc_lr, "s-", color="tab:red",
            label="Regresion logistica (discriminativo)")
    for n, a_nb, a_lr in zip(tamanos, acc_nb, acc_lr):
        if a_nb <= a_lr:
            off_nb, off_lr = (0, -16), (0, 8)
        else:
            off_nb, off_lr = (0, 8), (0, -16)
        ax.annotate("%.3f" % a_nb, (n, a_nb), textcoords="offset points",
                    xytext=off_nb, ha="center", fontsize=8, color="tab:blue")
        ax.annotate("%.3f" % a_lr, (n, a_lr), textcoords="offset points",
                    xytext=off_lr, ha="center", fontsize=8, color="tab:red")
    ax.set_xlabel("Numero de ejemplos de entrenamiento")
    ax.set_ylabel("Exactitud en el conjunto de prueba")
    ax.set_title("Curva de aprendizaje: generativo vs discriminativo\n"
                 "(Breast Cancer Wisconsin; prueba fija con n=%d)" % n_prueba)
    ax.set_xticks(tamanos)
    y_min = min(min(acc_nb), min(acc_lr))
    ax.set_ylim(max(0.0, y_min - 0.07), 1.01)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(ARCHIVO_FIGURA, dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 5) Todas las cifras calculadas -> resultados.json
    #    (las fracciones se guardan como PORCENTAJES: 5.0 ... 100.0,
    #     para no confundirlas con métricas de exactitud en [0, 1])
    # ------------------------------------------------------------------
    cifras = {
        "t3_fracciones_pct": [float(p) for p in FRACCIONES_PCT],
        "t3_fracciones_unidad": "porcentaje_del_conjunto_de_entrenamiento",
        "t3_random_state_division": float(RANDOM_STATE),
        "t3_test_size": float(TEST_SIZE),
        "t3_random_state_submuestra": float(RANDOM_STATE),
        "t3_n_ejemplos_entrenamiento_total": float(n_train_total),
        "t3_n_ejemplos_prueba": float(n_prueba),
        "t3_n_ejemplos_submuestra": [float(n) for n in tamanos],
        "t3_n_clase0_submuestra": [float(v) for v in n_clase0_sub],
        "t3_n_clase1_submuestra": [float(v) for v in n_clase1_sub],
        "t3_acc_gaussiannb": [float(a) for a in acc_nb],
        "t3_acc_logreg": [float(a) for a in acc_lr],
        "t3_dif_acc_logreg_menos_gaussiannb": [float(a - b) for a, b in zip(acc_lr, acc_nb)],
        "t3_modelo_ganador_por_tamano": ganadores,
        "t3_figura": ARCHIVO_FIGURA,
    }
    for et, n, a_nb, a_lr in zip(ETIQUETAS, tamanos, acc_nb, acc_lr):
        cifras["t3_n_train_" + et] = float(n)
        cifras["t3_acc_gaussiannb_" + et] = float(a_nb)
        cifras["t3_acc_logreg_" + et] = float(a_lr)
        cifras["t3_dif_acc_logreg_menos_gaussiannb_" + et] = float(a_lr - a_nb)

    previos = cargar_resultados_previos(ARCHIVO_RESULTADOS)
    previos.update(cifras)
    with open(ARCHIVO_RESULTADOS, "w", encoding="utf-8") as f:
        json.dump(previos, f, ensure_ascii=False, indent=2)

    print()
    print("Figura guardada en: %s" % ARCHIVO_FIGURA)
    print("Cifras guardadas en: %s" % ARCHIVO_RESULTADOS)


if __name__ == "__main__":
    main()
