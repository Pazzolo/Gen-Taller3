# -*- coding: utf-8 -*-
"""
T3 — Curva de aprendizaje: GaussianNB (generativo) contra regresión logística
(discriminativo) sobre Breast Cancer (Diagnostic).

Recrea de forma determinista la división 70/30 estratificada de la Parte 1
(test_size=0.30, stratify=y, random_state=42), genera submuestras
estratificadas del entrenamiento con el 5 %, 10 %, 25 %, 50 % y 100 %
(random_state=42), entrena ambos modelos en cada submuestra (el StandardScaler
se ajusta en cada submuestra y el conjunto de prueba solo se transforma) y
evalúa la exactitud siempre sobre el mismo conjunto de prueba. Guarda la
figura curva_aprendizaje_T3.png y todas las cifras en resultados.json.
"""

import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

ARCHIVO_RESULTADOS = "resultados.json"
ARCHIVO_FIGURA = "curva_aprendizaje_T3.png"
RANDOM_STATE = 42
FRACCIONES = [0.05, 0.10, 0.25, 0.50, 1.00]


def main():
    print("=== T3: Curva de aprendizaje (GaussianNB vs regresión logística) ===")

    # ------------------------------------------------------------------
    # 1) Datos y división de la Parte 1 (determinista: 70/30 estratificada)
    # ------------------------------------------------------------------
    datos = load_breast_cancer()
    X, y = datos.data, datos.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    n_train_total = int(X_train.shape[0])
    n_prueba = int(X_test.shape[0])
    print(
        f"División de la Parte 1 recreada: {n_train_total} entrenamiento / "
        f"{n_prueba} prueba (test_size=0.30, estratificada, random_state={RANDOM_STATE})"
    )

    # ------------------------------------------------------------------
    # 2) Submuestras estratificadas del entrenamiento + entrenamiento/evaluación
    # ------------------------------------------------------------------
    tamanos = []   # número real de ejemplos de cada submuestra
    acc_nb = []    # exactitud de GaussianNB en la prueba fija
    acc_lr = []    # exactitud de la regresión logística en la prueba fija
    detalle = {}

    print(
        "\nExactitud medida SIEMPRE sobre el mismo conjunto de prueba "
        f"({n_prueba} ejemplos):"
    )
    for frac in FRACCIONES:
        if frac >= 1.0:
            # 100 %: la submuestra es el entrenamiento completo
            X_sub, y_sub = X_train, y_train
        else:
            # Submuestra estratificada del entrenamiento (random_state=42)
            X_sub, _, y_sub, _ = train_test_split(
                X_train,
                y_train,
                train_size=frac,
                stratify=y_train,
                random_state=RANDOM_STATE,
            )
        n_sub = int(X_sub.shape[0])
        conteo = np.bincount(np.asarray(y_sub, dtype=int), minlength=2)

        # Escalador ajustado SOLO con la submuestra; la prueba solo se transforma
        escalador = StandardScaler().fit(X_sub)
        X_sub_esc = escalador.transform(X_sub)
        X_test_esc = escalador.transform(X_test)

        # Modelo generativo
        nb = GaussianNB()
        nb.fit(X_sub_esc, y_sub)
        a_nb = float(accuracy_score(y_test, nb.predict(X_test_esc)))

        # Modelo discriminativo
        lr = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)
        lr.fit(X_sub_esc, y_sub)
        a_lr = float(accuracy_score(y_test, lr.predict(X_test_esc)))

        tamanos.append(n_sub)
        acc_nb.append(a_nb)
        acc_lr.append(a_lr)
        pct = int(round(frac * 100))
        detalle[f"submuestra_{pct}_pct"] = {
            "fraccion_solicitada": float(frac),
            "n_ejemplos_entrenamiento": float(n_sub),
            "casos_por_clase_submuestra": {
                "malignant": float(conteo[0]),
                "benign": float(conteo[1]),
            },
            "accuracy_gaussian_nb": a_nb,
            "accuracy_regresion_logistica": a_lr,
            "ventaja_regresion_logistica_menos_nb": float(a_lr - a_nb),
        }
        print(
            f"  {pct:>3}% del entrenamiento (n={n_sub:>3}): "
            f"GaussianNB={a_nb:.4f} | RegLog={a_lr:.4f}"
        )

    # Resumen de qué modelo alcanza mayor exactitud en cada tamaño (calculado)
    print("\nModelo con mayor exactitud en cada tamaño:")
    for frac, n, a_nb_i, a_lr_i in zip(FRACCIONES, tamanos, acc_nb, acc_lr):
        if a_nb_i > a_lr_i:
            mejor = "GaussianNB"
        elif a_lr_i > a_nb_i:
            mejor = "Regresión logística"
        else:
            mejor = "empate"
        print(f"  {int(round(frac * 100)):>3}% (n={n:>3}): {mejor}")

    # ------------------------------------------------------------------
    # 3) Figura: exactitud contra número de ejemplos de entrenamiento
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    ax.plot(
        tamanos, acc_nb, marker="o", color="tab:blue",
        label="GaussianNB (generativo)",
    )
    ax.plot(
        tamanos, acc_lr, marker="s", color="tab:red",
        label="Regresión logística (discriminativo)",
    )
    for n, a in zip(tamanos, acc_nb):
        ax.annotate(
            f"{a:.3f}", (n, a), textcoords="offset points", xytext=(0, -13),
            ha="center", fontsize=8, color="tab:blue",
        )
    for n, a in zip(tamanos, acc_lr):
        ax.annotate(
            f"{a:.3f}", (n, a), textcoords="offset points", xytext=(0, 7),
            ha="center", fontsize=8, color="tab:red",
        )
    ax.set_xticks(tamanos)
    ax.set_xlabel("Número de ejemplos de entrenamiento")
    ax.set_ylabel("Exactitud (accuracy) en el conjunto de prueba")
    ax.set_title(
        "Curva de aprendizaje: GaussianNB vs regresión logística\n"
        f"Breast Cancer (Diagnostic) — prueba fija de {n_prueba} ejemplos"
    )
    y_min = min(min(acc_nb), min(acc_lr))
    ax.set_ylim(max(0.0, y_min - 0.08), 1.01)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(ARCHIVO_FIGURA, dpi=150)
    plt.close(fig)
    print(f"\nFigura guardada: {ARCHIVO_FIGURA}")

    # ------------------------------------------------------------------
    # 4) Todas las cifras calculadas -> resultados.json
    # ------------------------------------------------------------------
    bloque_t3 = {
        "division_entrenamiento_prueba": {
            "test_size": 0.30,
            "estratificada": True,
            "random_state": RANDOM_STATE,
            "n_ejemplos_entrenamiento": float(n_train_total),
            "n_ejemplos_prueba": float(n_prueba),
        },
        "fracciones_evaluadas": [float(f) for f in FRACCIONES],
        "tamanos_submuestra": [float(t) for t in tamanos],
        "accuracy_gaussian_nb_por_tamano": [float(a) for a in acc_nb],
        "accuracy_regresion_logistica_por_tamano": [float(a) for a in acc_lr],
        "detalle_por_submuestra": detalle,
        "accuracy_gaussian_nb_100_pct": float(acc_nb[-1]),
        "accuracy_regresion_logistica_100_pct": float(acc_lr[-1]),
        "figura": ARCHIVO_FIGURA,
        "parametros": {
            "submuestreo": (
                "train_test_split(X_train, y_train, train_size=frac, "
                "stratify=y_train, random_state=42); el 100% usa el "
                "entrenamiento completo"
            ),
            "escalador": (
                "StandardScaler ajustado en cada submuestra; el conjunto de "
                "prueba solo se transforma (nunca se ajusta con él)"
            ),
            "gaussian_nb": "GaussianNB() (valores por defecto)",
            "regresion_logistica": "LogisticRegression(max_iter=5000, random_state=42)",
        },
    }

    # Se conserva cualquier bloque previo (p. ej. T1) y se añade/actualiza T3
    try:
        with open(ARCHIVO_RESULTADOS, "r", encoding="utf-8") as f:
            previo = json.load(f)
        if not isinstance(previo, dict):
            previo = {}
    except (OSError, ValueError):
        previo = {}

    resultados_previos = previo.get("resultados")
    if not isinstance(resultados_previos, dict):
        resultados_previos = {}
    resultados_previos["T3"] = bloque_t3
    previo["resultados"] = resultados_previos

    with open(ARCHIVO_RESULTADOS, "w", encoding="utf-8") as f:
        json.dump(previo, f, ensure_ascii=False, indent=2)
    print(f"Cifras escritas en {ARCHIVO_RESULTADOS}")


if __name__ == "__main__":
    main()
