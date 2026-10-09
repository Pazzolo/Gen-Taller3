# -*- coding: utf-8 -*-
"""
T3 — Curva de aprendizaje sobre la misma división de T1.

- Misma división que T1: load_breast_cancer, test_size=0.3, estratificada, random_state=42.
- Submuestras ESTRATIFICADAS del entrenamiento: 5%, 10%, 25%, 50% y 100% (random_state=42).
- Modelos: GaussianNB y LogisticRegression (StandardScaler ajustado en CADA submuestra).
- Evaluación: accuracy SIEMPRE sobre el mismo conjunto de prueba de T1 (el escalador
  nunca se ajusta con el test).
- Figura: exactitud vs número de ejemplos de entrenamiento (una curva por modelo) -> PNG.
"""

import json

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

RANDOM_STATE = 42
FRACCIONES = [0.05, 0.10, 0.25, 0.50, 1.00]
FIGURA_PNG = "t3_curva_aprendizaje.png"

# Prefijos de los índices de la división de T1 (verificación de reproducibilidad)
T1_TRAIN_PREFIX = [469, 561, 443, 362, 271, 538, 510, 517, 558, 493,
                   369, 28, 432, 270, 312, 39, 94, 165, 79, 174]
T1_TEST_PREFIX = [14, 150, 275, 83, 86, 509, 140, 199, 292, 122,
                  354, 102, 64, 73, 66, 205, 121, 128, 504, 247]


def main():
    # ---------- 1) Datos y misma división que T1 ----------
    data = load_breast_cancer()
    X, y = data.data, data.target

    indices = np.arange(len(y))
    idx_train, idx_test, y_train, y_test = train_test_split(
        indices, y,
        test_size=0.3,
        stratify=y,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    X_train, X_test = X[idx_train], X[idx_test]

    coincide_train = [int(i) for i in idx_train[: len(T1_TRAIN_PREFIX)]] == T1_TRAIN_PREFIX
    coincide_test = [int(i) for i in idx_test[: len(T1_TEST_PREFIX)]] == T1_TEST_PREFIX
    coincide_T1 = bool(coincide_train and coincide_test)

    n_train, n_test = int(len(y_train)), int(len(y_test))

    # ---------- 2) Curva de aprendizaje ----------
    fracciones_usadas = []
    n_ejemplos_lista = []
    acc_gnb_lista = []
    acc_lr_lista = []
    puntos = []

    for frac in FRACCIONES:
        # Submuestra estratificada del ENTRENAMIENTO (random_state=42)
        if frac >= 1.0:
            idx_sub = np.arange(n_train)  # 100%: todo el entrenamiento
        else:
            idx_sub, _ = train_test_split(
                np.arange(n_train),
                train_size=frac,
                stratify=y_train,
                random_state=RANDOM_STATE,
                shuffle=True,
            )

        X_sub, y_sub = X_train[idx_sub], y_train[idx_sub]
        n_sub = int(len(y_sub))

        # Escalador ajustado SOLO con la submuestra correspondiente
        scaler = StandardScaler().fit(X_sub)
        X_sub_std = scaler.transform(X_sub)
        X_test_std = scaler.transform(X_test)  # mismo test siempre, solo transformado

        # --- GaussianNB ---
        gnb = GaussianNB()
        gnb.fit(X_sub_std, y_sub)
        acc_gnb = float(accuracy_score(y_test, gnb.predict(X_test_std)))

        # --- Regresión logística ---
        logreg = LogisticRegression(max_iter=5000)
        logreg.fit(X_sub_std, y_sub)
        acc_lr = float(accuracy_score(y_test, logreg.predict(X_test_std)))

        fracciones_usadas.append(float(frac))
        n_ejemplos_lista.append(n_sub)
        acc_gnb_lista.append(acc_gnb)
        acc_lr_lista.append(acc_lr)
        puntos.append({
            "fraccion": float(frac),
            "fraccion_porcentaje": float(frac * 100.0),
            "n_ejemplos_entrenamiento": n_sub,
            "accuracy_test_GaussianNB": acc_gnb,
            "accuracy_test_LogisticRegression": acc_lr,
        })

    # ---------- 3) Figura: exactitud vs número de ejemplos ----------
    plt.figure(figsize=(8.0, 5.5))
    plt.plot(n_ejemplos_lista, acc_gnb_lista, marker="o", color="tab:blue",
             label="GaussianNB")
    plt.plot(n_ejemplos_lista, acc_lr_lista, marker="s", color="tab:orange",
             label="Regresión logística")
    plt.xlabel("Número de ejemplos de entrenamiento")
    plt.ylabel("Exactitud (accuracy) en el conjunto de prueba")
    plt.title("Curva de aprendizaje — breast_cancer (división de T1, prueba fija)")
    xt_labels = ["%d\n(%.0f%%)" % (n, f * 100.0)
                 for n, f in zip(n_ejemplos_lista, fracciones_usadas)]
    plt.xticks(n_ejemplos_lista, xt_labels)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    ymin = min(min(acc_gnb_lista), min(acc_lr_lista))
    plt.ylim(min(0.80, ymin - 0.02), 1.02)
    plt.tight_layout()
    plt.savefig(FIGURA_PNG, dpi=150)
    plt.close()

    # ---------- 4) Guardar todas las cifras en resultados.json ----------
    resultados = {
        "resultados": {
            "subtarea": "T3_curva_aprendizaje",
            "dataset": "breast_cancer_wisconsin_diagnostic",
            "division_T1_recalculada": {
                "test_size": 0.3,
                "estratificado": True,
                "random_state": RANDOM_STATE,
                "n_train": n_train,
                "n_test": n_test,
                "coincide_indices_con_T1": coincide_T1,
            },
            "submuestreo": {
                "estratificado": True,
                "random_state": RANDOM_STATE,
                "fracciones": fracciones_usadas,
            },
            "escalador": "StandardScaler ajustado en cada submuestra (test solo transformado)",
            "metrica": "accuracy sobre el mismo conjunto de prueba de T1",
            "puntos_curva": puntos,
            "curvas": {
                "GaussianNB": {
                    "n_ejemplos": n_ejemplos_lista,
                    "accuracy_test": acc_gnb_lista,
                },
                "LogisticRegression": {
                    "n_ejemplos": n_ejemplos_lista,
                    "accuracy_test": acc_lr_lista,
                },
            },
            "figura_png": FIGURA_PNG,
        }
    }
    with open("resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    # ---------- 5) Impresión de cifras principales (4 decimales) ----------
    print("T3 — Curva de aprendizaje (división T1: train=%d, test=%d)" % (n_train, n_test))
    print("División idéntica a T1 (índices): %s" % ("sí" if coincide_T1 else "NO"))
    print("%-9s %-11s %-14s %-14s" % ("% train", "n_ejemplos", "GaussianNB", "LogReg"))
    for p in puntos:
        print(
            "%-9.1f %-11d %-14.4f %-14.4f"
            % (
                p["fraccion_porcentaje"],
                p["n_ejemplos_entrenamiento"],
                p["accuracy_test_GaussianNB"],
                p["accuracy_test_LogisticRegression"],
            )
        )
    print("Figura guardada en: %s" % FIGURA_PNG)
    print("Resultados guardados en: resultados.json")


if __name__ == "__main__":
    main()
