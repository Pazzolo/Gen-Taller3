# -*- coding: utf-8 -*-
"""
T2 — Dos clasificadores sobre la división de T1 (Breast Cancer Wisconsin Diagnostic)

Entrena:
  1) GaussianNB (modelo generativo)
  2) LogisticRegression (modelo discriminativo)
ambos sobre atributos estandarizados con StandardScaler ajustado SOLO con el
conjunto de entrenamiento, y evalúa accuracy y F1 macro sobre el conjunto de
prueba. Construye la tabla comparativa y guarda todas las cifras en
resultados.json. Sin figuras.
"""

import json
import os

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# ---------------- Parámetros fijados por el enunciado (T1) ----------------
TEST_SIZE = 0.3
RANDOM_STATE = 42
ARCHIVO_RESULTADOS = "resultados.json"


def cargar_resultados_previos():
    """Carga resultados.json si existe, para conservar las cifras de T1."""
    if os.path.exists(ARCHIVO_RESULTADOS):
        try:
            with open(ARCHIVO_RESULTADOS, "r", encoding="utf-8") as f:
                contenido = json.load(f)
            if isinstance(contenido, dict):
                return contenido
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def main():
    # ---------------- Datos (mismos que T1) ----------------
    data = load_breast_cancer()
    X, y = data.data, data.target

    # ---------------- División de T1: 70/30 estratificada, random_state=42 ----------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # ---------------- Estandarización: ajustada SOLO con el entrenamiento ----------------
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)   # fit + transform en entrenamiento
    X_test_std = scaler.transform(X_test)         # el test jamás participa en el ajuste

    # ---------------- Modelo 1: Naive Bayes gaussiano (generativo) ----------------
    gnb = GaussianNB()
    gnb.fit(X_train_std, y_train)
    y_pred_gnb = gnb.predict(X_test_std)

    # ---------------- Modelo 2: Regresión logística (discriminativo) ----------------
    logreg = LogisticRegression(max_iter=5000)
    logreg.fit(X_train_std, y_train)
    y_pred_lr = logreg.predict(X_test_std)

    # ---------------- Métricas sobre el conjunto de prueba ----------------
    acc_gnb = float(accuracy_score(y_test, y_pred_gnb))
    f1_gnb = float(f1_score(y_test, y_pred_gnb, average="macro"))
    acc_lr = float(accuracy_score(y_test, y_pred_lr))
    f1_lr = float(f1_score(y_test, y_pred_lr, average="macro"))

    err_gnb = float(np.sum(y_pred_gnb != y_test))
    err_lr = float(np.sum(y_pred_lr != y_test))

    n_train = float(X_train.shape[0])
    n_test = float(X_test.shape[0])

    # ---------------- Tabla comparativa ----------------
    print("=== T2: GaussianNB vs Regresion logistica (conjunto de prueba) ===")
    print(f"Ejemplos de prueba: {n_test:.0f}  (entrenamiento: {n_train:.0f})")
    print("-" * 62)
    encabezado = f"{'Modelo':<24}{'Accuracy':>12}{'F1 macro':>12}{'Errores':>12}"
    print(encabezado)
    print("-" * 62)
    print(f"{'GaussianNB':<24}{acc_gnb:>12.4f}{f1_gnb:>12.4f}{err_gnb:>12.0f}")
    print(f"{'Regresion logistica':<24}{acc_lr:>12.4f}{f1_lr:>12.4f}{err_lr:>12.0f}")
    print("-" * 62)
    print(f"Diferencia accuracy (LR - NB): {acc_lr - acc_gnb:.4f}")
    print(f"Diferencia F1 macro (LR - NB): {f1_lr - f1_gnb:.4f}")

    # ---------------- Guardar todas las cifras en resultados.json ----------------
    resultados = cargar_resultados_previos()
    resultados.update({
        "t2_n_ejemplos_entrenamiento": n_train,
        "t2_n_ejemplos_prueba": n_test,
        "t2_accuracy_gnb_prueba": acc_gnb,
        "t2_f1_macro_gnb_prueba": f1_gnb,
        "t2_errores_gnb_prueba": err_gnb,
        "t2_accuracy_lr_prueba": acc_lr,
        "t2_f1_macro_lr_prueba": f1_lr,
        "t2_errores_lr_prueba": err_lr,
        "t2_diferencia_accuracy_lr_menos_gnb": float(acc_lr - acc_gnb),
        "t2_diferencia_f1_macro_lr_menos_gnb": float(f1_lr - f1_gnb),
    })
    with open(ARCHIVO_RESULTADOS, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    print("Cifras guardadas en resultados.json")


if __name__ == "__main__":
    main()
