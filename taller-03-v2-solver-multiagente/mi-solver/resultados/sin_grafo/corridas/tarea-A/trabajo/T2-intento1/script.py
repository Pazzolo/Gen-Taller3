# -*- coding: utf-8 -*-
"""
Subtarea T2: Sobre la división de T1 (breast_cancer, test_size=0.3, estratificado,
random_state=42), entrenar GaussianNB y LogisticRegression (con StandardScaler
ajustado SOLO con el entrenamiento), evaluar accuracy y F1 macro sobre el
conjunto de prueba y producir la tabla comparativa de la Parte 2.

Sin figura. Resultados guardados en resultados.json.
"""

import json
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

RANDOM_STATE = 42
TEST_SIZE = 0.3

# ---------- 1. Datos (mismo dataset que T1) ----------
data = load_breast_cancer()
X = data.data
y = data.target
class_names = [str(s) for s in data.target_names]  # ['malignant', 'benign']

n_total, n_atributos = X.shape
conteo_total = {class_names[int(c)]: int(np.sum(y == c)) for c in np.unique(y)}

# ---------- 2. Misma división que T1 ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE,
)

conteo_train = {class_names[int(c)]: int(np.sum(y_train == c)) for c in np.unique(y_train)}
conteo_test = {class_names[int(c)]: int(np.sum(y_test == c)) for c in np.unique(y_test)}

# ---------- 3. Escalador ajustado SOLO con el entrenamiento ----------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit solo con train
X_test_scaled = scaler.transform(X_test)        # test transformado con parámetros de train

# ---------- 4. Modelos ----------
# 4a. GaussianNB sobre los atributos originales (sin escalar)
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)

# 4b. Regresión logística sobre los atributos estandarizados
logreg = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)
logreg.fit(X_train_scaled, y_train)
y_pred_lr = logreg.predict(X_test_scaled)

# ---------- 5. Métricas sobre el conjunto de prueba ----------
acc_gnb = float(accuracy_score(y_test, y_pred_gnb))
f1mac_gnb = float(f1_score(y_test, y_pred_gnb, average="macro"))
f1cls_gnb = [float(v) for v in f1_score(y_test, y_pred_gnb, average=None)]
cm_gnb = [[int(v) for v in fila] for fila in confusion_matrix(y_test, y_pred_gnb)]

acc_lr = float(accuracy_score(y_test, y_pred_lr))
f1mac_lr = float(f1_score(y_test, y_pred_lr, average="macro"))
f1cls_lr = [float(v) for v in f1_score(y_test, y_pred_lr, average=None)]
cm_lr = [[int(v) for v in fila] for fila in confusion_matrix(y_test, y_pred_lr)]

# ---------- 6. Tabla comparativa (Parte 2) ----------
tabla = [
    {
        "modelo": "GaussianNB",
        "accuracy": acc_gnb,
        "f1_macro": f1mac_gnb,
    },
    {
        "modelo": "LogisticRegression",
        "accuracy": acc_lr,
        "f1_macro": f1mac_lr,
    },
]

# ---------- 7. Guardar todas las cifras en resultados.json ----------
resultados = {
    "resultados": {
        "subtarea": "T2_dos_clasificadores_parte2",
        "dataset": "breast_cancer_wisconsin_diagnostic",
        "n_ejemplos_total": int(n_total),
        "n_atributos": int(n_atributos),
        "nombres_clases": class_names,
        "conteo_clases_total": conteo_total,
        "division": {
            "test_size": TEST_SIZE,
            "estratificado": True,
            "random_state": RANDOM_STATE,
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "conteo_clases_train": conteo_train,
            "conteo_clases_test": conteo_test,
        },
        "preprocesado": {
            "GaussianNB": "atributos originales (sin escalar)",
            "LogisticRegression": "StandardScaler ajustado solo con el conjunto de entrenamiento",
            "escalador_ajustado_solo_con_train": True,
        },
        "hiperparametros": {
            "GaussianNB": "valores por defecto",
            "LogisticRegression": {
                "max_iter": 5000,
                "random_state": RANDOM_STATE,
                "resto": "valores por defecto",
            },
        },
        "metricas_test": {
            "GaussianNB": {
                "accuracy": acc_gnb,
                "f1_macro": f1mac_gnb,
                "f1_por_clase": {
                    class_names[i]: f1cls_gnb[i] for i in range(len(class_names))
                },
                "matriz_confusion": {"clases": class_names, "valores": cm_gnb},
            },
            "LogisticRegression": {
                "accuracy": acc_lr,
                "f1_macro": f1mac_lr,
                "f1_por_clase": {
                    class_names[i]: f1cls_lr[i] for i in range(len(class_names))
                },
                "matriz_confusion": {"clases": class_names, "valores": cm_lr},
            },
        },
        "tabla_comparativa_parte2": tabla,
        "diferencias": {
            "accuracy_logreg_menos_gnb": acc_lr - acc_gnb,
            "f1_macro_logreg_menos_gnb": f1mac_lr - f1mac_gnb,
        },
    }
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ---------- 8. Impresión de cifras principales (4 decimales) ----------
print("Subtarea T2 — Parte 2: dos clasificadores (breast_cancer)")
print(
    f"Division: train={X_train.shape[0]}, test={X_test.shape[0]} "
    f"(test_size={TEST_SIZE}, estratificado, random_state={RANDOM_STATE})"
)
print()
print("Tabla comparativa — conjunto de prueba")
header = f"{'Modelo':<22}{'Accuracy':>10}{'F1 macro':>10}"
print(header)
print("-" * len(header))
for fila in tabla:
    print(f"{fila['modelo']:<22}{fila['accuracy']:>10.4f}{fila['f1_macro']:>10.4f}")
print()
print(f"GaussianNB         -> accuracy={acc_gnb:.4f}, f1_macro={f1mac_gnb:.4f}")
print(f"LogisticRegression -> accuracy={acc_lr:.4f}, f1_macro={f1mac_lr:.4f}")
print()
print("F1 por clase (test):")
for nombre, f1cls in [("GaussianNB", f1cls_gnb), ("LogisticRegression", f1cls_lr)]:
    print(
        f"  {nombre}: "
        + ", ".join(f"{class_names[i]}={f1cls[i]:.4f}" for i in range(len(class_names)))
    )
print()
print("Resultados guardados en resultados.json")
