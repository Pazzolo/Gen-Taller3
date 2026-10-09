# -*- coding: utf-8 -*-
"""
T2 — Generativo contra discriminativo (Breast Cancer, Diagnostic)
Entrena sobre el 100% del entrenamiento de T1:
  - GaussianNB (modelo generativo)
  - Regresión logística (modelo discriminativo)
ambos dentro de un Pipeline con StandardScaler ajustado SOLO con el entrenamiento.
Evalúa ambos en el conjunto de prueba fijo y reporta accuracy y F1 macro en una tabla.
"""

import json

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------- 1) Datos y división fija de T1 (determinista, se recalcula igual) ----------
TEST_SIZE = 0.3
RANDOM_STATE = 42

data = load_breast_cancer()
X, y = data.data, data.target
clases = list(data.target_names)  # 0: malignant, 1: benign

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

# ---------- 2) Modelos en Pipeline (StandardScaler ajustado solo con train) ----------
# Nota: para GaussianNB el escalado es una transformación afín por atributo y no
# altera sus predicciones; se incluye el mismo Pipeline por fidelidad al enunciado.
pipelines = {
    "GaussianNB": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GaussianNB()),
    ]),
    "RegresionLogistica": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
    ]),
}

# ---------- 3) Entrenamiento sobre el 100% del train de T1 y evaluación en test fijo ----------
resultados_modelos = {}
for nombre, pipe in pipelines.items():
    pipe.fit(X_train, y_train)  # el escalador se ajusta SOLO con el entrenamiento
    pred_train = pipe.predict(X_train)
    pred_test = pipe.predict(X_test)

    resultados_modelos[nombre] = {
        "accuracy_entrenamiento": float(accuracy_score(y_train, pred_train)),
        "f1_macro_entrenamiento": float(f1_score(y_train, pred_train, average="macro")),
        "accuracy_prueba": float(accuracy_score(y_test, pred_test)),
        "f1_macro_prueba": float(f1_score(y_test, pred_test, average="macro")),
        "matriz_confusion_prueba": confusion_matrix(y_test, pred_test).tolist(),
    }

# ---------- 4) Guardar TODAS las cifras en resultados.json ----------
salida_t2 = {
    "division_reproducida": {
        "dataset": "load_breast_cancer",
        "test_size": float(TEST_SIZE),
        "stratify": "y",
        "random_state": int(RANDOM_STATE),
        "n_total": int(X.shape[0]),
        "n_atributos": int(X.shape[1]),
        "n_entrenamiento": int(X_train.shape[0]),
        "n_prueba": int(X_test.shape[0]),
        "clases": clases,
        "casos_por_clase_entrenamiento": {
            clases[0]: int((y_train == 0).sum()),
            clases[1]: int((y_train == 1).sum()),
        },
        "casos_por_clase_prueba": {
            clases[0]: int((y_test == 0).sum()),
            clases[1]: int((y_test == 1).sum()),
        },
    },
    "modelos": resultados_modelos,
}

# Se conserva lo que pudiera existir en resultados.json (p. ej. T1) y se añade la clave "T2"
try:
    with open("resultados.json", "r", encoding="utf-8") as f:
        previo = json.load(f)
    if not isinstance(previo, dict):
        previo = {}
except (FileNotFoundError, ValueError):
    previo = {}

bloque = previo.get("resultados") if isinstance(previo.get("resultados"), dict) else {}
bloque["T2"] = salida_t2
previo["resultados"] = bloque

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(previo, f, indent=2, ensure_ascii=False)

# ---------- 5) Tabla impresa (cifras principales con 4 decimales) ----------
print("=== T2: GaussianNB vs Regresión Logística (Pipeline con StandardScaler) ===")
print(f"División reproducida de T1: train={X_train.shape[0]}, test={X_test.shape[0]} "
      f"(test_size={TEST_SIZE:.2f}, stratify=y, random_state={RANDOM_STATE})")
print()

encabezado = f"{'Modelo':<22}{'Accuracy':>10}{'F1 macro':>10}"
print("Tabla — conjunto de PRUEBA fijo:")
print(encabezado)
print("-" * len(encabezado))
for nombre in pipelines:
    m = resultados_modelos[nombre]
    print(f"{nombre:<22}{m['accuracy_prueba']:>10.4f}{m['f1_macro_prueba']:>10.4f}")
print()

print("Tabla — conjunto de ENTRENAMIENTO (referencia):")
print(encabezado)
print("-" * len(encabezado))
for nombre in pipelines:
    m = resultados_modelos[nombre]
    print(f"{nombre:<22}{m['accuracy_entrenamiento']:>10.4f}{m['f1_macro_entrenamiento']:>10.4f}")
print()

for nombre in pipelines:
    cm = resultados_modelos[nombre]["matriz_confusion_prueba"]
    print(f"Matriz de confusión en prueba ({nombre}) [filas=real, cols=pred; clases {clases}]:")
    for fila, etiqueta in zip(cm, clases):
        print(f"  {etiqueta:<10}{fila}")
print()
print("Cifras guardadas en resultados.json (clave 'T2')")
