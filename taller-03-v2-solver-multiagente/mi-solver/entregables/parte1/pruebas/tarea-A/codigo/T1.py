# -*- coding: utf-8 -*-
"""
Subtarea T1 — Carga del conjunto Breast Cancer Wisconsin (Diagnostic),
reporte de dimensiones y casos por clase, y única división entrenamiento/prueba
(70 % / 30 %, estratificada, random_state=42) que se usará en toda la tarea.

El conjunto de prueba NO se usa para ajustar nada (ni escaladores ni
hiperparámetros); aquí solo se crea la división y se documenta.
"""

import json
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# 1. Carga de datos
# ---------------------------------------------------------------------------
data = load_breast_cancer()
X = data.data          # matriz de atributos (n_ejemplos x n_atributos)
y = data.target        # etiquetas: 0 = malignant, 1 = benign
nombres_clases = list(data.target_names)   # ['malignant', 'benign']
nombres_atributos = list(data.feature_names)

n_ejemplos_total = int(X.shape[0])
n_atributos = int(X.shape[1])

# Casos de cada clase en el conjunto completo
clase_0 = nombres_clases[0]  # 'malignant'
clase_1 = nombres_clases[1]  # 'benign'
casos_clase_0_total = int((y == 0).sum())
casos_clase_1_total = int((y == 1).sum())

# ---------------------------------------------------------------------------
# 2. Única división entrenamiento (70 %) / prueba (30 %), estratificada
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    stratify=y,
    random_state=42,
)

n_train = int(X_train.shape[0])
n_test = int(X_test.shape[0])

casos_clase_0_train = int((y_train == 0).sum())
casos_clase_1_train = int((y_train == 1).sum())
casos_clase_0_test = int((y_test == 0).sum())
casos_clase_1_test = int((y_test == 1).sum())

# Proporciones (verificación de la estratificación)
prop_clase_0_total = casos_clase_0_total / n_ejemplos_total
prop_clase_1_total = casos_clase_1_total / n_ejemplos_total
prop_clase_0_train = casos_clase_0_train / n_train
prop_clase_1_train = casos_clase_1_train / n_train
prop_clase_0_test = casos_clase_0_test / n_test
prop_clase_1_test = casos_clase_1_test / n_test

# ---------------------------------------------------------------------------
# 3. Guardar TODAS las cifras calculadas en resultados.json
#    (números como float de Python, sin redondear)
# ---------------------------------------------------------------------------
resultados = {
    "t1_n_ejemplos_total": float(n_ejemplos_total),
    "t1_n_atributos": float(n_atributos),
    "t1_nombre_clase_0": clase_0,
    "t1_nombre_clase_1": clase_1,
    "t1_casos_clase_0_total": float(casos_clase_0_total),
    "t1_casos_clase_1_total": float(casos_clase_1_total),
    "t1_prop_clase_0_total": float(prop_clase_0_total),
    "t1_prop_clase_1_total": float(prop_clase_1_total),
    "t1_n_ejemplos_entrenamiento": float(n_train),
    "t1_n_ejemplos_prueba": float(n_test),
    "t1_casos_clase_0_entrenamiento": float(casos_clase_0_train),
    "t1_casos_clase_1_entrenamiento": float(casos_clase_1_train),
    "t1_casos_clase_0_prueba": float(casos_clase_0_test),
    "t1_casos_clase_1_prueba": float(casos_clase_1_test),
    "t1_prop_clase_0_entrenamiento": float(prop_clase_0_train),
    "t1_prop_clase_1_entrenamiento": float(prop_clase_1_train),
    "t1_prop_clase_0_prueba": float(prop_clase_0_test),
    "t1_prop_clase_1_prueba": float(prop_clase_1_test),
    "t1_test_size": 0.30,
    "t1_random_state": 42.0,
    "t1_nombres_atributos": nombres_atributos,
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------------------
# 4. Imprimir las cifras principales con 4 decimales
# ---------------------------------------------------------------------------
print("=== T1: Breast Cancer Wisconsin (Diagnostic) ===")
print(f"Numero de ejemplos total : {n_ejemplos_total:.4f}")
print(f"Numero de atributos      : {n_atributos:.4f}")
print(f"Casos clase '{clase_0}' (total): {casos_clase_0_total:.4f}")
print(f"Casos clase '{clase_1}'  (total): {casos_clase_1_total:.4f}")
print(f"Proporcion '{clase_0}' (total) : {prop_clase_0_total:.4f}")
print(f"Proporcion '{clase_1}'  (total): {prop_clase_1_total:.4f}")
print("--- Division 70/30 estratificada (random_state=42) ---")
print(f"Ejemplos entrenamiento   : {n_train:.4f}")
print(f"Ejemplos prueba          : {n_test:.4f}")
print(f"'{clase_0}' en entrenamiento   : {casos_clase_0_train:.4f} "
      f"(prop {prop_clase_0_train:.4f})")
print(f"'{clase_1}' en entrenamiento  : {casos_clase_1_train:.4f} "
      f"(prop {prop_clase_1_train:.4f})")
print(f"'{clase_0}' en prueba          : {casos_clase_0_test:.4f} "
      f"(prop {prop_clase_0_test:.4f})")
print(f"'{clase_1}' en prueba         : {casos_clase_1_test:.4f} "
      f"(prop {prop_clase_1_test:.4f})")
print("Cifras guardadas en resultados.json")
