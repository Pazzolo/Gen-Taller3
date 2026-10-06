# -*- coding: utf-8 -*-
"""
Subtarea T1 — Tarea E (Ridge y validación cruzada)
===================================================
Parte 1 — Datos:
  1. Cargar sklearn.datasets.load_diabetes (incluido en scikit-learn, sin descargas).
  2. Reportar número de ejemplos y de atributos.
  3. Dividir en entrenamiento (75 %) y prueba (25 %) con train_test_split(random_state=7).
     Esta es la ÚNICA división de la tarea; el conjunto de prueba se reserva para la Parte 3.

Figura: no se requiere.
Salidas: impresión de las cifras principales (4 decimales donde aplica) y
         guardado de todas las cifras en "resultados.json".
"""

import json

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

# Parámetros fijados por el enunciado (no se toca el conjunto de prueba aquí)
RANDOM_STATE = 7
TEST_SIZE = 0.25

# ---------------------------------------------------------------------------
# 1) Carga del conjunto Diabetes
# ---------------------------------------------------------------------------
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

n_ejemplos_total, n_atributos_total = X.shape

# ---------------------------------------------------------------------------
# 2) Única división de la tarea: 75 % entrenamiento / 25 % prueba
#    (shuffle por defecto de train_test_split, random_state=7)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

n_ejemplos_entrenamiento = int(X_train.shape[0])
n_ejemplos_prueba = int(X_test.shape[0])

proporcion_entrenamiento = n_ejemplos_entrenamiento / n_ejemplos_total
proporcion_prueba = n_ejemplos_prueba / n_ejemplos_total

# Comprobación de consistencia (sin usar el contenido de los datos)
assert n_ejemplos_entrenamiento + n_ejemplos_prueba == n_ejemplos_total
assert X_train.shape[1] == n_atributos_total == X_test.shape[1]

# ---------------------------------------------------------------------------
# 3) Guardar TODAS las cifras calculadas en resultados.json
#    (se conservan resultados de subtareas previas si el archivo ya existe)
# ---------------------------------------------------------------------------
resultados = {}
try:
    with open("resultados.json", "r", encoding="utf-8") as f:
        previo = json.load(f)
    if isinstance(previo, dict):
        resultados = previo
except (FileNotFoundError, json.JSONDecodeError, ValueError):
    resultados = {}

resultados.update(
    {
        "t1_n_ejemplos_total": float(n_ejemplos_total),
        "t1_n_atributos_total": float(n_atributos_total),
        "t1_n_ejemplos_entrenamiento": float(n_ejemplos_entrenamiento),
        "t1_n_ejemplos_prueba": float(n_ejemplos_prueba),
        "t1_proporcion_entrenamiento": float(proporcion_entrenamiento),
        "t1_proporcion_prueba": float(proporcion_prueba),
        "t1_random_state": float(RANDOM_STATE),
        "t1_test_size": float(TEST_SIZE),
    }
)

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

# ---------------------------------------------------------------------------
# 4) Reporte de las cifras principales
# ---------------------------------------------------------------------------
print("=== Subtarea T1: carga de datos y única división entrenamiento/prueba ===")
print("Conjunto: sklearn.datasets.load_diabetes")
print(f"Numero de ejemplos (total): {n_ejemplos_total}")
print(f"Numero de atributos (total): {n_atributos_total}")
print(f"Entrenamiento (75%): {n_ejemplos_entrenamiento} ejemplos "
      f"({proporcion_entrenamiento:.4f} del total)")
print(f"Prueba (25%): {n_ejemplos_prueba} ejemplos "
      f"({proporcion_prueba:.4f} del total)")
print(f"Division: train_test_split(test_size={TEST_SIZE}, random_state={RANDOM_STATE})")
print("El conjunto de prueba queda reservado; solo se usara una vez, al final de la Parte 3.")
print("Cifras guardadas en resultados.json")
