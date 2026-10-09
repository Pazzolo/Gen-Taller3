# -*- coding: utf-8 -*-
"""
SUBTAREA T1: Carga del conjunto Breast Cancer Wisconsin (Diagnostic),
reporte de dimensiones y casos por clase, y división única 70/30
estratificada con random_state=42.

Esta división es la única que se usa en toda la tarea: el conjunto de
prueba no se toca para ajustar nada. Al ser determinista (random_state=42
y stratify=y), las subtareas posteriores pueden reproducirla exactamente
con la misma llamada a train_test_split.
"""

import json
import os

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

RUTA_RESULTADOS = "resultados.json"


def cargar_resultados_previos():
    """Carga resultados.json si existe (para no sobrescribir otras subtareas)."""
    if os.path.exists(RUTA_RESULTADOS):
        try:
            with open(RUTA_RESULTADOS, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def guardar_resultados(resultados):
    with open(RUTA_RESULTADOS, "w") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)


def main():
    # ------------------------------------------------------------------
    # 1) Carga de datos
    # ------------------------------------------------------------------
    data = load_breast_cancer()
    X, y = data.data, data.target
    nombres_clases = [str(n) for n in data.target_names]

    n_ejemplos = int(X.shape[0])
    n_atributos = int(X.shape[1])

    # Casos por clase en el conjunto completo
    conteo_clases_total = {
        nombre: int(np.sum(y == i)) for i, nombre in enumerate(nombres_clases)
    }

    # ------------------------------------------------------------------
    # 2) División única 70% entrenamiento / 30% prueba, estratificada
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )

    n_train = int(X_train.shape[0])
    n_test = int(X_test.shape[0])

    conteo_clases_train = {
        nombre: int(np.sum(y_train == i)) for i, nombre in enumerate(nombres_clases)
    }
    conteo_clases_test = {
        nombre: int(np.sum(y_test == i)) for i, nombre in enumerate(nombres_clases)
    }

    prop_train = float(n_train) / float(n_ejemplos)
    prop_test = float(n_test) / float(n_ejemplos)

    # ------------------------------------------------------------------
    # 3) Guardar TODAS las cifras calculadas en resultados.json
    # ------------------------------------------------------------------
    resultados = cargar_resultados_previos()
    resultados["T1"] = {
        "n_ejemplos_total": float(n_ejemplos),
        "n_atributos": float(n_atributos),
        "casos_por_clase_total": {
            nombre: float(cuenta) for nombre, cuenta in conteo_clases_total.items()
        },
        "n_ejemplos_entrenamiento": float(n_train),
        "n_ejemplos_prueba": float(n_test),
        "proporcion_entrenamiento": prop_train,
        "proporcion_prueba": prop_test,
        "casos_por_clase_entrenamiento": {
            nombre: float(cuenta) for nombre, cuenta in conteo_clases_train.items()
        },
        "casos_por_clase_prueba": {
            nombre: float(cuenta) for nombre, cuenta in conteo_clases_test.items()
        },
        "parametros_division": {
            "test_size": 0.30,
            "stratify": "y",
            "random_state": 42,
        },
    }
    guardar_resultados(resultados)

    # ------------------------------------------------------------------
    # 4) Impresión de las cifras principales (4 decimales donde aplica)
    # ------------------------------------------------------------------
    print("=== T1: Carga y división de Breast Cancer (Diagnostic) ===")
    print(f"Ejemplos totales: {n_ejemplos}")
    print(f"Atributos: {n_atributos}")
    print("Casos por clase (total):")
    for nombre in nombres_clases:
        print(f"  {nombre}: {conteo_clases_total[nombre]}")
    print(f"Entrenamiento: {n_train} ejemplos "
          f"({prop_train:.4f} del total)")
    print(f"Prueba: {n_test} ejemplos ({prop_test:.4f} del total)")
    print("Casos por clase (entrenamiento / prueba):")
    for nombre in nombres_clases:
        print(f"  {nombre}: {conteo_clases_train[nombre]} / "
              f"{conteo_clases_test[nombre]}")
    print("División guardada (determinista: test_size=0.30, stratify=y, "
          "random_state=42) y cifras escritas en resultados.json")


if __name__ == "__main__":
    main()
