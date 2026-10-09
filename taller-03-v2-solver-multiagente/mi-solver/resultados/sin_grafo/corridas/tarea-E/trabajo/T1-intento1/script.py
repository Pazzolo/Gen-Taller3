# SUBTAREA T1: Cargar el conjunto Diabetes, reportar dimensiones y hacer la
# única división entrenamiento/prueba de la tarea (75%/25%, random_state=7).
# El conjunto de prueba queda reservado para la Parte 3.

import json
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split


def main():
    # ------------------------------------------------------------------
    # 1) Cargar los datos
    # ------------------------------------------------------------------
    data = load_diabetes()
    X = data.data
    y = data.target

    n_ejemplos = int(X.shape[0])
    n_atributos = int(X.shape[1])

    # ------------------------------------------------------------------
    # 2) División única 75% / 25% con random_state=7
    #    (estratificación no aplica: es un problema de regresión)
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=7
    )

    n_train = int(X_train.shape[0])
    n_test = int(X_test.shape[0])
    prop_train = float(n_train / n_ejemplos)
    prop_test = float(n_test / n_ejemplos)

    # Estadísticas descriptivas del objetivo (solo informativas; el conjunto
    # de prueba NO se usa para nada más en esta subtarea)
    media_y_total = float(np.mean(y))
    desv_y_total = float(np.std(y))
    media_y_train = float(np.mean(y_train))
    desv_y_train = float(np.std(y_train))

    # ------------------------------------------------------------------
    # 3) Guardar todas las cifras calculadas en resultados.json
    # ------------------------------------------------------------------
    resultados = {
        "subtarea": "T1_carga_y_division_diabetes",
        "dataset": "sklearn.datasets.load_diabetes",
        "n_ejemplos_total": n_ejemplos,
        "n_atributos": n_atributos,
        "n_ejemplos_entrenamiento": n_train,
        "n_ejemplos_prueba": n_test,
        "proporcion_entrenamiento": prop_train,
        "proporcion_prueba": prop_test,
        "test_size": 0.25,
        "random_state": 7,
        "media_objetivo_total": media_y_total,
        "desv_est_objetivo_total": desv_y_total,
        "media_objetivo_entrenamiento": media_y_train,
        "desv_est_objetivo_entrenamiento": desv_y_train,
    }

    with open("resultados.json", "w") as f:
        json.dump(resultados, f, indent=2)

    # ------------------------------------------------------------------
    # 4) Reporte de las cifras principales con 4 decimales
    # ------------------------------------------------------------------
    print("=== T1: Carga y división del conjunto Diabetes ===")
    print(f"Ejemplos totales: {n_ejemplos}")
    print(f"Atributos: {n_atributos}")
    print(f"Entrenamiento: {n_train} ejemplos ({prop_train * 100:.4f} %)")
    print(f"Prueba: {n_test} ejemplos ({prop_test * 100:.4f} %)")
    print(f"Media del objetivo (total): {media_y_total:.4f}")
    print(f"Desv. est. del objetivo (total): {desv_y_total:.4f}")
    print(f"Media del objetivo (entrenamiento): {media_y_train:.4f}")
    print(f"Desv. est. del objetivo (entrenamiento): {desv_y_train:.4f}")
    print("Resultados guardados en resultados.json")
    print("El conjunto de prueba queda reservado para la Parte 3.")


if __name__ == "__main__":
    main()
