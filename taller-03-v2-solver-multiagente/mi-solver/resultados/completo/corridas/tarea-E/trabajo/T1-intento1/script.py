# Subtarea T1 — Parte 1: Carga del conjunto Diabetes y única división train/test 75%/25%
# El conjunto de prueba queda reservado para la Parte 3 (no se usa aquí para nada más).

import json
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split


def main():
    # ------------------------------------------------------------------
    # 1) Cargar el conjunto Diabetes incluido en scikit-learn
    # ------------------------------------------------------------------
    data = load_diabetes()
    X = data.data
    y = data.target

    n_ejemplos = int(X.shape[0])
    n_atributos = int(X.shape[1])

    # ------------------------------------------------------------------
    # 2) Única división train/test de la tarea: 75% / 25%, random_state=7
    #    (sin estratificación: es un problema de regresión)
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=7
    )

    n_train = int(X_train.shape[0])
    n_test = int(X_test.shape[0])
    proporcion_train = n_train / n_ejemplos
    proporcion_test = n_test / n_ejemplos

    # ------------------------------------------------------------------
    # 3) Guardar todas las cifras calculadas en resultados.json
    # ------------------------------------------------------------------
    resultados = {
        "n_ejemplos": float(n_ejemplos),
        "n_atributos": float(n_atributos),
        "n_train": float(n_train),
        "n_test": float(n_test),
        "proporcion_train": float(proporcion_train),
        "proporcion_test": float(proporcion_test),
        "test_size": 0.25,
        "random_state": 7.0,
        "forma_X_train": list(X_train.shape),
        "forma_X_test": list(X_test.shape),
        "nombres_atributos": list(data.feature_names),
    }

    with open("resultados.json", "w") as f:
        json.dump(resultados, f, indent=2)

    # ------------------------------------------------------------------
    # 4) Reporte de las cifras principales (proporciones con 4 decimales)
    # ------------------------------------------------------------------
    print("Parte 1 — Datos (sklearn.datasets.load_diabetes)")
    print(f"Numero de ejemplos: {n_ejemplos}")
    print(f"Numero de atributos: {n_atributos}")
    print(f"Division train/test: 75%/25% con random_state=7")
    print(f"Tamano de train: {n_train} ejemplos (proporcion = {proporcion_train:.4f})")
    print(f"Tamano de test:  {n_test} ejemplos (proporcion = {proporcion_test:.4f})")
    print(f"Forma de X_train: {X_train.shape} | Forma de X_test: {X_test.shape}")
    print("El conjunto de prueba queda reservado para la Parte 3.")
    print("Cifras guardadas en resultados.json")


if __name__ == "__main__":
    main()
