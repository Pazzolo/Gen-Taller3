# T1: Carga de Breast Cancer (Wisconsin Diagnostic), reporte descriptivo
# y única división 70/30 estratificada con random_state=42.

import json
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def main():
    # ------------------------------------------------------------------
    # 1) Carga del conjunto de datos
    # ------------------------------------------------------------------
    data = load_breast_cancer()
    X = data.data
    y = data.target
    nombres_clases = [str(s) for s in data.target_names]

    n_ejemplos, n_atributos = X.shape

    # Conteos por clase en el conjunto completo
    valores_clase, conteos_clase = np.unique(y, return_counts=True)
    conteo_total = {}
    conteo_total_codigo = {}
    for v, c in zip(valores_clase, conteos_clase):
        v_int = int(v)
        conteo_total_codigo[v_int] = int(c)
        conteo_total[nombres_clases[v_int]] = int(c)

    # ------------------------------------------------------------------
    # 2) Única división 70% entrenamiento / 30% prueba, estratificada
    # ------------------------------------------------------------------
    indices = np.arange(n_ejemplos)
    idx_train, idx_test, y_train, y_test = train_test_split(
        indices,
        y,
        test_size=0.30,
        stratify=y,
        random_state=42,
    )

    # Conteos por clase dentro de cada partición (verificación del estrato)
    val_tr, cont_tr = np.unique(y_train, return_counts=True)
    val_te, cont_te = np.unique(y_test, return_counts=True)
    conteo_train = {nombres_clases[int(v)]: int(c) for v, c in zip(val_tr, cont_tr)}
    conteo_test = {nombres_clases[int(v)]: int(c) for v, c in zip(val_te, cont_te)}

    n_train = int(len(idx_train))
    n_test = int(len(idx_test))

    # ------------------------------------------------------------------
    # 3) Guardar todas las cifras en resultados.json
    # ------------------------------------------------------------------
    resultados = {
        "subtarea": "T1_carga_y_division",
        "dataset": "breast_cancer_wisconsin_diagnostic",
        "n_ejemplos_total": int(n_ejemplos),
        "n_atributos": int(n_atributos),
        "n_clases": int(len(valores_clase)),
        "nombres_clases": nombres_clases,
        "conteo_clases_total": conteo_total,
        "conteo_clases_total_por_codigo": conteo_total_codigo,
        "proporcion_clase_0_malignant": float(conteo_total_codigo.get(0, 0) / n_ejemplos),
        "proporcion_clase_1_benign": float(conteo_total_codigo.get(1, 0) / n_ejemplos),
        "division": {
            "test_size": 0.30,
            "estratificado": True,
            "random_state": 42,
            "n_train": n_train,
            "n_test": n_test,
            "proporcion_train": float(n_train / n_ejemplos),
            "proporcion_test": float(n_test / n_ejemplos),
            "conteo_clases_train": conteo_train,
            "conteo_clases_test": conteo_test,
            "indices_train": [int(i) for i in idx_train],
            "indices_test": [int(i) for i in idx_test],
        },
    }

    with open("resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # 4) Reporte por consola (cifras principales con 4 decimales)
    # ------------------------------------------------------------------
    print("=== T1: Breast Cancer Wisconsin (Diagnostic) ===")
    print(f"Numero de ejemplos: {n_ejemplos}")
    print(f"Numero de atributos: {n_atributos}")
    print("Casos por clase (total):")
    for nombre in nombres_clases:
        print(f"  {nombre}: {conteo_total[nombre]}")
    print(f"Proporcion malignant: {conteo_total['malignant'] / n_ejemplos:.4f}")
    print(f"Proporcion benign:    {conteo_total['benign'] / n_ejemplos:.4f}")
    print("")
    print("Division 70/30 estratificada (random_state=42):")
    print(f"  Entrenamiento: {n_train} ejemplos ({n_train / n_ejemplos:.4f} del total)")
    print(f"  Prueba:        {n_test} ejemplos ({n_test / n_ejemplos:.4f} del total)")
    print("Casos por clase en entrenamiento:")
    for nombre in nombres_clases:
        print(f"  {nombre}: {conteo_train[nombre]}")
    print("Casos por clase en prueba:")
    for nombre in nombres_clases:
        print(f"  {nombre}: {conteo_test[nombre]}")
    print("")
    print("Resultados guardados en resultados.json")


if __name__ == "__main__":
    main()
