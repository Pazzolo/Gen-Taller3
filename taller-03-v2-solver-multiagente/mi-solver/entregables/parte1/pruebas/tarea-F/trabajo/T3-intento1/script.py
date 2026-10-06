# -*- coding: utf-8 -*-
"""
T3: Con el k elegido en T2 (silueta), comparar los grupos de k-means contra las
clases reales del Wine mediante el índice de Rand ajustado (ARI).

Script autocontenido: recarga los datos, repite la estandarización de T1 y la
selección de k por silueta de T2 (criterio interno, sin etiquetas), y solo al
final calcula el ARI contra las clases reales. No usa figura.
"""

import json
from math import comb

import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

RANDOM_STATE = 42
K_VALUES = [2, 3, 4, 5, 6, 7, 8]
N_INIT = 10


def main():
    # ---------- Datos (Wine, desde scikit-learn) ----------
    data = load_wine()
    X = data.data
    y = data.target
    class_names = list(data.target_names)
    n_ejemplos, n_atributos = X.shape

    # ---------- Estandarización (misma que T1/T2, ajustada solo con X) ----------
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # ---------- Repetir la selección de k de T2 (criterio interno: silueta) ----------
    silueta_por_k = {}
    inercia_por_k = {}
    for k in K_VALUES:
        km_k = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
        labels_k = km_k.fit_predict(X_std)
        silueta_por_k[k] = float(silhouette_score(X_std, labels_k))
        inercia_por_k[k] = float(km_k.inertia_)

    # k elegido solo con el criterio interno (silueta más alta), como en T2
    k_elegido = max(silueta_por_k, key=silueta_por_k.get)
    silueta_k_elegido = silueta_por_k[k_elegido]

    # ---------- KMeans final con el k elegido ----------
    km = KMeans(n_clusters=k_elegido, random_state=RANDOM_STATE, n_init=N_INIT)
    labels = km.fit_predict(X_std)

    # ---------- Comparación con las clases reales (solo al final) ----------
    ari = float(adjusted_rand_score(y, labels))

    # Tabla de contingencia clusters x clases reales
    n_clusters = int(k_elegido)
    n_clases = len(class_names)
    cont = np.zeros((n_clusters, n_clases), dtype=int)
    for i in range(n_clusters):
        for j in range(n_clases):
            cont[i, j] = int(np.sum((labels == i) & (y == j)))

    # Componentes del índice de Rand (sin ajustar) a partir de la tabla
    total_pares = comb(n_ejemplos, 2)
    a = sum(comb(int(cont[i, j]), 2) for i in range(n_clusters) for j in range(n_clases))
    sum_filas = sum(comb(int(cont[i, :].sum()), 2) for i in range(n_clusters))
    sum_cols = sum(comb(int(cont[:, j].sum()), 2) for j in range(n_clases))
    b = sum_filas - a  # mismo cluster, distinta clase
    c = sum_cols - a   # distinto cluster, misma clase
    d = total_pares - a - b - c
    rand_sin_ajustar = (a + d) / total_pares

    # Pureza (acuerdo por clase mayoritaria en cada cluster), solo descriptiva
    pureza = float(sum(int(cont[i, :].max()) for i in range(n_clusters)) / n_ejemplos)

    # Tamaños
    tamanos_clusters = {str(i): int(cont[i, :].sum()) for i in range(n_clusters)}
    tamanos_clases = {class_names[j]: int(cont[:, j].sum()) for j in range(n_clases)}

    # ---------- Guardar todas las cifras en resultados.json ----------
    resultados = {
        "subtarea": "T3_ari_kmeans_vs_clases_reales",
        "n_ejemplos": int(n_ejemplos),
        "n_atributos": int(n_atributos),
        "random_state": int(RANDOM_STATE),
        "k_values_evaluados": [int(k) for k in K_VALUES],
        "silueta_por_k": {str(k): silueta_por_k[k] for k in K_VALUES},
        "inercia_por_k": {str(k): inercia_por_k[k] for k in K_VALUES},
        "k_elegido_en_T2": int(k_elegido),
        "silueta_k_elegido": float(silueta_k_elegido),
        "n_clases_reales": int(n_clases),
        "ari": ari,
        "rand_index_sin_ajustar": float(rand_sin_ajustar),
        "pares_mismo_cluster_misma_clase": int(a),
        "pares_mismo_cluster_distinta_clase": int(b),
        "pares_distinto_cluster_misma_clase": int(c),
        "pares_distinto_cluster_distinta_clase": int(d),
        "pureza_asignacion_mayoritaria": pureza,
        "tabla_contingencia_clusters_x_clases": {
            f"cluster_{i}": {
                class_names[j]: int(cont[i, j]) for j in range(n_clases)
            } for i in range(n_clusters)
        },
        "tamanos_clusters": tamanos_clusters,
        "tamanos_clases_reales": tamanos_clases,
    }
    with open("resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    # ---------- Impresión de cifras principales (4 decimales) ----------
    print("=== T3: ARI de k-means (k elegido en T2) contra clases reales del Wine ===")
    print(f"Ejemplos: {n_ejemplos}  Atributos: {n_atributos}  Clases reales: {n_clases}")
    print(f"k elegido en T2 (silueta mas alta, criterio interno): {k_elegido}")
    print(f"Silueta del k elegido: {silueta_k_elegido:.4f}")
    print(f"ARI (adjusted_rand_score) contra clases reales: {ari:.4f}")
    print(f"Indice de Rand sin ajustar: {rand_sin_ajustar:.4f}")
    print(f"Pureza (clase mayoritaria por cluster): {pureza:.4f}")
    print("Tabla de contingencia (clusters x clases reales):")
    print("".ljust(14) + "".join(nm.ljust(12) for nm in class_names) + "total")
    for i in range(n_clusters):
        fila = f"cluster_{i}".ljust(14)
        fila += "".join(str(int(cont[i, j])).ljust(12) for j in range(n_clases))
        fila += str(int(cont[i, :].sum()))
        print(fila)
    print(f"Tamanos de clusters (k={k_elegido}): {[int(cont[i, :].sum()) for i in range(n_clusters)]}")
    print(f"Tamanos de clases reales: {[int(cont[:, j].sum()) for j in range(n_clases)]}")
    print("Resultados guardados en resultados.json")


if __name__ == "__main__":
    main()
