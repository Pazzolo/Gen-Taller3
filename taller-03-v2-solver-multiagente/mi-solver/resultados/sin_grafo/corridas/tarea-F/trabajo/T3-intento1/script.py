# -*- coding: utf-8 -*-
"""
SUBTAREA T3: Comparar los grupos del clustering KMeans (con el k elegido en T2)
con las clases reales de Wine mediante el índice de Rand ajustado (ARI).

El script es independiente: recarga los datos, reestandariza y vuelve a elegir k
por silueta (mismo procedimiento y semillas que T2) antes de calcular el ARI.
Las etiquetas reales NO se usan para elegir k; solo para evaluar el clustering.
"""

import json
import numpy as np

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score


def main():
    # ------------------------------------------------------------------
    # Parte 1: datos y escala (igual que en T1/T2)
    # ------------------------------------------------------------------
    wine = load_wine()
    X = wine.data
    y = wine.target

    n_ejemplos, n_atributos = X.shape
    clases_reales = np.unique(y)
    n_clases = len(clases_reales)

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # ------------------------------------------------------------------
    # Parte 2: re-elección de k por silueta (k en 2..8, sin usar etiquetas)
    # ------------------------------------------------------------------
    rango_k = list(range(2, 9))
    silueta_por_k = {}
    inercia_por_k = {}
    for k in rango_k:
        km = KMeans(n_clusters=k, n_init=10, random_state=0)
        etiquetas_k = km.fit_predict(X_std)
        silueta_por_k[k] = float(silhouette_score(X_std, etiquetas_k))
        inercia_por_k[k] = float(km.inertia_)

    k_optimo = max(silueta_por_k, key=silueta_por_k.get)
    silueta_maxima = silueta_por_k[k_optimo]

    # ------------------------------------------------------------------
    # Parte 3: ARI entre etiquetas del clustering (k óptimo) y clases reales
    # ------------------------------------------------------------------
    km_final = KMeans(n_clusters=k_optimo, n_init=10, random_state=0)
    etiquetas = km_final.fit_predict(X_std)

    ari = float(adjusted_rand_score(y, etiquetas))

    # Tabla de contingencia clusters x clases reales (comparación detallada)
    tabla_contingencia = {}
    for c in range(k_optimo):
        fila = {}
        for cl in clases_reales:
            fila[str(int(cl))] = int(np.sum((etiquetas == c) & (y == cl)))
        tabla_contingencia[str(c)] = fila

    # Tamaños de clusters y de clases reales
    tamano_clusters = {str(c): int(np.sum(etiquetas == c)) for c in range(k_optimo)}
    tamano_clases = {str(int(cl)): int(np.sum(y == cl)) for cl in clases_reales}

    # Pureza del clustering (cifra complementaria al ARI)
    aciertos_maximos = 0
    for c in range(k_optimo):
        cuentas = [tabla_contingencia[str(c)][str(int(cl))] for cl in clases_reales]
        aciertos_maximos += max(cuentas)
    pureza = float(aciertos_maximos / n_ejemplos)

    # ------------------------------------------------------------------
    # Guardar TODAS las cifras en resultados.json (un solo objeto)
    # ------------------------------------------------------------------
    resultados = {
        "subtarea": "T3_ari_kmeans_vs_clases_wine",
        "n_ejemplos": int(n_ejemplos),
        "n_atributos": int(n_atributos),
        "n_clases": int(n_clases),
        "rango_k": [int(k) for k in rango_k],
        "silueta_por_k": {str(k): v for k, v in silueta_por_k.items()},
        "inercia_por_k": {str(k): v for k, v in inercia_por_k.items()},
        "k_optimo": int(k_optimo),
        "silueta_maxima": float(silueta_maxima),
        "ari": ari,
        "pureza": pureza,
        "tabla_contingencia_clusters_clases": tabla_contingencia,
        "tamano_clusters": tamano_clusters,
        "tamano_clases_reales": tamano_clases,
    }

    with open("resultados.json", "w") as f:
        json.dump(resultados, f, indent=2)

    # ------------------------------------------------------------------
    # Impresión de cifras principales (4 decimales)
    # ------------------------------------------------------------------
    print("=== T3: ARI entre clustering KMeans y clases reales de Wine ===")
    print("Datos: %d ejemplos, %d atributos, %d clases reales" % (n_ejemplos, n_atributos, n_clases))
    print("k elegido por silueta (recalculado como en T2, sin usar etiquetas): k=%d (silueta=%.4f)"
          % (k_optimo, silueta_maxima))
    print("Indice de Rand ajustado (ARI) clustering vs wine.target: %.4f" % ari)
    print("Pureza del clustering (cifra complementaria): %.4f" % pureza)
    print("Tabla de contingencia (filas = cluster, columnas = clase real):")
    encabezado = "  cluster | " + " | ".join("clase %d" % int(cl) for cl in clases_reales) + " | total"
    print(encabezado)
    for c in range(k_optimo):
        fila = [tabla_contingencia[str(c)][str(int(cl))] for cl in clases_reales]
        print("  %7d | " % c + " | ".join("%7d" % v for v in fila) + " | %5d" % tamano_clusters[str(c)])
    print("Tamanos de clases reales: " + ", ".join("clase %d: %d" % (int(cl), tamano_clases[str(int(cl))]) for cl in clases_reales))
    print("Cifras guardadas en resultados.json")


if __name__ == "__main__":
    main()
