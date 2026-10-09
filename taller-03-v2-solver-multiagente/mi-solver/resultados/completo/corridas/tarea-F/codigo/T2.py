# -*- coding: utf-8 -*-
"""
T2: Eleccion del numero de grupos k para KMeans sobre los datos Wine
estandarizados (recalculo de T1), usando el coeficiente de silueta como
criterio interno (las etiquetas reales NO se usan para elegir k).

- KMeans(n_clusters=k, n_init=10, random_state=0) para k = 2..8
- Coeficiente de silueta para cada k (sobre todos los datos)
- k elegido = silueta maxima
- ARI contra las clases reales solo como diagnostico final (despues de elegir k)
- Todas las cifras se guardan en resultados.json
"""

import json
import os

import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

RUTA_RESULTADOS = "resultados.json"


def cargar_resultados_previos():
    """Carga resultados.json si existe, para no perder cifras de subtareas previas."""
    if os.path.exists(RUTA_RESULTADOS):
        try:
            with open(RUTA_RESULTADOS, "r", encoding="utf-8") as f:
                datos = json.load(f)
            if isinstance(datos, dict):
                return datos
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def guardar_resultados(resultados):
    with open(RUTA_RESULTADOS, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)


def main():
    print("=== T2: KMeans + coeficiente de silueta (datos Wine estandarizados) ===")

    # ---------- Datos y estandarizacion (recalculo independiente de T1) ----------
    wine = load_wine()
    X = wine.data
    y = wine.target  # SOLO para el ARI final; nunca interviene en la eleccion de k

    scaler = StandardScaler()
    X_esc = scaler.fit_transform(X)  # el escalador se ajusta solo con las features
    n_ejemplos, n_atributos = X_esc.shape
    print(f"Datos estandarizados: {n_ejemplos} ejemplos x {n_atributos} atributos")

    # ---------- KMeans para cada k en 2..8 y silueta ----------
    ks = list(range(2, 9))
    siluetas = {}
    inercias = {}
    etiquetas_por_k = {}
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=0)
        labels = km.fit_predict(X_esc)
        sil = silhouette_score(X_esc, labels, metric="euclidean")
        siluetas[k] = float(sil)
        inercias[k] = float(km.inertia_)
        etiquetas_por_k[k] = labels

    # ---------- Eleccion de k: maxima silueta (criterio interno, sin etiquetas) ----------
    k_elegido = max(siluetas, key=siluetas.get)
    silueta_maxima = siluetas[k_elegido]

    # ---------- Comparacion con clases reales SOLO despues de elegir k ----------
    labels_elegido = etiquetas_por_k[k_elegido]
    ari_elegido = float(adjusted_rand_score(y, labels_elegido))
    conteos = np.bincount(labels_elegido, minlength=k_elegido)
    tamanos_clusters = {int(c): int(n_c) for c, n_c in enumerate(conteos)}

    # ---------- Tabla silueta-vs-k para el reporte ----------
    print()
    print("Tabla silueta-vs-k (KMeans, n_init=10, random_state=0, datos estandarizados):")
    print(f"{'k':>3} | {'silueta':>10} | {'inercia':>12}")
    print("-" * 34)
    for k in ks:
        marca = "  <- elegido" if k == k_elegido else ""
        print(f"{k:>3} | {siluetas[k]:>10.4f} | {inercias[k]:>12.4f}{marca}")

    print()
    print(f"k elegido (silueta mas alta): {k_elegido}")
    print(f"Silueta maxima: {silueta_maxima:.4f}")
    print(f"ARI contra clases reales (solo diagnostico final): {ari_elegido:.4f}")
    print(f"Tamanos de los clusters con k={k_elegido}: {tamanos_clusters}")

    # ---------- Guardar TODAS las cifras en resultados.json ----------
    resultados = cargar_resultados_previos()
    resultados.update(
        {
            "t2_criterio_eleccion": "coeficiente_de_silueta_maximo",
            "t2_n_ejemplos": int(n_ejemplos),
            "t2_n_atributos": int(n_atributos),
            "t2_rango_k": [int(min(ks)), int(max(ks))],
            "t2_kmeans_params": {"n_init": 10, "random_state": 0},
            "t2_silueta_por_k": {str(k): float(siluetas[k]) for k in ks},
            "t2_inercia_por_k": {str(k): float(inercias[k]) for k in ks},
            "t2_k_elegido": int(k_elegido),
            "t2_silueta_maxima": float(silueta_maxima),
            "t2_ari_k_elegido": float(ari_elegido),
            "t2_tamanos_clusters_k_elegido": tamanos_clusters,
        }
    )
    guardar_resultados(resultados)
    print()
    print("Cifras guardadas en resultados.json")


if __name__ == "__main__":
    main()
