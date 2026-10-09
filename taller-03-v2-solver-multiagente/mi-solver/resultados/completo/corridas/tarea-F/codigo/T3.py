# SUBTAREA T3: Comparar los grupos de KMeans (con el k elegido en T2 por silueta)
# con las clases reales del Wine dataset mediante el índice de Rand ajustado (ARI).
# Script autocontenido: reproduce la elección de k de T2 (criterio interno, sin etiquetas)
# y recién al final usa las clases reales solo para el diagnóstico (ARI).

import json

import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

# Parámetros fijados por el enunciado / resultados de T2 (no se tunea nada con etiquetas)
RANDOM_STATE = 0
N_INIT = 10
K_RANGE = list(range(2, 9))  # mismo rango que T2: k = 2..8


def main():
    # ------------------------------------------------------------------
    # 1) Datos: Wine dataset, estandarizados (igual que en T2)
    # ------------------------------------------------------------------
    wine = load_wine()
    X = wine.data
    y = wine.target  # clases reales: SOLO para el diagnóstico final (ARI)

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)  # escalador ajustado solo con las features

    n_ejemplos, n_atributos = X.shape
    n_clases = len(wine.target_names)

    # ------------------------------------------------------------------
    # 2) Reproducir T2: silueta por k (criterio interno, sin usar etiquetas)
    # ------------------------------------------------------------------
    siluetas = {}
    inercias = {}
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=N_INIT, random_state=RANDOM_STATE)
        labels_k = km.fit_predict(X_std)
        siluetas[k] = float(silhouette_score(X_std, labels_k))
        inercias[k] = float(km.inertia_)

    # Elección de k: silueta máxima (mismo criterio que T2)
    k_elegido = int(max(siluetas, key=siluetas.get))
    silueta_max = float(siluetas[k_elegido])

    # ------------------------------------------------------------------
    # 3) KMeans final con el k elegido y ARI contra las clases reales
    # ------------------------------------------------------------------
    km_final = KMeans(n_clusters=k_elegido, n_init=N_INIT, random_state=RANDOM_STATE)
    labels_final = km_final.fit_predict(X_std)

    ari = float(adjusted_rand_score(y, labels_final))

    # Tamaños de los clusters con el k elegido
    tamanos = {str(int(c)): int(np.sum(labels_final == c))
               for c in np.unique(labels_final)}

    # Tabla de contingencia: clusters vs clases reales (detalle del ARI)
    contingencia = {}
    for c in np.unique(labels_final):
        fila = {}
        for idx_clase, nombre_clase in enumerate(wine.target_names):
            fila[str(nombre_clase)] = int(np.sum((labels_final == c) & (y == idx_clase)))
        contingencia[str(int(c))] = fila

    # ARI para cada k (diagnóstico adicional; la elección de k NO usa esto)
    ari_por_k = {}
    for k in K_RANGE:
        km_k = KMeans(n_clusters=k, n_init=N_INIT, random_state=RANDOM_STATE)
        labels_k = km_k.fit_predict(X_std)
        ari_por_k[str(k)] = float(adjusted_rand_score(y, labels_k))

    # ------------------------------------------------------------------
    # 4) Guardar TODAS las cifras en resultados.json
    # ------------------------------------------------------------------
    resultados = {
        "t3_criterio_eleccion": "coeficiente_de_silueta_maximo",
        "t3_n_ejemplos": int(n_ejemplos),
        "t3_n_atributos": int(n_atributos),
        "t3_n_clases_reales": int(n_clases),
        "t3_nombres_clases": [str(n) for n in wine.target_names],
        "t3_kmeans_params": {"n_init": N_INIT, "random_state": RANDOM_STATE},
        "t3_rango_k": [int(min(K_RANGE)), int(max(K_RANGE))],
        "t3_silueta_por_k": {str(k): float(v) for k, v in siluetas.items()},
        "t3_inercia_por_k": {str(k): float(v) for k, v in inercias.items()},
        "t3_k_elegido": int(k_elegido),
        "t3_silueta_maxima": float(silueta_max),
        "t3_ari_k_elegido": float(ari),
        "t3_ari_por_k": {str(k): float(v) for k, v in ari_por_k.items()},
        "t3_tamanos_clusters_k_elegido": tamanos,
        "t3_contingencia_clusters_vs_clases": contingencia,
    }

    with open("resultados.json", "w") as f:
        json.dump({"resultados": resultados}, f, indent=2)

    # ------------------------------------------------------------------
    # 5) Reporte por consola (cifras principales con 4 decimales)
    # ------------------------------------------------------------------
    print("=== T3: ARI de KMeans (k elegido en T2) contra las clases reales del Wine ===")
    print(f"Datos estandarizados: {n_ejemplos} ejemplos x {n_atributos} atributos")
    print(f"Clases reales: {n_clases} ({', '.join(wine.target_names)})")

    print(f"\nTabla silueta-vs-k (KMeans, n_init={N_INIT}, random_state={RANDOM_STATE}):")
    print("  k |    silueta")
    print("------------------")
    for k in sorted(siluetas):
        marca = "  <- elegido" if k == k_elegido else ""
        print(f"  {k} |     {siluetas[k]:.4f}{marca}")

    print(f"\nk elegido (silueta mas alta): {k_elegido}")
    print(f"Silueta maxima: {silueta_max:.4f}")
    print(f"ARI contra clases reales (k={k_elegido}): {ari:.4f}")
    print(f"Tamanos de los clusters con k={k_elegido}: "
          + "{" + ", ".join(f"{c}: {t}" for c, t in tamanos.items()) + "}")

    print("\nTabla de contingencia (clusters vs clases reales):")
    print("cluster | " + " | ".join(str(n) for n in wine.target_names))
    for c in sorted(contingencia, key=int):
        fila = contingencia[c]
        print(f"      {c} | " + " | ".join(str(fila[str(n)]) for n in wine.target_names))

    print("\nARI por k (solo diagnostico; no se uso para elegir k):")
    for k in sorted(ari_por_k, key=int):
        print(f"  k={k}: ARI={ari_por_k[k]:.4f}")

    print("\nCifras guardadas en resultados.json")


if __name__ == "__main__":
    main()
