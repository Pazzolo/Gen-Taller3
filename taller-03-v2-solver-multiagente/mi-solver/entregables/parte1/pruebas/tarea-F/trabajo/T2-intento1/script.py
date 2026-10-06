import json
import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score


def main():
    # ------------------------------------------------------------------
    # Datos y estandarizacion (replica independiente de T1)
    # ------------------------------------------------------------------
    wine = load_wine()
    X = wine.data
    y = wine.target  # clases reales: SOLO se usan al final, para el ARI

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    n_ejemplos, n_atributos = X_std.shape

    # ------------------------------------------------------------------
    # Parte 2: silueta por k (criterio interno, sin etiquetas reales)
    # ------------------------------------------------------------------
    k_values = list(range(2, 9))  # k en 2..8
    siluetas = []
    inercias = []
    etiquetas_por_k = {}

    for k in k_values:
        km = KMeans(n_clusters=k, n_init=10, random_state=0)
        labels_k = km.fit_predict(X_std)
        sil_k = float(silhouette_score(X_std, labels_k, metric="euclidean"))
        siluetas.append(sil_k)
        inercias.append(float(km.inertia_))
        etiquetas_por_k[k] = labels_k

    # Eleccion de k: silueta mas alta (sin usar etiquetas reales)
    idx_mejor = int(np.argmax(siluetas))
    mejor_k = int(k_values[idx_mejor])
    mejor_silueta = float(siluetas[idx_mejor])

    # Recien al final se miran las clases reales: ARI del k elegido
    labels_mejor = etiquetas_por_k[mejor_k]
    ari_mejor_k = float(adjusted_rand_score(y, labels_mejor))
    conteo_clusters = np.bincount(labels_mejor, minlength=mejor_k)

    # ------------------------------------------------------------------
    # Guardar todas las cifras en resultados.json
    # ------------------------------------------------------------------
    resultados = {
        "subtarea": "T2_kmeans_silueta_por_k",
        "n_ejemplos": int(n_ejemplos),
        "n_atributos": int(n_atributos),
        "k_values": [int(k) for k in k_values],
        "silueta_por_k": {str(k): float(s) for k, s in zip(k_values, siluetas)},
        "inercia_por_k": {str(k): float(i) for k, i in zip(k_values, inercias)},
        "mejor_k": mejor_k,
        "mejor_silueta": mejor_silueta,
        "ari_mejor_k": ari_mejor_k,
        "conteo_clusters_mejor_k": {
            str(c): int(n) for c, n in enumerate(conteo_clusters)
        },
        "n_clases_reales": int(len(np.unique(y))),
    }

    with open("resultados.json", "w") as f:
        json.dump(resultados, f, indent=2)

    # ------------------------------------------------------------------
    # Salida principal con 4 decimales
    # ------------------------------------------------------------------
    print("=== T2: KMeans y silueta por k (datos estandarizados de T1) ===")
    print(f"Ejemplos: {n_ejemplos}  Atributos: {n_atributos}")
    print("Tabla silueta-por-k (criterio interno, sin etiquetas reales):")
    print("  k    silueta    inercia")
    for k, s, ine in zip(k_values, siluetas, inercias):
        print(f"  {k}   {s:.4f}   {ine:.4f}")
    print(f"k elegido (silueta mas alta): {mejor_k}")
    print(f"Silueta del k elegido: {mejor_silueta:.4f}")
    print(f"Tamanos de clusters (k={mejor_k}): {[int(n) for n in conteo_clusters]}")
    print(
        f"ARI contra clases reales (solo al final, no para elegir k): {ari_mejor_k:.4f}"
    )
    print("Resultados guardados en resultados.json")


if __name__ == "__main__":
    main()
