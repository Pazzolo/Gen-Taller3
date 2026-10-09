import json
import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

RUTA_JSON = "resultados.json"


def main():
    # ---------- Datos y escala (recrea la parte de T1) ----------
    wine = load_wine()
    X = wine.data
    y_true = wine.target  # etiquetas reales: solo referencia, NO se usan para elegir k

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    n_ejemplos, n_atributos = X.shape
    n_clases = len(np.unique(y_true))

    # ---------- Parte 2: silueta por k (k = 2..8) ----------
    ks = list(range(2, 9))
    siluetas = {}
    inercias = {}
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=0)
        etiquetas = km.fit_predict(X_std)
        siluetas[k] = float(silhouette_score(X_std, etiquetas, metric="euclidean"))
        inercias[k] = float(km.inertia_)

    # Eleccion de k: maxima silueta (sin usar etiquetas reales)
    k_optimo = max(siluetas, key=siluetas.get)
    silueta_optima = siluetas[k_optimo]

    # Referencia informativa (no interviene en la eleccion de k)
    silueta_etiquetas_reales = float(silhouette_score(X_std, y_true, metric="euclidean"))

    # ---------- Guardar todas las cifras en resultados.json ----------
    resultados_nuevos = {
        "subtarea": "T2_kmeans_silueta_eleccion_k",
        "n_ejemplos": int(n_ejemplos),
        "n_atributos": int(n_atributos),
        "n_clases": int(n_clases),
        "rango_k": [int(k) for k in ks],
        "silueta_por_k": {str(k): siluetas[k] for k in ks},
        "inercia_por_k": {str(k): inercias[k] for k in ks},
        "tabla_silueta": [{"k": int(k), "silueta": siluetas[k]} for k in ks],
        "k_optimo": int(k_optimo),
        "silueta_maxima": float(silueta_optima),
        "silueta_referencia_etiquetas_reales": silueta_etiquetas_reales,
    }

    try:
        with open(RUTA_JSON, "r") as f:
            resultados = json.load(f)
        if not isinstance(resultados, dict):
            resultados = {}
    except (FileNotFoundError, json.JSONDecodeError):
        resultados = {}

    resultados.update(resultados_nuevos)

    with open(RUTA_JSON, "w") as f:
        json.dump(resultados, f, indent=2)

    # ---------- Impresion de cifras principales (4 decimales) ----------
    print("=== T2: KMeans + coeficiente de silueta por k (Wine estandarizado) ===")
    print(f"Ejemplos: {n_ejemplos} | Atributos: {n_atributos} | Clases reales: {n_clases} (no usadas para elegir k)")
    print("Tabla silueta por k:")
    for k in ks:
        print(f"  k={k}: silueta={siluetas[k]:.4f} (inercia={inercias[k]:.4f})")
    print(f"k optimo (silueta mas alta): k={k_optimo} con silueta={silueta_optima:.4f}")
    print(f"Referencia (no usada para elegir k): silueta con etiquetas reales = {silueta_etiquetas_reales:.4f}")
    print("Cifras guardadas en resultados.json")


if __name__ == "__main__":
    main()
