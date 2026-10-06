# -*- coding: utf-8 -*-
"""
T4 — Experimento de control sin estandarizar (Tarea F, Wine).

Repite el barrido k=2..8 con KMeans(n_clusters=k, n_init=10, random_state=0)
sobre los datos CRUDOS de T1 (load_wine, sin StandardScaler). Para cada k
calcula la silueta, elige k por máxima silueta (criterio interno, sin usar
etiquetas) y al final compara contra las clases reales con ARI.

Para cuantificar el efecto de omitir StandardScaler, el mismo script
recalcula como referencia el barrido idéntico sobre los datos estandarizados
y reporta las diferencias (k elegido, silueta y ARI).

Figura: no se pide. Todas las cifras se guardan en resultados.json.
"""

import json
import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

RANDOM_STATE = 0
N_INIT = 10
K_MIN, K_MAX = 2, 8


def barrido_kmeans(X, y, ks):
    """Barrido de k con la configuración fija del enunciado.
    Devuelve, para cada k: silueta, ARI contra clases reales, inercia,
    tamaños de clúster y número de iteraciones."""
    resultados = {}
    for k in ks:
        km = KMeans(n_clusters=k, n_init=N_INIT, random_state=RANDOM_STATE)
        etiquetas = km.fit_predict(X)
        resultados[str(k)] = {
            "silueta": float(silhouette_score(X, etiquetas)),
            "ari": float(adjusted_rand_score(y, etiquetas)),
            "inercia": float(km.inertia_),
            "tamanos_clusters": [int(t) for t in np.bincount(etiquetas, minlength=k)],
            "n_iteraciones": int(km.n_iter_),
        }
    return resultados


def elegir_k_por_silueta(res):
    """Criterio interno: k con mayor silueta media (no usa etiquetas)."""
    return int(max(res, key=lambda k: res[k]["silueta"]))


def main():
    # ---------- Datos crudos de T1 (sin estandarizar) ----------
    datos = load_wine()
    X_crudo = datos.data.astype(float)
    y = datos.target
    clases_reales = [str(s) for s in datos.target_names]
    ks = list(range(K_MIN, K_MAX + 1))

    # ---------- Barrido sobre datos CRUDOS (control) ----------
    crudo = barrido_kmeans(X_crudo, y, ks)
    k_crudo = elegir_k_por_silueta(crudo)
    sil_crudo = crudo[str(k_crudo)]["silueta"]
    ari_crudo = crudo[str(k_crudo)]["ari"]

    # ---------- Referencia: mismo barrido con StandardScaler ----------
    # (recalculado aquí porque el script es independiente; el escalador se
    #  ajusta solo con X, sin etiquetas, igual que en T1)
    scaler = StandardScaler().fit(X_crudo)
    X_escalado = scaler.transform(X_crudo)
    escalado = barrido_kmeans(X_escalado, y, ks)
    k_escalado = elegir_k_por_silueta(escalado)
    sil_escalado = escalado[str(k_escalado)]["silueta"]
    ari_escalado = escalado[str(k_escalado)]["ari"]

    # ---------- Extra: valores en k=3 (número de clases reales) ----------
    sil_crudo_k3 = crudo["3"]["silueta"]
    ari_crudo_k3 = crudo["3"]["ari"]
    sil_escalado_k3 = escalado["3"]["silueta"]
    ari_escalado_k3 = escalado["3"]["ari"]

    # ---------- Efecto de omitir StandardScaler ----------
    resultados = {
        "subtarea": "T4_control_sin_estandarizar",
        "descripcion": ("Barrido k=2..8 con KMeans(n_clusters=k, n_init=10, "
                        "random_state=0) sobre los datos crudos de Wine (sin "
                        "StandardScaler); referencia con datos estandarizados."),
        "n_ejemplos": int(X_crudo.shape[0]),
        "n_atributos": int(X_crudo.shape[1]),
        "n_clases": int(len(np.unique(y))),
        "clases_reales": clases_reales,
        "k_values": [int(k) for k in ks],

        # --- Datos crudos (control sin estandarizar) ---
        "silueta_por_k_crudo": {k: v["silueta"] for k, v in crudo.items()},
        "ari_por_k_crudo": {k: v["ari"] for k, v in crudo.items()},
        "inercia_por_k_crudo": {k: v["inercia"] for k, v in crudo.items()},
        "tamanos_clusters_por_k_crudo": {k: v["tamanos_clusters"] for k, v in crudo.items()},
        "n_iteraciones_por_k_crudo": {k: v["n_iteraciones"] for k, v in crudo.items()},
        "k_elegido_crudo": int(k_crudo),
        "silueta_k_elegido_crudo": float(sil_crudo),
        "ari_k_elegido_crudo": float(ari_crudo),
        "silueta_crudo_en_k3": float(sil_crudo_k3),
        "ari_crudo_en_k3": float(ari_crudo_k3),

        # --- Referencia estandarizada ---
        "silueta_por_k_escalado": {k: v["silueta"] for k, v in escalado.items()},
        "ari_por_k_escalado": {k: v["ari"] for k, v in escalado.items()},
        "inercia_por_k_escalado": {k: v["inercia"] for k, v in escalado.items()},
        "tamanos_clusters_por_k_escalado": {k: v["tamanos_clusters"] for k, v in escalado.items()},
        "n_iteraciones_por_k_escalado": {k: v["n_iteraciones"] for k, v in escalado.items()},
        "k_elegido_escalado": int(k_escalado),
        "silueta_k_elegido_escalado": float(sil_escalado),
        "ari_k_elegido_escalado": float(ari_escalado),
        "silueta_escalado_en_k3": float(sil_escalado_k3),
        "ari_escalado_en_k3": float(ari_escalado_k3),

        # --- Cuantificación del efecto ---
        "efecto_k_elegido_cambia": bool(k_crudo != k_escalado),
        "efecto_delta_k": int(k_crudo - k_escalado),
        "efecto_delta_silueta_k_elegido": float(sil_crudo - sil_escalado),
        "efecto_delta_ari_k_elegido": float(ari_crudo - ari_escalado),
        "efecto_delta_silueta_en_k3": float(sil_crudo_k3 - sil_escalado_k3),
        "efecto_delta_ari_en_k3": float(ari_crudo_k3 - ari_escalado_k3),
    }

    with open("resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    # ---------- Impresión de cifras principales (4 decimales) ----------
    print("=== T4: Control sin estandarizar (datos crudos de Wine) ===")
    print(f"n={resultados['n_ejemplos']}, d={resultados['n_atributos']}, "
          f"clases reales={resultados['n_clases']}")

    print("\nBarrido k=2..8 sobre datos CRUDOS (sin StandardScaler):")
    print("   k   silueta     ARI")
    for k in ks:
        kk = str(k)
        print(f"   {k}   {crudo[kk]['silueta']:.4f}    {crudo[kk]['ari']:.4f}")
    print(f"\nk elegido en crudo (max silueta): {k_crudo}")
    print(f"  silueta: {sil_crudo:.4f}")
    print(f"  ARI vs clases reales: {ari_crudo:.4f}")
    print(f"  tamanos de clusters: {crudo[str(k_crudo)]['tamanos_clusters']}")

    print("\nReferencia: mismo barrido con datos ESTANDARIZADOS:")
    print("   k   silueta     ARI")
    for k in ks:
        kk = str(k)
        print(f"   {k}   {escalado[kk]['silueta']:.4f}    {escalado[kk]['ari']:.4f}")
    print(f"\nk elegido en escalado (max silueta): {k_escalado}")
    print(f"  silueta: {sil_escalado:.4f}")
    print(f"  ARI vs clases reales: {ari_escalado:.4f}")

    print("\nEfecto de omitir StandardScaler:")
    print(f"  k elegido: {k_crudo} (crudo) vs {k_escalado} (escalado) -> "
          f"cambia: {k_crudo != k_escalado}")
    print(f"  ARI en k elegido: {ari_crudo:.4f} (crudo) vs {ari_escalado:.4f} "
          f"(escalado) -> delta {ari_crudo - ari_escalado:.4f}")
    print(f"  ARI en k=3:       {ari_crudo_k3:.4f} (crudo) vs {ari_escalado_k3:.4f} "
          f"(escalado) -> delta {ari_crudo_k3 - ari_escalado_k3:.4f}")
    print(f"  Silueta en k elegido: {sil_crudo:.4f} (crudo) vs {sil_escalado:.4f} "
          f"(escalado) -> delta {sil_crudo - sil_escalado:.4f}")

    print("\nCifras guardadas en resultados.json")


if __name__ == "__main__":
    main()
