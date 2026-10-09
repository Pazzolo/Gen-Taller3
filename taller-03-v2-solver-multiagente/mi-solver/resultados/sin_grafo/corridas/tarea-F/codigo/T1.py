# -*- coding: utf-8 -*-
"""
SUBTAREA T1: Cargar el conjunto Wine, reportar ejemplos/atributos/clases y
estandarizar los atributos con StandardScaler (sin usar las etiquetas).
Guarda todas las cifras en resultados.json e imprime las principales.
"""

import json

import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler


def main():
    # ------------------------------------------------------------------
    # 1) Cargar el conjunto Wine
    # ------------------------------------------------------------------
    wine = load_wine()
    X = wine.data          # matriz de atributos (n_ejemplos x n_atributos)
    y = wine.target        # etiquetas (solo para REPORTAR; no se usan para escalar)

    n_ejemplos = int(X.shape[0])
    n_atributos = int(X.shape[1])
    nombres_clases = [str(c) for c in wine.target_names]
    n_clases = int(len(nombres_clases))

    # Distribución de ejemplos por clase (solo informativa)
    distribucion_clases = {
        nombres_clases[int(c)]: int(np.sum(y == c)) for c in np.unique(y)
    }

    # ------------------------------------------------------------------
    # 2) Escalas originales (para evidenciar que las escalas son distintas)
    # ------------------------------------------------------------------
    medias_orig = X.mean(axis=0)
    desvs_orig = X.std(axis=0)

    media_min_original = float(np.min(medias_orig))
    media_max_original = float(np.max(medias_orig))
    desv_min_original = float(np.min(desvs_orig))
    desv_max_original = float(np.max(desvs_orig))

    # ------------------------------------------------------------------
    # 3) Estandarizar con StandardScaler (ajustado SOLO con X, sin etiquetas)
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)  # las etiquetas y no intervienen en absoluto

    # Verificación de la estandarización
    medias_std = X_std.mean(axis=0)
    desvs_std = X_std.std(axis=0)

    media_max_abs_tras_estandarizar = float(np.max(np.abs(medias_std)))
    desv_min_tras_estandarizar = float(np.min(desvs_std))
    desv_max_tras_estandarizar = float(np.max(desvs_std))

    # Cifras por atributo (todas como float de Python, sin redondear)
    medias_orig_por_atributo = [float(m) for m in medias_orig]
    desvs_orig_por_atributo = [float(d) for d in desvs_orig]
    medias_std_por_atributo = [float(m) for m in medias_std]
    desvs_std_por_atributo = [float(d) for d in desvs_std]

    # ------------------------------------------------------------------
    # 4) Guardar TODAS las cifras calculadas en resultados.json
    # ------------------------------------------------------------------
    resultados = {
        "subtarea": "T1_carga_y_estandarizacion_wine",
        "n_ejemplos": n_ejemplos,
        "n_atributos": n_atributos,
        "n_clases": n_clases,
        "nombres_clases": nombres_clases,
        "distribucion_clases": distribucion_clases,
        "media_min_original": media_min_original,
        "media_max_original": media_max_original,
        "desv_min_original": desv_min_original,
        "desv_max_original": desv_max_original,
        "media_max_abs_tras_estandarizar": media_max_abs_tras_estandarizar,
        "desv_min_tras_estandarizar": desv_min_tras_estandarizar,
        "desv_max_tras_estandarizar": desv_max_tras_estandarizar,
        "medias_originales_por_atributo": medias_orig_por_atributo,
        "desviaciones_originales_por_atributo": desvs_orig_por_atributo,
        "medias_tras_estandarizar_por_atributo": medias_std_por_atributo,
        "desviaciones_tras_estandarizar_por_atributo": desvs_std_por_atributo,
    }

    with open("resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # 5) Imprimir las cifras principales con 4 decimales
    # ------------------------------------------------------------------
    print("=== T1: Carga y estandarizacion del conjunto Wine ===")
    print(f"Ejemplos: {n_ejemplos}")
    print(f"Atributos: {n_atributos}")
    print(f"Clases: {n_clases} -> {nombres_clases}")
    print(f"Distribucion por clase: {distribucion_clases}")
    print(
        f"Escalas originales: medias en [{media_min_original:.4f}, "
        f"{media_max_original:.4f}], desviaciones en [{desv_min_original:.4f}, "
        f"{desv_max_original:.4f}]"
    )
    print(
        f"Tras StandardScaler: |media| maxima = {media_max_abs_tras_estandarizar:.4f}, "
        f"desviacion minima = {desv_min_tras_estandarizar:.4f}, "
        f"desviacion maxima = {desv_max_tras_estandarizar:.4f}"
    )
    print("Cifras guardadas en resultados.json")


if __name__ == "__main__":
    main()
