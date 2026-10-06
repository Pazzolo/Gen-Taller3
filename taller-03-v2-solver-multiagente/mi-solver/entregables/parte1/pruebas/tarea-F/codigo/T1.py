# -*- coding: utf-8 -*-
"""
SUBTAREA T1 — Carga del conjunto Wine, reporte de dimensiones y estandarización.

- Carga sklearn.datasets.load_wine.
- Reporta número de ejemplos, atributos y clases.
- Estandariza los atributos con StandardScaler (ajustado SOLO con X; las
  etiquetas y no se usan para nada del preprocesamiento ni para agrupar).
- Guarda X escalado y y en un archivo .npz para subtareas posteriores.
- Guarda todas las cifras calculadas en resultados.json (floats de Python,
  sin redondear) e imprime las principales con 4 decimales.
- Figura: no se requiere en esta subtarea.
"""

import json
import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

ARCHIVO_DATOS = "datos_wine_escalado.npz"
ARCHIVO_RESULTADOS = "resultados.json"


def main():
    # ------------------------------------------------------------------
    # 1) Carga de datos
    # ------------------------------------------------------------------
    wine = load_wine()
    X = wine.data          # matriz (n_ejemplos, n_atributos)
    y = wine.target        # etiquetas reales (solo para comparar al final)

    n_ejemplos, n_atributos = X.shape
    clases_reales = [str(c) for c in wine.target_names]
    etiquetas_unicas = np.unique(y)
    n_clases = int(etiquetas_unicas.size)

    # Conteo de ejemplos por clase real (informativo; no se usa para agrupar)
    conteo_por_clase = {
        clases_reales[int(k)]: int(np.sum(y == k)) for k in etiquetas_unicas
    }

    # ------------------------------------------------------------------
    # 2) Estadísticas ANTES de escalar (para evidenciar las escalas distintas)
    # ------------------------------------------------------------------
    medias_antes = X.mean(axis=0)
    stds_antes = X.std(axis=0, ddof=0)
    minimos_antes = X.min(axis=0)
    maximos_antes = X.max(axis=0)

    # ------------------------------------------------------------------
    # 3) Estandarización con StandardScaler
    #    (fit solo sobre X; y no interviene en absoluto)
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X_escalado = scaler.fit_transform(X)

    medias_despues = X_escalado.mean(axis=0)
    stds_despues = X_escalado.std(axis=0, ddof=0)

    # ------------------------------------------------------------------
    # 4) Guardar X escalado y y para las subtareas siguientes
    # ------------------------------------------------------------------
    np.savez(ARCHIVO_DATOS, X_escalado=X_escalado, y=y)

    # ------------------------------------------------------------------
    # 5) Guardar TODAS las cifras en resultados.json (floats de Python)
    # ------------------------------------------------------------------
    resultados = {
        "subtarea": "T1_carga_y_estandarizacion_wine",
        "n_ejemplos": int(n_ejemplos),
        "n_atributos": int(n_atributos),
        "n_clases": int(n_clases),
        "clases_reales": clases_reales,
        "conteo_por_clase": conteo_por_clase,
        "nombres_atributos": [str(a) for a in wine.feature_names],
        # Escalas antes de estandarizar
        "media_global_antes": float(X.mean()),
        "std_global_antes": float(X.std()),
        "min_media_atributo_antes": float(medias_antes.min()),
        "max_media_atributo_antes": float(medias_antes.max()),
        "min_std_atributo_antes": float(stds_antes.min()),
        "max_std_atributo_antes": float(stds_antes.max()),
        "min_min_atributo_antes": float(minimos_antes.min()),
        "max_max_atributo_antes": float(maximos_antes.max()),
        "medias_por_atributo_antes": [float(v) for v in medias_antes],
        "stds_por_atributo_antes": [float(v) for v in stds_antes],
        # Parámetros aprendidos por el escalador
        "scaler_mean": [float(v) for v in scaler.mean_],
        "scaler_scale": [float(v) for v in scaler.scale_],
        # Verificación después de estandarizar
        "max_abs_media_atributo_despues": float(np.abs(medias_despues).max()),
        "min_std_atributo_despues": float(stds_despues.min()),
        "max_std_atributo_despues": float(stds_despues.max()),
        "medias_por_atributo_despues": [float(v) for v in medias_despues],
        "stds_por_atributo_despues": [float(v) for v in stds_despues],
        # Forma de los arreglos guardados
        "forma_X_escalado": [int(X_escalado.shape[0]), int(X_escalado.shape[1])],
        "forma_y": [int(y.shape[0])],
        "archivo_datos": ARCHIVO_DATOS,
    }

    with open(ARCHIVO_RESULTADOS, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # 6) Impresión de las cifras principales (4 decimales)
    # ------------------------------------------------------------------
    print("=== T1: Carga y estandarizacion del conjunto Wine ===")
    print(f"Ejemplos: {n_ejemplos}")
    print(f"Atributos: {n_atributos}")
    print(f"Clases: {n_clases} -> {clases_reales}")
    print(f"Ejemplos por clase: {[conteo_por_clase[c] for c in clases_reales]}")
    print("Escalas antes de estandarizar:")
    print(f"  medias por atributo: [{medias_antes.min():.4f}, {medias_antes.max():.4f}]")
    print(f"  desv. est. por atributo: [{stds_antes.min():.4f}, {stds_antes.max():.4f}]")
    print("Verificacion despues de estandarizar:")
    print(f"  max |media| por atributo: {np.abs(medias_despues).max():.4f}")
    print(f"  desv. est. por atributo: [{stds_despues.min():.4f}, {stds_despues.max():.4f}]")
    print(f"X escalado (forma {X_escalado.shape}) y y guardados en '{ARCHIVO_DATOS}'")
    print(f"Cifras completas guardadas en '{ARCHIVO_RESULTADOS}'")


if __name__ == "__main__":
    main()
