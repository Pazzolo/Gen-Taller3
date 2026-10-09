import json
import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler


def main():
    # ------------------------------------------------------------------
    # 1) Cargar el conjunto Wine de scikit-learn
    # ------------------------------------------------------------------
    wine = load_wine()
    X = wine.data
    y = wine.target
    feature_names = list(wine.feature_names)
    target_names = list(wine.target_names)

    n_ejemplos = int(X.shape[0])
    n_atributos = int(X.shape[1])
    n_clases = int(len(target_names))

    clases_unicas, conteos = np.unique(y, return_counts=True)
    ejemplos_por_clase = {
        target_names[int(c)]: float(n) for c, n in zip(clases_unicas, conteos)
    }

    # ------------------------------------------------------------------
    # 2) Escalas originales (para evidenciar la necesidad de estandarizar)
    # ------------------------------------------------------------------
    medias_orig = X.mean(axis=0)
    stds_orig = X.std(axis=0, ddof=0)

    # ------------------------------------------------------------------
    # 3) Estandarizar los atributos con StandardScaler
    #    (solo se ajusta con X; no hay fuga de información: no se usan etiquetas)
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X_esc = scaler.fit_transform(X)

    medias_esc = X_esc.mean(axis=0)
    stds_esc = X_esc.std(axis=0, ddof=0)

    # ------------------------------------------------------------------
    # 4) Guardar TODAS las cifras calculadas en resultados.json
    # ------------------------------------------------------------------
    resultados = {
        "n_ejemplos": float(n_ejemplos),
        "n_atributos": float(n_atributos),
        "n_clases": float(n_clases),
        "nombres_clases": target_names,
        "nombres_atributos": feature_names,
        "ejemplos_por_clase": ejemplos_por_clase,
        "media_min_atributos_originales": float(medias_orig.min()),
        "media_max_atributos_originales": float(medias_orig.max()),
        "std_min_atributos_originales": float(stds_orig.min()),
        "std_max_atributos_originales": float(stds_orig.max()),
        "media_max_abs_escalada": float(np.abs(medias_esc).max()),
        "std_min_escalado": float(stds_esc.min()),
        "std_max_escalado": float(stds_esc.max()),
        "medias_atributos_originales": [float(v) for v in medias_orig],
        "stds_atributos_originales": [float(v) for v in stds_orig],
        "medias_atributos_escalados": [float(v) for v in medias_esc],
        "stds_atributos_escalados": [float(v) for v in stds_esc],
    }

    with open("resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # 5) Guardar los datos escalados (y las etiquetas) para las siguientes partes
    # ------------------------------------------------------------------
    np.save("wine_X_escalado.npy", X_esc)
    np.save("wine_y.npy", y)

    # ------------------------------------------------------------------
    # 6) Reporte por consola (cifras principales con 4 decimales)
    # ------------------------------------------------------------------
    print("=== T1: Datos Wine y estandarizacion ===")
    print(f"Ejemplos: {n_ejemplos}")
    print(f"Atributos: {n_atributos}")
    print(f"Clases: {n_clases} -> {target_names}")
    for nombre in target_names:
        print(f"  {nombre}: {int(ejemplos_por_clase[nombre])} ejemplos")
    print("Escalas originales (por atributo):")
    print(f"  media min = {medias_orig.min():.4f} | media max = {medias_orig.max():.4f}")
    print(f"  std   min = {stds_orig.min():.4f} | std   max = {stds_orig.max():.4f}")
    print("Tras StandardScaler:")
    print(f"  max |media| = {np.abs(medias_esc).max():.4f}")
    print(f"  std min = {stds_esc.min():.4f} | std max = {stds_esc.max():.4f}")
    print("Archivos guardados: resultados.json, wine_X_escalado.npy, wine_y.npy")


if __name__ == "__main__":
    main()
