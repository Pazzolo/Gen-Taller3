# -*- coding: utf-8 -*-
"""
SUBTAREA T4: Figura matplotlib — R2 medio de validación cruzada (de T2) contra
alpha, con el eje x en escala logarítmica, visible como salida del notebook.

Qué hace este script:
1. Lee los resultados generados por T2 desde "resultados.json" (misma carpeta):
   alphas_evaluados, r2_medio_cv_por_alpha, alpha_elegido,
   r2_medio_cv_alpha_elegido, etc. NO recalcula la validación cruzada y NO usa
   valores incrustados de respaldo: si el archivo no existe o no contiene datos
   válidos de T2, el script aborta con un error.
2. Dibuja el R2 medio de CV contra alpha con ax.set_xscale("log"), marca el
   alpha elegido por T2, añade etiquetas, título, leyenda y rejilla.
3. Guarda la figura como "t4_r2_cv_vs_alpha.png" con fig.savefig(...) y la
   muestra con plt.show() (sin plt.close() previo) para que quede visible como
   salida de la celda del notebook.
4. Escribe "resultados.json" conservando los datos de T2 y añadiendo los
   campos de esta subtarea (figura, datos graficados, cifras principales).

El conjunto de prueba reservado por T2 (111 ejemplos) no se usa aquí para nada:
en esta subtarea no se ajusta ningún modelo ni se elige ningún hiperparámetro.
"""

import json
import math

import matplotlib.pyplot as plt

ARCHIVO_RESULTADOS = "resultados.json"
ARCHIVO_FIGURA = "t4_r2_cv_vs_alpha.png"


def buscar_datos_t2(obj):
    """Busca recursivamente el dict con los resultados de CV de T2."""
    candidatos = []

    def recorrer(o):
        if isinstance(o, dict):
            if ("alphas_evaluados" in o
                    and "r2_medio_cv_por_alpha" in o
                    and "alpha_elegido" in o
                    and "r2_medio_cv_alpha_elegido" in o):
                candidatos.append(o)
            for v in o.values():
                recorrer(v)
        elif isinstance(o, list):
            for v in o:
                recorrer(v)

    recorrer(obj)
    for d in candidatos:
        if str(d.get("subtarea", "")).startswith("T2"):
            return d
    if candidatos:
        return candidatos[0]
    return None


def validar_y_extraer_datos_t2(d):
    """Valida los datos de T2 y los devuelve como listas de float.

    Lanza RuntimeError si los datos no son utilizables (sin fallback).
    """
    try:
        alphas = [float(a) for a in d["alphas_evaluados"]]
        r2_medio = [float(v) for v in d["r2_medio_cv_por_alpha"]]
        alpha_elegido = float(d["alpha_elegido"])
        r2_medio_elegido = float(d["r2_medio_cv_alpha_elegido"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "resultados.json no contiene los campos esperados de T2 "
            "(alphas_evaluados, r2_medio_cv_por_alpha, alpha_elegido, "
            "r2_medio_cv_alpha_elegido): " + str(exc)
        )

    if len(alphas) != 13 or len(r2_medio) != 13:
        raise RuntimeError(
            "Se esperaban 13 alphas y 13 valores de R2 medio de CV (T2); "
            "encontrados: %d y %d" % (len(alphas), len(r2_medio))
        )
    if not all(math.isfinite(a) and a > 0 for a in alphas):
        raise RuntimeError("Los alphas de T2 deben ser positivos y finitos.")
    if not all(math.isfinite(v) for v in r2_medio):
        raise RuntimeError("Los R2 medios de CV de T2 contienen valores no finitos.")

    # Localizar el alpha elegido dentro de la rejilla y comprobar coherencia
    idx_mejor = None
    for i, a in enumerate(alphas):
        if math.isclose(a, alpha_elegido, rel_tol=1e-9, abs_tol=1e-12):
            idx_mejor = i
            break
    if idx_mejor is None:
        raise RuntimeError(
            "El alpha_elegido de T2 (%r) no pertenece a la rejilla "
            "alphas_evaluados." % alpha_elegido
        )
    if not math.isclose(r2_medio[idx_mejor], r2_medio_elegido,
                        rel_tol=1e-9, abs_tol=1e-12):
        raise RuntimeError(
            "Incoherencia en T2: r2_medio_cv_alpha_elegido no coincide con "
            "r2_medio_cv_por_alpha en el alpha elegido."
        )

    # Campos opcionales
    r2_std = None
    if "r2_std_cv_por_alpha" in d:
        try:
            candidata = [float(v) for v in d["r2_std_cv_por_alpha"]]
            if len(candidata) == 13 and all(math.isfinite(v) for v in candidata):
                r2_std = candidata
        except (TypeError, ValueError):
            r2_std = None

    r2_std_elegido = None
    if "r2_std_cv_alpha_elegido" in d:
        try:
            v = float(d["r2_std_cv_alpha_elegido"])
            if math.isfinite(v):
                r2_std_elegido = v
        except (TypeError, ValueError):
            r2_std_elegido = None

    r2_por_fold = None
    if "r2_por_fold_alpha_elegido" in d:
        try:
            candidata = [float(v) for v in d["r2_por_fold_alpha_elegido"]]
            if candidata and all(math.isfinite(v) for v in candidata):
                r2_por_fold = candidata
        except (TypeError, ValueError):
            r2_por_fold = None

    n_train = d.get("n_ejemplos_entrenamiento")
    n_test = d.get("n_ejemplos_prueba_reservados")

    return {
        "alphas": alphas,
        "r2_medio": r2_medio,
        "r2_std": r2_std,
        "alpha_elegido": alpha_elegido,
        "r2_medio_elegido": r2_medio_elegido,
        "r2_std_elegido": r2_std_elegido,
        "r2_por_fold": r2_por_fold,
        "idx_mejor": idx_mejor,
        "n_train": n_train,
        "n_test": n_test,
    }


def cargar_datos_t2():
    """Lee y valida los resultados de T2 desde resultados.json (sin fallback)."""
    try:
        with open(ARCHIVO_RESULTADOS, "r", encoding="utf-8") as f:
            contenido = json.load(f)
    except OSError:
        raise RuntimeError(
            "No se encontró '%s' con los resultados de T2 en la carpeta "
            "actual. Ejecuta primero la subtarea T2; esta subtarea no usa "
            "valores de respaldo." % ARCHIVO_RESULTADOS
        )
    except ValueError as exc:
        raise RuntimeError(
            "'%s' no es un JSON válido: %s" % (ARCHIVO_RESULTADOS, exc)
        )

    datos = buscar_datos_t2(contenido)
    if datos is None:
        raise RuntimeError(
            "'%s' no contiene los resultados de validación cruzada de T2 "
            "(faltan alphas_evaluados / r2_medio_cv_por_alpha / alpha_elegido)."
            % ARCHIVO_RESULTADOS
        )
    extraidos = validar_y_extraer_datos_t2(datos)
    return contenido, extraidos


def main():
    # ------------------------------------------------------------------
    # 1) Leer los resultados reales de T2 (sin valores incrustados)
    # ------------------------------------------------------------------
    contenido_previo, t2 = cargar_datos_t2()

    alphas = t2["alphas"]
    r2_medio = t2["r2_medio"]
    r2_std = t2["r2_std"]
    alpha_elegido = t2["alpha_elegido"]
    r2_medio_elegido = t2["r2_medio_elegido"]
    r2_std_elegido = t2["r2_std_elegido"]
    r2_por_fold = t2["r2_por_fold"]
    idx_mejor = t2["idx_mejor"]
    n_train = t2["n_train"]
    n_test = t2["n_test"]

    # ------------------------------------------------------------------
    # 2) Figura: R2 medio de CV (T2) contra alpha, eje x logarítmico
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(alphas, r2_medio, marker="o", color="tab:blue", linewidth=1.8,
            label="R2 medio de CV (5 folds, T2)")
    if r2_std is not None:
        ax.fill_between(alphas,
                        [m - s for m, s in zip(r2_medio, r2_std)],
                        [m + s for m, s in zip(r2_medio, r2_std)],
                        color="tab:blue", alpha=0.15,
                        label="± 1 desv. est. (CV)")
    ax.axvline(alpha_elegido, color="tab:red", linestyle="--", linewidth=1.2,
               label="alpha elegido (T2) = %.4g" % alpha_elegido)
    ax.plot([alpha_elegido], [r2_medio_elegido], marker="*", markersize=14,
            color="tab:red", linestyle="None",
            label="R2 medio CV = %.4f" % r2_medio_elegido)
    ax.set_xscale("log")
    ax.set_xlabel("alpha (escala logarítmica)")
    ax.set_ylabel("R2 medio de CV")
    ax.set_title("R2 medio de validación cruzada vs alpha (T2)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    # Guardar el PNG y mostrar la figura sin cerrarla antes, de modo que
    # quede visible como salida de la celda del notebook.
    fig.savefig(ARCHIVO_FIGURA, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    # ------------------------------------------------------------------
    # 3) Guardar en resultados.json (conservando los datos de T2)
    # ------------------------------------------------------------------
    salida = dict(contenido_previo) if isinstance(contenido_previo, dict) else {}
    salida["subtarea"] = "T4_figura_r2_medio_cv_vs_alpha"
    salida["descripcion"] = ("Figura: R2 medio de validacion cruzada (resultados "
                             "de T2) contra alpha, eje x en escala logaritmica")
    salida["fuente_datos"] = ("resultados.json de T2 (leido y validado, no "
                              "recalculado y sin valores de respaldo)")
    salida["figura_archivo"] = ARCHIVO_FIGURA
    salida["figura_eje_x_escala"] = "log"
    salida["figura_mostrada_con_plt_show"] = True
    salida["n_alphas_evaluados"] = len(alphas)
    salida["alphas_graficados"] = alphas
    salida["r2_medio_cv_graficados"] = r2_medio
    if r2_std is not None:
        salida["r2_std_cv_graficados"] = r2_std
    salida["indice_mejor_alpha"] = idx_mejor
    salida["alpha_elegido"] = alpha_elegido
    salida["r2_medio_cv_alpha_elegido"] = r2_medio_elegido
    if r2_std_elegido is not None:
        salida["r2_std_cv_alpha_elegido"] = r2_std_elegido
    if r2_por_fold is not None:
        salida["r2_por_fold_alpha_elegido"] = r2_por_fold
    if n_train is not None:
        salida["n_ejemplos_entrenamiento_T2"] = n_train
    if n_test is not None:
        salida["n_ejemplos_prueba_reservados_T2"] = n_test

    with open(ARCHIVO_RESULTADOS, "w", encoding="utf-8") as f:
        json.dump(salida, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # 4) Impresión de las cifras principales (4 decimales)
    # ------------------------------------------------------------------
    print("=== T4: Figura — R2 medio de validación cruzada (de T2) contra alpha ===")
    print("Fuente de los datos: resultados.json de T2 (leido y validado, sin respaldo)")
    if n_train is not None:
        linea_train = "Entrenamiento (T2): %s ejemplos" % n_train
        if n_test is not None:
            linea_train += " | Prueba reservada (T2, no usada aquí): %s" % n_test
        print(linea_train)
    print("Eje x de la figura en escala logarítmica (ax.set_xscale('log'))")
    print("-" * 56)
    print("%12s  %12s" % ("Alpha", "R2 medio CV"))
    for a, m in zip(alphas, r2_medio):
        print("%12.4f  %12.4f" % (a, m))
    print("-" * 56)
    print("Alpha elegido (T2): %.4f" % alpha_elegido)
    print("R2 medio de validación cruzada (T2): %.4f" % r2_medio_elegido)
    if r2_std_elegido is not None:
        print("Desv. est. del R2 en CV (T2): %.4f" % r2_std_elegido)
    if r2_por_fold is not None:
        print("R2 por fold (alpha elegido, T2): "
              + ", ".join("%.4f" % v for v in r2_por_fold))
    print("Figura guardada en: %s (y mostrada con plt.show())" % ARCHIVO_FIGURA)
    print("Resultados guardados en: %s" % ARCHIVO_RESULTADOS)


if __name__ == "__main__":
    main()
