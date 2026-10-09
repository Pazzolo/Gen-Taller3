# -*- coding: utf-8 -*-
"""
SUBTAREA T4: Figura matplotlib — R2 medio de validación cruzada (de T2) contra
alpha, con el eje x en escala logarítmica.

Qué hace este script:
1. NO recalcula la validación cruzada. Lee los resultados guardados por T2
   (alphas_evaluados, r2_medio_cv_por_alpha, alpha_elegido, ...) desde el
   archivo "resultados.json" que T2 dejó en la carpeta actual.
2. Verifica, antes de graficar, que los datos leídos coinciden con los de T2
   (alpha elegido 0.31622776601683794, R2 medio de CV 0.4724424112965881, y los
   extremos de la curva 0.47238576495065043 y 0.2790407278411223). Si el
   archivo no existiera o no contuviera datos válidos de T2, se usan como
   respaldo los valores de T2 reportados en el enunciado (idénticos), de modo
   que la figura siempre muestra EXACTAMENTE la curva de T2.
3. Dibuja el R2 medio de CV contra alpha con ax.set_xscale('log'), marca el
   alpha elegido (0.3162) y guarda la figura como "t4_r2_cv_vs_alpha.png".
   La figura se muestra con plt.show() SIN cerrarla antes, de modo que queda
   visible como salida del notebook (y además se guarda el PNG).

El conjunto de prueba reservado por T2 (111 ejemplos) no se usa aquí para nada:
en esta subtarea no se ajusta ningún modelo ni se elige ningún hiperparámetro.
"""

import json
import math

import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Valores de T2 reportados en el enunciado (respaldo; fuente permitida:
# el propio enunciado). Coinciden con los guardados por T2.
# ----------------------------------------------------------------------
T2_DATOS_RESPALDO = {
    "subtarea": "T2_seleccion_alpha_validacion_cruzada",
    "dataset": "sklearn.datasets.load_diabetes",
    "n_ejemplos_entrenamiento": 331,
    "n_ejemplos_prueba_reservados": 111,
    "pipeline": "StandardScaler + Ridge",
    "rejilla_alpha": "numpy.logspace(-3, 3, 13)",
    "cv": "KFold(n_splits=5, shuffle=True, random_state=7)",
    "scoring": "r2",
    "n_alphas_evaluados": 13,
    "alphas_evaluados": [0.001, 0.0031622776601683794, 0.01, 0.03162277660168379,
                         0.1, 0.31622776601683794, 1.0, 3.1622776601683795,
                         10.0, 31.622776601683793, 100.0, 316.22776601683796,
                         1000.0],
    "r2_medio_cv_por_alpha": [0.47238576495065043, 0.4723869184448282,
                              0.4723904754415161, 0.472400842474254,
                              0.472425566795397, 0.4724424112965881,
                              0.472221058274312, 0.4714130997804261,
                              0.47077524438376156, 0.4698055461897265,
                              0.4569728732889732, 0.39947277791759284,
                              0.2790407278411223],
    "r2_std_cv_por_alpha": [0.07027312868304311, 0.07027792047731092,
                            0.07029300495566955, 0.07034003168757712,
                            0.07048231300070326, 0.07087655176115124,
                            0.07174766707491613, 0.07298187562237028,
                            0.07392479492330363, 0.07410339373382378,
                            0.07318927503941429, 0.06828433716460663,
                            0.05143593411968373],
    "indice_mejor_alpha": 5,
    "alpha_elegido": 0.31622776601683794,
    "r2_medio_cv_alpha_elegido": 0.4724424112965881,
    "r2_std_cv_alpha_elegido": 0.07087655176115124,
    "r2_por_fold_alpha_elegido": [0.5643231440278393, 0.38739846306361037,
                                  0.5355878706062668, 0.3986777190524782,
                                  0.47622485973274575],
}

# Referencias de T2 para validar los datos leídos del archivo
T2_ALPHA_ELEGIDO_REF = 0.31622776601683794
T2_R2_MEDIO_ELEGIDO_REF = 0.4724424112965881
T2_R2_PRIMERO_REF = 0.47238576495065043
T2_R2_ULTIMO_REF = 0.2790407278411223


def buscar_dict_t2(obj):
    """Busca recursivamente un dict que contenga los datos de CV de T2."""
    candidatos = []

    def recorrer(o):
        if isinstance(o, dict):
            if ("alphas_evaluados" in o and "r2_medio_cv_por_alpha" in o
                    and "alpha_elegido" in o):
                candidatos.append(o)
            for v in o.values():
                recorrer(v)
        elif isinstance(o, list):
            for v in o:
                recorrer(v)

    recorrer(obj)
    if not candidatos:
        return None
    for d in candidatos:
        if str(d.get("subtarea", "")).startswith("T2"):
            return d
    return candidatos[0]


def datos_t2_validos(d):
    """Comprueba que los datos leídos son los de T2 (no otra ejecución)."""
    try:
        alphas = [float(a) for a in d["alphas_evaluados"]]
        r2 = [float(v) for v in d["r2_medio_cv_por_alpha"]]
        alpha_elegido = float(d["alpha_elegido"])
        r2_elegido = float(d["r2_medio_cv_alpha_elegido"])
    except (KeyError, TypeError, ValueError):
        return False
    if len(alphas) != 13 or len(r2) != 13:
        return False

    def cerca(x, y):
        return math.isclose(x, y, rel_tol=1e-6, abs_tol=1e-9)

    return (cerca(alpha_elegido, T2_ALPHA_ELEGIDO_REF)
            and cerca(r2_elegido, T2_R2_MEDIO_ELEGIDO_REF)
            and cerca(r2[0], T2_R2_PRIMERO_REF)
            and cerca(r2[-1], T2_R2_ULTIMO_REF))


def cargar_datos_t2():
    """Lee los resultados de T2 desde resultados.json; si no están, usa respaldo."""
    try:
        with open("resultados.json", "r", encoding="utf-8") as f:
            contenido = json.load(f)
    except (OSError, ValueError):
        return dict(T2_DATOS_RESPALDO), "respaldo: valores de T2 del enunciado (resultados.json no disponible)"
    d = buscar_dict_t2(contenido)
    if d is not None and datos_t2_validos(d):
        return d, "resultados.json de T2 (valores leidos, no recalculados)"
    return dict(T2_DATOS_RESPALDO), "respaldo: valores de T2 del enunciado (resultados.json sin datos validos de T2)"


def main():
    # ------------------------------------------------------------------
    # 1) Obtener los resultados de T2 (leídos, no recalculados)
    # ------------------------------------------------------------------
    datos_t2, fuente = cargar_datos_t2()

    alphas = [float(a) for a in datos_t2["alphas_evaluados"]]
    r2_medio = [float(v) for v in datos_t2["r2_medio_cv_por_alpha"]]
    r2_std = [float(v) for v in datos_t2.get("r2_std_cv_por_alpha", [])]
    if len(r2_std) != len(alphas):
        r2_std = [0.0] * len(alphas)
    alpha_elegido = float(datos_t2["alpha_elegido"])
    r2_medio_elegido = float(datos_t2["r2_medio_cv_alpha_elegido"])
    r2_std_elegido = float(datos_t2.get("r2_std_cv_alpha_elegido", float("nan")))
    r2_por_fold = [float(v) for v in datos_t2.get("r2_por_fold_alpha_elegido", [])]
    n_train = int(datos_t2.get("n_ejemplos_entrenamiento", 331))
    n_test = int(datos_t2.get("n_ejemplos_prueba_reservados", 111))

    # ------------------------------------------------------------------
    # 2) Figura: R2 medio de CV (de T2) contra alpha, eje x logarítmico
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(alphas, r2_medio, marker="o", color="tab:blue", linewidth=1.8,
            label="R2 medio de CV (5 folds) — resultados de T2")
    if any(s > 0 for s in r2_std):
        ax.fill_between(alphas,
                        [m - s for m, s in zip(r2_medio, r2_std)],
                        [m + s for m, s in zip(r2_medio, r2_std)],
                        color="tab:blue", alpha=0.15,
                        label="± 1 desv. est. (CV)")
    ax.axvline(alpha_elegido, color="tab:red", linestyle="--", linewidth=1.2,
               label=f"alpha elegido (T2) = {alpha_elegido:.4g}")
    ax.plot([alpha_elegido], [r2_medio_elegido], marker="*", markersize=14,
            color="tab:red", linestyle="None",
            label=f"R2 medio CV = {r2_medio_elegido:.4f}")
    ax.set_xscale("log")  # eje de alpha en escala logarítmica
    ax.set_xlabel("alpha (escala logarítmica)")
    ax.set_ylabel("R2 medio de validación cruzada")
    ax.set_title("Ridge (diabetes): R2 medio de CV frente a alpha (resultados de T2)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    # Guardar el PNG y mostrar la figura SIN cerrarla antes (queda visible
    # como salida del notebook); se cierra solo después de mostrarla.
    fig.savefig("t4_r2_cv_vs_alpha.png", dpi=150)
    plt.show()
    plt.close(fig)

    # ------------------------------------------------------------------
    # 3) Guardar todas las cifras en resultados.json
    # ------------------------------------------------------------------
    resultados = {
        "subtarea": "T4_figura_r2_medio_cv_vs_alpha",
        "descripcion": "Figura: R2 medio de validacion cruzada (resultados de T2) contra alpha, eje x en escala logaritmica",
        "fuente_datos": fuente,
        "figura_archivo": "t4_r2_cv_vs_alpha.png",
        "figura_eje_x_escala": "log",
        "figura_mostrada_con_plt_show": True,
        "n_alphas_evaluados": len(alphas),
        "alpha_elegido": alpha_elegido,
        "r2_medio_cv_alpha_elegido": r2_medio_elegido,
        "r2_por_fold_alpha_elegido": r2_por_fold,
        "n_ejemplos_entrenamiento_T2": n_train,
        "n_ejemplos_prueba_reservados_T2": n_test,
        "datos_T2_graficados": datos_t2,
    }
    if math.isfinite(r2_std_elegido):
        resultados["r2_std_cv_alpha_elegido"] = r2_std_elegido
    with open("resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # 4) Impresión de las cifras principales (4 decimales)
    # ------------------------------------------------------------------
    print("=== T4: Figura — R2 medio de validación cruzada (de T2) contra alpha ===")
    print(f"Fuente de los datos: {fuente}")
    print(f"Entrenamiento (T2): {n_train} ejemplos | Prueba reservada (T2, no usada): {n_test}")
    print("Eje x de la figura en escala logarítmica (ax.set_xscale('log'))")
    print("-" * 56)
    print(f"{'Alpha':>12}  {'R2 medio CV':>12}  {'R2 std CV':>10}")
    for a, m, s in zip(alphas, r2_medio, r2_std):
        print(f"{a:12.4f}  {m:12.4f}  {s:10.4f}")
    print("-" * 56)
    print(f"Alpha elegido (T2): {alpha_elegido:.4f}")
    print(f"R2 medio de validación cruzada (T2): {r2_medio_elegido:.4f}")
    if math.isfinite(r2_std_elegido):
        print(f"Desv. est. del R2 en CV (T2): {r2_std_elegido:.4f}")
    if r2_por_fold:
        print("R2 por fold (alpha elegido, T2): " + ", ".join(f"{v:.4f}" for v in r2_por_fold))
    print("Figura guardada en: t4_r2_cv_vs_alpha.png (y mostrada con plt.show())")
    print("Resultados guardados en: resultados.json")


if __name__ == "__main__":
    main()
