import json
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


def main():
    # ---------- Datos y división idéntica a la subtarea T1 ----------
    data = load_diabetes()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=7
    )

    # ---------- Pipeline StandardScaler + Ridge ----------
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge()),
    ])

    # ---------- Rejilla de alphas y esquema de validación cruzada ----------
    alphas = np.logspace(-3, 3, 13)
    param_grid = {"ridge__alpha": alphas}
    cv = KFold(n_splits=5, shuffle=True, random_state=7)

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="r2",
        cv=cv,
        return_train_score=False,
    )

    # Se ajusta SOLO con el conjunto de entrenamiento
    grid.fit(X_train, y_train)

    # ---------- Cifras a reportar ----------
    best_index = int(grid.best_index_)
    best_alpha = float(grid.best_params_["ridge__alpha"])
    best_cv_r2 = float(grid.best_score_)

    alphas_list = [float(a) for a in alphas]
    mean_scores = [float(s) for s in grid.cv_results_["mean_test_score"]]
    std_scores = [float(s) for s in grid.cv_results_["std_test_score"]]

    # R2 de cada fold para el alpha elegido
    split_keys = [
        k for k in grid.cv_results_
        if k.startswith("split") and k.endswith("_test_score")
    ]
    split_keys_sorted = sorted(
        split_keys, key=lambda s: int(s.split("_")[0].replace("split", ""))
    )
    best_fold_scores = [
        float(grid.cv_results_[k][best_index]) for k in split_keys_sorted
    ]

    resultados = {
        "subtarea": "T2_seleccion_alpha_validacion_cruzada",
        "dataset": "sklearn.datasets.load_diabetes",
        "n_ejemplos_entrenamiento": int(X_train.shape[0]),
        "n_ejemplos_prueba_reservados": int(X_test.shape[0]),
        "pipeline": "StandardScaler + Ridge",
        "rejilla_alpha": "numpy.logspace(-3, 3, 13)",
        "cv": "KFold(n_splits=5, shuffle=True, random_state=7)",
        "scoring": "r2",
        "n_alphas_evaluados": len(alphas_list),
        "alphas_evaluados": alphas_list,
        "r2_medio_cv_por_alpha": mean_scores,
        "r2_std_cv_por_alpha": std_scores,
        "indice_mejor_alpha": best_index,
        "alpha_elegido": best_alpha,
        "r2_medio_cv_alpha_elegido": best_cv_r2,
        "r2_std_cv_alpha_elegido": std_scores[best_index],
        "r2_por_fold_alpha_elegido": best_fold_scores,
    }

    # ---------- Guardar en resultados.json (conservando subtareas previas) ----------
    previo = {}
    try:
        with open("resultados.json", "r") as f:
            cargado = json.load(f)
        if isinstance(cargado, dict):
            previo = dict(cargado)
    except Exception:
        previo = {}

    if "resultados" in previo and isinstance(previo["resultados"], dict):
        nombre_previo = previo["resultados"].get("subtarea", "subtarea_anterior")
        previo[nombre_previo] = previo.pop("resultados")

    previo["resultados"] = resultados
    with open("resultados.json", "w") as f:
        json.dump(previo, f, indent=2)

    # ---------- Impresión de cifras principales (4 decimales) ----------
    print("=== T2: Selección de alpha por validación cruzada (solo entrenamiento) ===")
    print(f"Entrenamiento: {X_train.shape[0]} ejemplos | Prueba reservada (no usada aquí): {X_test.shape[0]}")
    print(f"Alphas evaluados: {len(alphas_list)} valores de numpy.logspace(-3, 3, 13)")
    print(f"CV: KFold(n_splits=5, shuffle=True, random_state=7) | scoring: r2")
    print("-" * 52)
    print(f"{'Alpha':>12} {'R2 medio CV':>14} {'R2 std CV':>12}")
    for a, m, s in zip(alphas_list, mean_scores, std_scores):
        print(f"{a:12.4f} {m:14.4f} {s:12.4f}")
    print("-" * 52)
    print(f"Alpha elegido: {best_alpha:.4f}")
    print(f"R2 medio de validación cruzada: {best_cv_r2:.4f}")
    print(f"Desv. est. del R2 en CV (alpha elegido): {std_scores[best_index]:.4f}")
    print("R2 por fold (alpha elegido): " + ", ".join(f"{s:.4f}" for s in best_fold_scores))
    print("Resultados guardados en resultados.json")


if __name__ == "__main__":
    main()
