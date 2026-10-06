# === Subtarea T4: figura del R2 medio de validación cruzada (de T2) contra alpha ===
# Recalcula de forma independiente la búsqueda de T2 (mismos datos, partición,
# pipeline, grid de alphas y esquema de CV) y dibuja el R2 medio de CV frente a
# alpha con el eje de alpha en escala logarítmica (semilogx).
# La figura se guarda como PNG y además queda visible como salida de la celda.

import json

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------
# 1) Datos y partición (idénticos a T2: 331 entrenamiento / 111 prueba)
#    El conjunto de prueba NO se usa aquí para nada.
# ------------------------------------------------------------------
datos = load_diabetes()
X, y = datos.data, datos.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=7, shuffle=True
)

# ------------------------------------------------------------------
# 2) Misma búsqueda que T2: StandardScaler -> Ridge, grid de 13 alphas,
#    KFold(5, shuffle=True, random_state=7), scoring='r2'
# ------------------------------------------------------------------
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge()),
])

alphas = np.logspace(-3, 3, 13)
param_grid = {"ridge__alpha": alphas}

cv = KFold(n_splits=5, shuffle=True, random_state=7)
busqueda = GridSearchCV(pipeline, param_grid, cv=cv, scoring="r2")
busqueda.fit(X_train, y_train)

alphas_eval = np.array(
    busqueda.cv_results_["param_ridge__alpha"], dtype=float
)
r2_medio = np.array(busqueda.cv_results_["mean_test_score"], dtype=float)
r2_std = np.array(busqueda.cv_results_["std_test_score"], dtype=float)

indice_mejor = int(np.argmax(r2_medio))
mejor_alpha = float(alphas_eval[indice_mejor])
mejor_r2 = float(r2_medio[indice_mejor])

# ------------------------------------------------------------------
# 3) Figura: R2 medio de CV contra alpha, eje x en escala logarítmica
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogx(
    alphas_eval,
    r2_medio,
    marker="o",
    color="tab:blue",
    linewidth=1.5,
    label="R2 medio de CV (5 folds)",
)
ax.axvline(
    mejor_alpha,
    color="tab:red",
    linestyle="--",
    linewidth=1.2,
    label=f"Mejor alpha = {mejor_alpha:.4g}",
)
ax.set_xlabel("alpha (escala logarítmica)")
ax.set_ylabel("R2 medio de validación cruzada")
ax.set_title("Ridge en diabetes: R2 medio de CV frente a alpha")
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="lower left")
fig.tight_layout()

# Guardar el PNG y mostrar la figura como salida de la celda del notebook
fig.savefig("t4_r2_vs_alpha.png", dpi=150)
display(fig)
plt.close(fig)

# ------------------------------------------------------------------
# 4) Guardar TODAS las cifras calculadas en resultados.json
#    (se conservan las claves de subtareas anteriores si el archivo existe)
# ------------------------------------------------------------------
try:
    with open("resultados.json", "r", encoding="utf-8") as f:
        resultados = json.load(f)
    if not isinstance(resultados, dict):
        resultados = {}
except (FileNotFoundError, json.JSONDecodeError):
    resultados = {}

resultados["t4_n_alphas_evaluados"] = int(len(alphas_eval))
resultados["t4_alphas_evaluados"] = [float(a) for a in alphas_eval]
resultados["t4_r2_medio_cv_por_alpha"] = [float(v) for v in r2_medio]
resultados["t4_r2_std_cv_por_alpha"] = [float(v) for v in r2_std]
resultados["t4_indice_mejor_alpha"] = indice_mejor
resultados["t4_alpha_elegido"] = mejor_alpha
resultados["t4_r2_medio_cv_mejor_alpha"] = mejor_r2
resultados["t4_n_splits_cv"] = 5
resultados["t4_shuffle_cv"] = True
resultados["t4_random_state_cv"] = 7
resultados["t4_scoring"] = "r2"
resultados["t4_n_ejemplos_entrenamiento"] = int(X_train.shape[0])
resultados["t4_n_ejemplos_prueba_reservados"] = int(X_test.shape[0])
resultados["t4_figura"] = "t4_r2_vs_alpha.png"

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

# ------------------------------------------------------------------
# 5) Impresión de las cifras principales con 4 decimales
# ------------------------------------------------------------------
print("=== Subtarea T4: figura R2 medio de CV frente a alpha ===")
print(f"Alphas evaluados: {len(alphas_eval)} (np.logspace(-3, 3, 13))")
print("R2 medio de validación cruzada por alpha:")
for a, m, s in zip(alphas_eval, r2_medio, r2_std):
    marca = "  <-- mejor" if float(a) == mejor_alpha else ""
    print(f"  alpha={a:10.4f} : R2 medio CV = {m:.4f} (+/- {s:.4f}){marca}")
print()
print(f"Mejor alpha: {mejor_alpha:.4f}")
print(f"R2 medio de CV en el mejor alpha: {mejor_r2:.4f}")
print("Figura guardada como t4_r2_vs_alpha.png y mostrada arriba")
print("Cifras guardadas en resultados.json")
