"""Validación del golden set sin solver (2.a): una solución correcta y una que cae en la trampa, por tarea.

    ../../../../solver-v2/.venv/bin/python construir.py
    cd ../../../../solver-v2 && .venv/bin/python evaluar_solver.py --golden ../mi-solver/golden_tareas.json \\
        --solo E F --solo-evaluar ../mi-solver/entregables/parte2/validacion/correcta \\
        --salida ../mi-solver/entregables/parte2/validacion/resultados_correcta.csv      (y lo mismo con trampa)

Escribe correcta/tarea-{E,F}/ y trampa/tarea-{E,F}/ con la forma que dejaría un solver: E es un notebook
ejecutado (nbclient); F es un reporte.pdf de una página más el resultados.json que lo respalda. Las dos
versiones de cada tarea son iguales salvo la trampa, para que lo que falle sea la trampa y nada más:
    E · trampa: alpha elegido con el R² de la prueba, no con validación cruzada
    F · trampa: k-means sobre los atributos sin estandarizar
"""
import json
import shutil
from pathlib import Path

import nbformat
import pymupdf
from nbclient import NotebookClient

AQUI = Path(__file__).resolve().parent

# ------------------------------------------------------------------------------------------- Tarea E
E_DATOS = """from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
X, y = load_diabetes(return_X_y=True)
print(f"{X.shape[0]} ejemplos, {X.shape[1]} atributos")
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=7)"""

E_ALPHA = {
    "correcta": """import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
grilla = np.logspace(-3, 3, 13)
gs = GridSearchCV(make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": grilla},
                  cv=KFold(n_splits=5, shuffle=True, random_state=7), scoring="r2").fit(Xtr, ytr)
alpha = gs.best_params_["ridge__alpha"]
r2_cv = gs.cv_results_["mean_test_score"]
print(f"alpha elegido: {alpha:.4f}")
print(f"R² medio de validación cruzada: {gs.best_score_:.4f}")
modelo = gs.best_estimator_""",
    # la trampa: se mira la prueba para elegir alpha
    "trampa": """import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
grilla = np.logspace(-3, 3, 13)
r2_prueba = [r2_score(yte, make_pipeline(StandardScaler(), Ridge(alpha=a)).fit(Xtr, ytr).predict(Xte)) for a in grilla]
alpha = grilla[int(np.argmax(r2_prueba))]
r2_cv = [cross_val_score(make_pipeline(StandardScaler(), Ridge(alpha=a)), Xtr, ytr, scoring="r2",
                         cv=KFold(n_splits=5, shuffle=True, random_state=7)).mean() for a in grilla]
print(f"alpha elegido: {alpha:.4f}")
print(f"R² medio de validación cruzada: {r2_cv[int(np.argmax(r2_prueba))]:.4f}")
modelo = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(Xtr, ytr)""",
}

E_PRUEBA = """from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
p = modelo.predict(Xte)
mco = make_pipeline(StandardScaler(), LinearRegression()).fit(Xtr, ytr).predict(Xte)
print(f"Ridge     · R² en prueba: {r2_score(yte, p):.4f} · RMSE: {mean_squared_error(yte, p) ** 0.5:.4f}")
print(f"Sin regularizar · R² en prueba: {r2_score(yte, mco):.4f} · RMSE: {mean_squared_error(yte, mco) ** 0.5:.4f}")"""

E_FIGURA = """%matplotlib inline
import matplotlib.pyplot as plt
plt.figure(figsize=(5, 3))
plt.semilogx(grilla, r2_cv, marker="o")
plt.xlabel("alpha"); plt.ylabel("R² medio de validación cruzada")
plt.show()"""

E_TEXTO = ("Elegir `alpha` con el R² de la prueba convierte a la prueba en un conjunto más de ajuste: el valor "
           "que se reporta es el mejor de 13 intentos sobre los mismos datos, así que sale optimista. La validación "
           "cruzada elige con el entrenamiento, y la prueba queda para una sola medición al final. La diferencia "
           "entre las dos cifras de R² de arriba mide cuánto cambia el desempeño al pasar de los pliegues a datos "
           "que el modelo no vio.")


def tarea_e(version: str, destino: Path) -> None:
    nb = nbformat.v4.new_notebook()
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    nb.cells = [md("## Parte 1 — Datos"), code(E_DATOS),
                md("## Parte 2 — Selección de alpha por validación cruzada"), code(E_ALPHA[version]),
                md("## Parte 3 — Evaluación en prueba"), code(E_PRUEBA),
                md("## Parte 4 — Figura"), code(E_FIGURA),
                md("## Parte 5 — Pregunta conceptual"), md(E_TEXTO)]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    NotebookClient(nb, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(destino)}}).execute()
    nbformat.write(nb, destino / "tarea_e.ipynb")


# ------------------------------------------------------------------------------------------- Tarea F
def tarea_f(version: str, destino: Path) -> None:
    from sklearn.cluster import KMeans
    from sklearn.datasets import load_wine
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    from sklearn.preprocessing import StandardScaler

    X, y = load_wine(return_X_y=True)
    Z = StandardScaler().fit_transform(X) if version == "correcta" else X   # la trampa: sin estandarizar
    eti = {k: KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Z) for k in range(2, 9)}
    sil = {k: round(float(silhouette_score(Z, e)), 4) for k, e in eti.items()}
    k = max(sil, key=sil.get)
    ari = round(float(adjusted_rand_score(y, eti[k])), 4)
    res = {"ejemplos": int(X.shape[0]), "atributos": int(X.shape[1]), "clases": int(len(set(y))),
           "silueta_por_k": sil, "k_elegido": k, "silueta_elegida": sil[k], "ari": ari}
    (destino / "resultados.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    escala = "estandarizados con <code>StandardScaler</code>" if version == "correcta" else "en su escala original"
    filas = "".join(f"<tr><td>{kk}</td><td>{s:.4f}</td></tr>" for kk, s in sil.items())
    html = f"""
<h1>Tarea F — k-means y silueta sobre Wine</h1>
<h2>Objetivo</h2><p>Elegir el número de grupos de Wine solo con la silueta y comparar después con las clases.</p>
<h2>Método</h2><p>{res['ejemplos']} ejemplos, {res['atributos']} atributos y {res['clases']} clases, {escala}.
KMeans(n_init=10, random_state=0) para k de 2 a 8; se elige el k de mayor silueta y luego se calcula el ARI.</p>
<h2>Resultados</h2>
<table border="1" cellpadding="2"><tr><th>k</th><th>silueta</th></tr>{filas}</table>
<p>k elegido: {k}. Silueta del k elegido: {sil[k]:.4f}. ARI contra las clases: {ari:.4f}.</p>
<h2>Discusión</h2><p>Sin estandarizar, la prolina (valores de cientos) domina la distancia y k-means agrupa casi
solo por ella; la silueta puede subir aunque los grupos se parezcan menos a las clases.</p>"""
    doc = pymupdf.open()
    pagina = doc.new_page()
    pagina.insert_htmlbox(pymupdf.Rect(50, 50, 545, 800), html, css="* {font-family: sans-serif; font-size: 10px}")
    doc.save(destino / "reporte.pdf")


if __name__ == "__main__":
    for version in ("correcta", "trampa"):
        for tarea, fn in (("E", tarea_e), ("F", tarea_f)):
            destino = AQUI / version / f"tarea-{tarea}"
            shutil.rmtree(destino, ignore_errors=True)
            destino.mkdir(parents=True)
            fn(version, destino)
            print(f"✓ {version}/tarea-{tarea}: {sorted(p.name for p in destino.iterdir())}")
