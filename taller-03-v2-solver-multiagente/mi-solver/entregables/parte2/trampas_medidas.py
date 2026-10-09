"""Qué cifras saca quien hace bien las Tareas E y F, y quien cae en cada trampa (2.a, «cómo se eligieron»).

    ../../../solver-v2/.venv/bin/python trampas_medidas.py

Los protocolos son los mismos del `verdad_py` de golden_tareas.json. Escribe trampas_medidas.json.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_diabetes, load_wine
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import adjusted_rand_score, mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

r4 = lambda v: round(float(v), 4)
out = {}

# Tarea E: Ridge sobre diabetes. Bien: alpha por validación cruzada en el entrenamiento.
X, y = load_diabetes(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=7)
grilla = np.logspace(-3, 3, 13)
gs = GridSearchCV(make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": grilla},
                  cv=KFold(5, shuffle=True, random_state=7), scoring="r2").fit(Xtr, ytr)
p = gs.predict(Xte)
out["E_bien"] = {"alpha": r4(gs.best_params_["ridge__alpha"]), "r2_prueba": r4(r2_score(yte, p)),
                 "rmse_prueba": r4(mean_squared_error(yte, p) ** 0.5)}
# Trampa: elegir alpha con el R² de la prueba.
r2_por_alpha = {a: r2_score(yte, make_pipeline(StandardScaler(), Ridge(alpha=a)).fit(Xtr, ytr).predict(Xte)) for a in grilla}
a_trampa = max(r2_por_alpha, key=r2_por_alpha.get)
out["E_trampa"] = {"alpha": r4(a_trampa), "r2_prueba": r4(r2_por_alpha[a_trampa])}
# Sin regularizar: para ver qué comprobación separa Ridge de MCO.
p = LinearRegression().fit(Xtr, ytr).predict(Xte)
out["E_sin_regularizar"] = {"r2_prueba": r4(r2_score(yte, p)), "rmse_prueba": r4(mean_squared_error(yte, p) ** 0.5)}

# Tarea F: k-means sobre wine, k por silueta.
X, y = load_wine(return_X_y=True)


def por_silueta(Z):
    eti = {k: KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Z) for k in range(2, 9)}
    sil = {k: silhouette_score(Z, e) for k, e in eti.items()}
    k = max(sil, key=sil.get)
    return k, sil[k], eti


Z = StandardScaler().fit_transform(X)
k, s, eti = por_silueta(Z)
out["F_bien"] = {"k": k, "silueta": r4(s), "ari": r4(adjusted_rand_score(y, eti[k]))}
k, s, eti_crudo = por_silueta(X)                      # trampa: sin estandarizar
out["F_trampa_sin_estandarizar"] = {"k": k, "silueta": r4(s), "ari": r4(adjusted_rand_score(y, eti_crudo[k]))}
ari = {k: adjusted_rand_score(y, e) for k, e in eti.items()}   # trampa descartada: k mirando las etiquetas
out["F_descartada_k_por_etiquetas"] = {"k": max(ari, key=ari.get)}

for k, v in out.items():
    print(k, v)
Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=2), encoding="utf-8")
