"""La curva de aprendizaje medida de la Tarea A, para compararla con la que inventó el LLM en la 0.a.

    ../../../solver-v2/.venv/bin/python 0a_curva_medida.py

Protocolo del enunciado (Partes 1 y 3): partición 70/30 estratificada con random_state=42; submuestras
estratificadas del entrenamiento con random_state=42; siempre el mismo conjunto de prueba. NB sobre los
atributos crudos; regresión logística con el escalador ajustado solo sobre la submuestra.
Escribe 0a_curva_medida.json.
"""
import json
from pathlib import Path

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

curva = []
for frac in (0.05, 0.10, 0.25, 0.50, 1.00):
    if frac < 1:
        Xs, _, ys, _ = train_test_split(Xtr, ytr, train_size=frac, stratify=ytr, random_state=42)
    else:
        Xs, ys = Xtr, ytr
    nb = GaussianNB().fit(Xs, ys)
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(Xs, ys)
    curva.append({"fraccion": frac, "n": len(ys),
                  "nb_accuracy": round(accuracy_score(yte, nb.predict(Xte)), 4),
                  "lr_accuracy": round(accuracy_score(yte, lr.predict(Xte)), 4)})
    print(curva[-1])

Path(__file__).with_suffix(".json").write_text(json.dumps(curva, indent=2), encoding="utf-8")
