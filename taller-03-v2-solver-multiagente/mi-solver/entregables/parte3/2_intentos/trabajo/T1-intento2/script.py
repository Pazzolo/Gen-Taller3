import json
from sklearn.datasets import load_breast_cancer
from sklearn.naive_bayes import GaussianNB
X, y = load_breast_cancer(return_X_y=True)
acc = GaussianNB().fit(X, y).score(X, y)
json.dump({'nb_accuracy': acc}, open('resultados.json', 'w'))
