# Tarea A — Generativo contra discriminativo con pocos datos

**MMIA 6013 · Solver Taller 03 v2 · `reporte.md`**

## Introducción

Ng y Jordan (2002) predijeron que un clasificador **generativo** alcanza su error asintótico con muchos menos ejemplos que uno **discriminativo** (en la lectura clásica, O(log n) frente a O(n)), pero que ese error asintótico es mayor: la suposición estructural del generativo (independencia condicional) introduce sesgo, mientras que el discriminativo, con menos sesgo, sufre más varianza cuando los datos escasean. Esta tarea comprueba ese cruce en datos reales (Breast Cancer Wisconsin) comparando un Naive Bayes gaussiano con una regresión logística.

## Metodología

Se usa `load_breast_cancer`. Se hace **una única** división 70/30 estratificada (`random_state=42`); el conjunto de prueba no se usa para ajustar nada. Para la curva se submuestrea el entrenamiento de forma estratificada (5–100 %, misma semilla). El `StandardScaler` se ajusta solo con cada submuestra de entrenamiento; la regresión logística se entrena sobre atributos estandarizados y el NB sobre los crudos (un escalado monótono por atributo no altera sus predicciones). Todo se evalúa siempre sobre el mismo test.

```python
import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42)

def correr(frac):
    if frac < 1.0:
        idx, _ = train_test_split(np.arange(len(y_tr)), train_size=frac,
                                  stratify=y_tr, random_state=42)
    else:
        idx = np.arange(len(y_tr))
    esc = StandardScaler().fit(X_tr[idx])
    modelos = {"Naive Bayes gaussiano": GaussianNB().fit(X_tr[idx], y_tr[idx]),
               "Regresión logística": LogisticRegression(max_iter=5000)
                                      .fit(esc.transform(X_tr[idx]), y_tr[idx])}
    res = {}
    for nombre, m in modelos.items():
        Xe = X_te if "Bayes" in nombre else esc.transform(X_te)
        pred = m.predict(Xe)
        res[nombre] = (accuracy_score(y_te, pred),
                       f1_score(y_te, pred, average="macro"),
                       confusion_matrix(y_te, pred))
    return res

final = correr(1.0)                                   # Parte 2
fracs = [0.05, 0.10, 0.25, 0.50, 1.00]                # Parte 3
ns = [int(np.ceil(f * len(y_tr))) for f in fracs]
curva = {n: [correr(f)[n][0] for f in fracs]
         for n in ["Naive Bayes gaussiano", "Regresión logística"]}

plt.figure(figsize=(7, 4.5))
for n, v in curva.items(): plt.plot(ns, v, "o-", label=n)
plt.xscale("log"); plt.xticks(ns, ns)
plt.xlabel("Ejemplos de entrenamiento"); plt.ylabel("Exactitud en prueba")
plt.title("Generativo vs. discriminativo (Breast Cancer)")
plt.legend(); plt.grid(alpha=0.3)
plt.savefig("curva_aprendizaje.png", dpi=150, bbox_inches="tight")
```

## Resultados

**Parte 1 — Datos.** 569 ejemplos, 30 atributos; 212 malignos y 357 benignos. Entrenamiento: 398 (148 malignos, 250 benignos). Prueba: 171 (64 malignos, 107 benignos).

**Parte 2 — Modelos completos (test, n = 171).**

| Modelo | Exactitud | F1 macro |
|---|---|---|
| Naive Bayes gaussiano | 0.9415 | 0.9370 |
| Regresión logística | **0.9825** | **0.9813** |

**Parte 3 — Curva de aprendizaje** (exactitud, mismo test):

| n entrenamiento | 20 | 40 | 100 | 199 | 398 |
|---|---|---|---|---|---|
| NB gaussiano | 0.9123 | 0.9240 | 0.9298 | 0.9357 | 0.9415 |
| Reg. logística | 0.8889 | 0.9415 | 0.9649 | 0.9766 | 0.9825 |

![Curva de aprendizaje](curva_aprendizaje.png)

*Figura 1. Exactitud en prueba contra tamaño de entrenamiento. El NB parte arriba y crece poco (curva casi plana); la regresión logística parte abajo, lo supera entre 20 y 40 ejemplos y amplía la ventaja hasta 4.1 puntos con n = 398.*

## Discusión

El cruce predicho por Ng y Jordan **sí se observa**, y ocurre entre 20 y 40 ejemplos:

- **Pocos datos (n = 20): gana el generativo** (0.9123 vs 0.8889). Con 30 atributos y 20 observaciones, la regresión logística debe estimar una dirección conjunta en un espacio de dimensión casi igual al número de ejemplos: varianza altísima. El NB solo estima 30 medias y 30 varianzas por clase, una por atributo y de forma independiente; cada estimación es estable aun con poquísimos datos.
- **Desde n = 40: gana el discriminativo** y la brecha crece monótonamente (0.0175 en n = 40 → 0.0410 en n = 398). La regresión logística converge a un error asintótico menor porque su sesgo es menor: su frontera lineal conjunta explota combinaciones de atributos, mientras el NB paga su suposición de independencia condicional, claramente violada aquí (radio, perímetro y área medios correlacionan ≈ 0.99). Ese sesgo pone un piso a su desempeño: su curva sube solo 0.029 de n = 20 a n = 398, señal de que está cerca de su asíntota desde muy temprano — la convergencia rápida que predice la teoría.
- En términos sesgo–varianza: el NB es de **varianza baja y sesgo alto**; la regresión logística, de **sesgo bajo y varianza alta**, que la regularización L2 (C = 1) y los datos controlan. El cruce marca el punto donde la reducción de varianza del discriminativo supera la ventaja inicial del generativo.
- Nota clínica: con el modelo completo, el NB deja sin detectar 7 de 64 malignos (recall 0.891) frente a 2 de 64 de la regresión logística (recall 0.969); la diferencia de F1 macro (0.9370 vs 0.9813) refleja ese costo en la clase minoritaria.

## Conclusiones

Con una única división 70/30 y evaluación fija en prueba, el experimento reproduce el patrón de Ng y Jordan sobre datos reales: el Naive Bayes gaussiano es mejor con ~20 ejemplos y alcanza su asíntota (exactitud ≈ 0.94) casi de inmediato; la regresión logística lo supera a partir de ~40 ejemplos y llega a 0.9825 con los 398 disponibles. La lección práctica: con muy pocos datos conviene un generativo, robusto y barato de ajustar; en cuanto hay datos suficientes, el discriminativo, menos sesgado, gana.