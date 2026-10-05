# Tarea E — Regularización Ridge y validación cruzada sin tocar la prueba

**MMIA 6013 · Tarea propia (provisional) para el golden set del Taller 03 v2**
**Entrega:** un notebook de Jupyter (`.ipynb`) **ejecutado**, con las salidas visibles.

La regularización cambia el sesgo y la varianza de un modelo lineal, y su fuerza se elige con
validación cruzada. Esta tarea lo hace sobre datos reales sin que el conjunto de prueba influya en
ninguna decisión.

## Parte 1 — Datos

Usa el conjunto *Diabetes* que trae scikit-learn (`sklearn.datasets.load_diabetes`). Viene incluido
en la librería: no descargues ningún otro conjunto. Reporta cuántos ejemplos y cuántos atributos
tiene. Divide los datos en entrenamiento y prueba, 75 % y 25 %, con `random_state=7`. Esa división es
la única de la tarea, y el conjunto de prueba se usa una sola vez, al final de la Parte 3.

## Parte 2 — Selección de alpha por validación cruzada

Construye un pipeline con `StandardScaler` y `Ridge`. Elige `alpha` entre los 13 valores de
`numpy.logspace(-3, 3, 13)` con `GridSearchCV`, validación cruzada `KFold(n_splits=5, shuffle=True,
random_state=7)` y `scoring="r2"`, usando **solo el conjunto de entrenamiento**. Reporta el `alpha`
elegido y el R² medio de validación cruzada.

## Parte 3 — Evaluación en prueba

Con el `alpha` elegido, reporta el R² y el RMSE sobre el conjunto de prueba. Compáralos con una
regresión lineal sin regularizar (`LinearRegression`, con el mismo escalado) evaluada sobre la misma
prueba.

## Parte 4 — Figura

Dibuja el R² medio de validación cruzada contra `alpha`, con el eje de `alpha` en escala logarítmica.
La figura queda visible en el notebook.

## Parte 5 — Pregunta conceptual

Explica por qué elegir `alpha` mirando el R² del conjunto de prueba daría una estimación optimista del
desempeño, y qué diferencia hay entre el R² de validación cruzada y el de prueba en tus resultados.

## Formato

Un solo notebook, ejecutado de principio a fin sin errores, con una celda de Markdown que encabece
cada parte («Parte 1», «Parte 2»…). Las cifras se calculan en el notebook, no se escriben a mano.
