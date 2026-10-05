# Tarea F — Cuántos grupos hay en los vinos: k-means y silueta

**MMIA 6013 · Tarea propia (provisional) para el golden set del Taller 03 v2**
**Entrega:** un reporte en **PDF** (`reporte.pdf`) de **una página** como máximo, con el código.

El agrupamiento no supervisado no usa etiquetas para decidir, pero se puede comparar después contra
ellas. Esta tarea elige el número de grupos solo con un criterio interno y recién al final mira las
clases reales.

## Parte 1 — Datos y escala

Usa el conjunto *Wine* de scikit-learn (`sklearn.datasets.load_wine`, incluido en la librería).
Reporta cuántos ejemplos, atributos y clases tiene. Los atributos están en escalas muy distintas:
**estandarízalos** con `StandardScaler` antes de agrupar.

## Parte 2 — Elección de k

Para cada k entre 2 y 8, ajusta `KMeans(n_clusters=k, n_init=10, random_state=0)` sobre los datos
estandarizados y calcula el coeficiente de silueta. Elige el k con la silueta más alta. Las etiquetas
reales **no** se usan para elegir k.

## Parte 3 — Comparación con las clases

Con el k elegido, compara los grupos con las clases reales mediante el índice de Rand ajustado (ARI).

## Parte 4 — Discusión

Explica qué pasaría con la elección de k y con el ARI si no se estandarizaran los atributos, y por qué.

## El reporte

En PDF, **una página** como máximo, con las secciones **Objetivo**, **Método**, **Resultados** (una
tabla con la silueta de cada k, el k elegido y el ARI) y **Discusión**. Toda cifra sale de la ejecución.
