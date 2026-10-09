# Tarea F — Cuántos grupos hay en los vinos: k-means y silueta

## Objetivo

Elegir el número de grupos k del conjunto Wine usando solo un criterio interno, el coeficiente de silueta, sobre datos estandarizados; después, comparar el clustering con las clases reales mediante el índice de Rand ajustado (ARI). Las etiquetas reales no intervienen en la elección de k.

## Método

Cargué `load_wine` de scikit-learn: 178 ejemplos, 13 atributos y 3 clases (59, 71 y 48 ejemplos). Las escalas originales son muy dispares (medias entre 0.3619 y 746.8933; desviaciones entre 0.1241 y 314.0217), así que estandaricé los 13 atributos con `StandardScaler`, que dejó medias 0.0000 y desviaciones 1.0000. Para cada k de 2 a 8 ajusté `KMeans(n_clusters=k, n_init=10, random_state=0)` sobre los datos estandarizados y calculé la silueta; elegí el k de silueta máxima y, con ese k, calculé el ARI contra las clases reales. El código se entrega aparte.

## Resultados

| k | Silueta |
|---|---------|
| 2 | 0.2683 |
| 3 | **0.2849** |
| 4 | 0.2457 |
| 5 | 0.2026 |
| 6 | 0.1960 |
| 7 | 0.1386 |
| 8 | 0.1581 |
| **k elegido** | **3** |
| **ARI (k=3 vs. clases reales)** | **0.8975** |

El k con silueta más alta es 3 (0.2849). Con k=3, el ARI frente a las clases reales es 0.8975 y la pureza 0.9663. Los grupos quedan casi alineados con las clases: uno con 65 ejemplos, todos de la clase 1; otro con 48 de la clase 2 y 3 de la clase 1; y otro con 59 de la clase 0 y 3 de la clase 1. Como referencia, la silueta de las etiquetas reales es 0.2798, muy cercana a la del clustering.

## Discusión

Sin estandarizar, la distancia euclídea quedaría dominada por los atributos de gran escala: la desviación de proline es 314.0217 y la de magnesium 14.2423, frente a 0.1241 en el atributo de menor escala. Una diferencia típica en proline pesa en la distancia varios órdenes de magnitud más que una diferencia en los atributos pequeños, de modo que k-means agruparía esencialmente por proline (y en parte por magnesium) e ignoraría el resto de los 13 atributos. La silueta se mediría sobre esa geometría distorsionada, así que el k elegido probablemente cambiaría; es plausible que favoreciera k=2, con los datos partidos por la variable dominante. El ARI también bajaría, porque los grupos reflejarían la escala de proline y no el perfil químico completo que define a las clases. No ejecuté ese escenario, así que no reporto cifras para él. Lo que sí muestran los resultados es el contraste: con datos estandarizados la silueta es modesta (máximo 0.2849), pero el k=3 que ella elige coincide con las 3 clases reales (ARI 0.8975, pureza 0.9663); la estandarización es lo que permite que los 13 atributos contribuyan de forma comparable a la distancia.
