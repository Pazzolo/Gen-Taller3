# Tarea F — ¿Cuántos grupos hay en los vinos? K-means y silueta

MMIA 6013 · Reporte de una página (el código se adjunta aparte)

## Objetivo

Elegir cuántos grupos tiene el conjunto Wine usando solo un criterio interno, el coeficiente de silueta, sobre datos estandarizados. Las clases reales no participan en la decisión: se miran únicamente al final, para calcular el índice de Rand ajustado (ARI) del k elegido.

## Método

Cargué `load_wine` de scikit-learn: 178 ejemplos, 13 atributos y 3 clases (class_0: 59, class_1: 71, class_2: 48). Las escalas originales son muy dispares —la media de los atributos va de 0.3619 (nonflavanoid_phenols) a 746.8933 (proline), y su desviación estándar de 0.1241 a 314.0217—, así que apliqué `StandardScaler` y verifiqué que las medias quedaran en ~0 y las desviaciones en 1.0000. Sobre los datos escalados ajusté `KMeans(n_clusters=k, n_init=10, random_state=0)` para cada k de 2 a 8 y calculé el coeficiente de silueta; elegí el k con silueta más alta, sin usar las etiquetas reales. Con ese k calculé el ARI contra las clases.

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

**k elegido: 3** (silueta máxima 0.2849). El **ARI contra las clases reales es 0.8975**. La inercia cae monotónicamente de 1659.0080 (k=2) a 944.5589 (k=8), así que no sirve para elegir k. Los clusters con k=3 tienen 65, 51 y 62 ejemplos, y la tabla de contingencia muestra un acuerdo casi perfecto: el cluster 0 contiene 65 vinos, todos de class_1; el cluster 1 reúne los 48 de class_2 más 3 de class_1; y el cluster 2, los 59 de class_0 más otros 3 de class_1.

## Discusión

Sin estandarizar, las distancias euclídeas —y con ellas el error cuadrático que k-means minimiza y la propia silueta— quedarían dominadas por los atributos de escala grande: la desviación estándar de proline (314.0217) es más de tres órdenes de magnitud mayor que la de nonflavanoid_phenols (0.1241), y la de magnesium (14.2423) supera por mucho a la de hue (0.2279). Los grupos se decidirían casi solo por proline, mientras que atributos como flavanoids u od280/od315_of_diluted_wines casi no influirían.

Esperaría dos efectos (no ejecuté ese escenario, así que no reporto cifras de él). Primero, la silueta mediría la estructura de una o dos variables dominantes y el k con valor máximo probablemente cambiaría —con una variable dominante es plausible que favorezca k=2—, de modo que la elección dejaría de reflejar la estructura completa de los datos. Segundo, el ARI bajaría, porque clusters basados en una sola variable coinciden poco con clases definidas por todo el perfil químico. La sensibilidad al k se aprecia incluso con datos escalados: el ARI diagnóstico con k=2 fue 0.3887, muy por debajo del 0.8975 de k=3. Estandarizar da a los 13 atributos el mismo peso (desviaciones 1.0000 tras escalar) y es la razón de que k=3 recupere casi exactamente las tres clases.
