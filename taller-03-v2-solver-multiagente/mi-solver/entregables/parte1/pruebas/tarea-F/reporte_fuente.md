# Tarea F — Cuántos grupos hay en los vinos: k-means y silueta

## Objetivo

Elegir el número de grupos k en el conjunto Wine usando solo un criterio interno (coeficiente de silueta), sin usar las etiquetas, y comparar al final los grupos con las clases reales mediante el índice de Rand ajustado (ARI). Como extra, medir con un control qué cambia si no se estandarizan los atributos.

## Método

Cargué `load_wine` de scikit-learn: 178 ejemplos, 13 atributos y 3 clases (class_0: 59, class_1: 71, class_2: 48). Las escalas son muy distintas (medias por atributo entre 0.3619 y 746.8933; desviaciones estándar entre 0.1241 y 314.0217), así que apliqué `StandardScaler`; tras escalar, cada atributo quedó con media 0 (máximo |media| = 7.84e-15) y desviación 1. Para cada k de 2 a 8 ajusté `KMeans(n_clusters=k, n_init=10, random_state=0)` sobre los datos estandarizados y calculé la silueta; elegí el k con silueta más alta. Las clases reales solo entraron al final, para el ARI. El control repite el mismo barrido sobre los datos crudos, sin escalar. El código se entrega aparte.

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

El k elegido es 3 (silueta 0.2849), que coincide con el número de clases reales; los grupos tienen 65, 51 y 62 ejemplos. Contra las clases reales, el ARI es 0.8975 (Rand sin ajustar: 0.9543; pureza: 0.9663). En la tabla de contingencia, el grupo de 65 es solo class_1; el de 62 tiene 59 de class_0 y 3 de class_1; el de 51 tiene 48 de class_2 y 3 de class_1. Los únicos 6 errores son vinos de class_1 que caen en los otros dos grupos.

## Discusión

Sin estandarizar, los atributos con escalas grandes dominan la distancia euclídea: proline (media 746.8933, desv. 314.0217) y magnesium (99.7416, 14.2423) tienen desviaciones más de mil veces mayores que hue (0.2279) o nonflavanoid_phenols (0.1241), así que k-means agrupa casi solo por proline. El control lo confirma: en datos crudos la silueta máxima se mueve a k = 2 (0.6569, con grupos de 123 y 55) en vez de k = 3, y el ARI cae de 0.8975 a 0.3694 (delta −0.5281); incluso forzando k = 3 sobre datos crudos, el ARI es solo 0.3711. Llama la atención que la silueta cruda (0.6569) sea más alta que la escalada (0.2849), pero eso no indica un mejor agrupamiento: separar por proline produce grupos compactos en un espacio donde las demás variables casi no pesan. La silueta hereda el sesgo de la distancia con la que se calcula; si la escala sesga la distancia, sesga también el k elegido. Estandarizar da el mismo peso a los 13 atributos y recupera tanto el k correcto (3) como un ARI alto (0.8975).
